# Model definitions

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from tqdm import tqdm
import pandas as pd
import math

from utils import contact_f1, outer_concat, mat2bp, probing_f1

import torch
import gc

def get_tensor_memory_mb(tensor):
    """Get memory usage of a tensor in MB"""
    if tensor is None:
        return 0
    return tensor.element_size() * tensor.nelement() / (1024**2)

def print_gpu_memory_breakdown():
    """Print detailed GPU memory breakdown"""
    if not torch.cuda.is_available():
        return
    
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    
    print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    print(torch.cuda.memory_summary())
    # Print all tensors on GPU
    total_tensor_memory = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            size_mb = get_tensor_memory_mb(obj)
            # if size_mb > 10:  # Only show tensors > 10MB
            # print(f"  Tensor {obj.shape}: {size_mb:.1f} MB")
            # print(type(obj), obj.size())
            total_tensor_memory += size_mb
    
    print(f"Total tracked tensors: {total_tensor_memory:.1f} MB")

class ResidualLayer1D(nn.Module):
    def __init__(
        self,
        dilation,
        resnet_bottleneck_factor,
        filters,
        kernel_size,
    ):
        super().__init__()

        num_bottleneck_units = math.floor(resnet_bottleneck_factor * filters)

        self.layer = nn.Sequential(
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Conv1d(
                filters,
                num_bottleneck_units,
                kernel_size,
                dilation=dilation,
                padding="same",
            ),
            nn.BatchNorm1d(num_bottleneck_units),
            nn.ReLU(),
            nn.Conv1d(num_bottleneck_units, filters, kernel_size=1, padding="same"),
        )

    def forward(self, x):
        # print("FORWARD RESNET1D")
        out =  x + self.layer(x)
        # print(f"out RESNET1D: {out.shape}")
        return out

class ResNet2DBlock(nn.Module):
    def __init__(self, embed_dim, kernel_size=3, bias=False):
        super().__init__()

        # Bottleneck architecture
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, bias=bias, padding="same"),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        residual = x
        x = self.conv_net(x)
        x = x + residual
        return x

class ResNet2D(nn.Module):
    def __init__(self, embed_dim, num_blocks, kernel_size=3, bias=False):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResNet2DBlock(embed_dim, kernel_size, bias=bias) for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class SecondaryStructurePredictor(nn.Module):
    def __init__(
        self, embed_dim, num_blocks=2,
        conv_dim=64, kernel_size=3,
        negative_weight=0.1,
        device='cpu', lr=1e-5
    ):
        super().__init__()
        self.lr = lr
        self.threshold = 0.1
        self.linear_in = nn.Linear(embed_dim, (int)(conv_dim/2))

        kernel=3
        filters=16
        # embedding_dim=4
        num_layers=2
        dilation_resnet1d=3
        resnet_bottleneck_factor=0.5
        rank=64

        pad = (kernel - 1) // 2

        self.resnet1d = [nn.Conv1d((int)(conv_dim/2), filters, kernel, padding="same")]
        for k in range(num_layers):
            self.resnet1d.append(
                ResidualLayer1D(
                    dilation_resnet1d,
                    resnet_bottleneck_factor,
                    filters,
                    kernel,
                )
            )

        self.convrank1 = nn.Conv1d(
            in_channels=filters,
            out_channels=rank,
            kernel_size=kernel,
            padding=pad,
            stride=1,
        )

        self.embedding_learner_1d = nn.Sequential(
            *self.resnet1d,
            self.convrank1)

        self.probing_predictor = nn.Sequential( # conv aca
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
        )        
        # Lineal que toma las 32 features "representativas" y predice coneccion o no/residuos
        self.last_linear=nn.Linear(32, 1)
        self.resnet = ResNet2D(rank*2, num_blocks, kernel_size)
        self.conv_out = nn.Conv2d(conv_dim*2, 1, kernel_size=kernel_size, padding="same")
        self.device = device
        self.class_weight = torch.tensor([negative_weight, 1.0]).float().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.to(device)

    def loss_func(self, yhat, y, probing_pred=None, probing_target=None):
        """Calculate combined loss"""
        # Original contact map loss
        y = y.view(y.shape[0], -1)
        yhat = yhat.view(yhat.shape[0], -1)
        yhat = yhat.unsqueeze(1)
        yhat = torch.cat((-yhat, yhat), dim=1)
        contact_loss = cross_entropy(yhat, y, ignore_index=-1, weight=self.class_weight)
        
        # Probing prediction loss (if provided)
        probing_loss = 0
        if probing_pred is not None and probing_target is not None:
            probing_loss = nn.MSELoss()(probing_pred.squeeze(), probing_target.squeeze())
        
        return contact_loss, probing_loss  # You can adjust the weighting
        # revisar comportamiento de las loss por separado, interesa ver que pasa con cada uno

    def forward(self, x, return_probing=False):
        """Forward pass through the network"""
        # print("\n=== MEMORY BEFORE FORWARD ===")
        # print_gpu_memory_breakdown()
        
        # 1D processing
        x_1d = self.linear_in(x)
        # print(f"After linear_in: x_1d {x_1d.shape} = {get_tensor_memory_mb(x_1d):.1f} MB")
        
        x_1d = x_1d.permute(0, 2, 1)
        # print(f"After permute: x_1d {x_1d.shape} = {get_tensor_memory_mb(x_1d):.1f} MB")
        
        embedding_1d = self.embedding_learner_1d(x_1d) if return_probing else None
        # print(f"After embedding_learner_1d: {embedding_1d.shape} = {get_tensor_memory_mb(embedding_1d):.1f} MB")
        embedding_1d = embedding_1d.permute(0, 2, 1)
        # print(f"After permute: embedding_1d {embedding_1d.shape} = {get_tensor_memory_mb(embedding_1d):.1f} MB")
        

        # print("\n=== BEFORE OUTER_CONCAT ===")
        # print_gpu_memory_breakdown()
        x_2d = outer_concat(embedding_1d, embedding_1d)
        # print(f"After outer_concat: x_2d {x_2d.shape} = {get_tensor_memory_mb(x_2d):.1f} MB")

        # print("\n=== BEFORE RESNET ===")
        # print_gpu_memory_breakdown()
        x_2d = x_2d.permute(0, 3, 1, 2)
        # print(f"After permute to NCHW: x_2d {x_2d.shape} = {get_tensor_memory_mb(x_2d):.1f} MB")
        
        # Break down ResNet2D forward pass block by block
        for block_idx, block in enumerate(self.resnet.blocks):
            # print(f"\n=== RESNET BLOCK {block_idx} START ===")
            # print_gpu_memory_breakdown()
            
            # Store residual
            residual = x_2d
            # print(f"Residual stored: {residual.shape} = {get_tensor_memory_mb(residual):.1f} MB")
            
            # Break down ResNet2DBlock conv_net operations
            conv_layers = list(block.conv_net.children())
            
            # First conv + norm + relu
            x_2d = conv_layers[0](x_2d)  # Conv2d 1x1
            # print(f"After 1x1 conv: x_2d {x_2d.shape} = {get_tensor_memory_mb(x_2d):.1f} MB")
            x_2d = conv_layers[1](x_2d)  # InstanceNorm2d
            # print(f"After norm1: x_2d {x_2d.shape} = {get_tensor_memory_mb(x_2d):.1f} MB")
            x_2d = conv_layers[2](x_2d)  # ReLU
            # print(f"After relu1: x_2d {x_2d.shape} = {get_tensor_memory_mb(x_2d):.1f} MB")
            # print_gpu_memory_breakdown()

            # Second conv + norm + relu  
            x_2d = conv_layers[3](x_2d)  # Conv2d 3x3
            # print(f"After 3x3 conv: x_2d {x_2d.shape} = {get_tensor_memory_mb(x_2d):.1f} MB")
            # print_gpu_memory_breakdown()
            x_2d = conv_layers[4](x_2d)  # InstanceNorm2d
            # print(f"After norm2: x_2d {x_2d.shape} = {get_tensor_memory_mb(x_2d):.1f} MB")
            x_2d = conv_layers[5](x_2d)  # ReLU
            # print(f"After relu2: x_2d {x_2d.shape} = {get_tensor_memory_mb(x_2d):.1f} MB")
            # print_gpu_memory_breakdown()

            # Third conv + norm + relu
            x_2d = conv_layers[6](x_2d)  # Conv2d 1x1
            # print(f"After final 1x1 conv: x_2d {x_2d.shape} = {get_tensor_memory_mb(x_2d):.1f} MB")
            x_2d = conv_layers[7](x_2d)  # InstanceNorm2d  
            # print(f"After norm3: x_2d {x_2d.shape} = {get_tensor_memory_mb(x_2d):.1f} MB")
            x_2d = conv_layers[8](x_2d)  # ReLU
            # print(f"After relu3: x_2d {x_2d.shape} = {get_tensor_memory_mb(x_2d):.1f} MB")
            # print_gpu_memory_breakdown()
            
            # Residual connection
            x_2d = x_2d + residual
            # print(f"After residual add: x_2d {x_2d.shape} = {get_tensor_memory_mb(x_2d):.1f} MB")
            
            # print(f"\n=== RESNET BLOCK {block_idx} END ===")
            # print_gpu_memory_breakdown()
        
        # print(f"\n=== AFTER ALL RESNET BLOCKS ===")
        # print(f"Final resnet output: x_2d {x_2d.shape} = {get_tensor_memory_mb(x_2d):.1f} MB")
    
        # Probing prediction branch
        x_2d_mean = torch.mean(x_2d, 2) # rows
        probing_pred = self.probing_predictor(x_2d_mean)
        probing_pred = probing_pred.permute(0, 2, 1)
        probing_pred = self.last_linear(probing_pred)
        probing_pred = torch.sigmoid(probing_pred)

        # Contact prediction branch
        x_2d = self.conv_out(x_2d)
        x_2d = x_2d.squeeze(-3)
        x_2d = torch.triu(x_2d, diagonal=1)
        x_2d = x_2d + x_2d.transpose(-1, -2)
        
        if return_probing:
            return x_2d.squeeze(-1), probing_pred
        return x_2d.squeeze(-1)

    def fit(self, loader):
        """Train the model for one epoch"""
        self.train()
        loss_acum = 0
        f1_acum = 0
        contact_loss_acum = 0
        probing_loss_acum = 0
        f1_probing_acum = 0
        # X=torch.zeros(64, 510, 4)
        # y=-torch.ones((64, 510, 510), dtype=torch.long)
        # probing_target=torch.zeros(128, 510)
        for batch in tqdm(loader):
            X = batch["seq_embs_pad"].to(self.device)
            y = batch["contacts"].to(self.device)
            probing_target = batch["probings"].to(self.device)
            # print(f"Batch data:")
            # print(f"  X {X.shape}: {get_tensor_memory_mb(X):.1f} MB")
            # print(f"  y {y.shape}: {get_tensor_memory_mb(y):.1f} MB") 
            # print(f"  probing_target {probing_target.shape}: {get_tensor_memory_mb(probing_target):.1f} MB")
        
            # Forward pass with probing prediction
            y_pred, probing_pred = self(X, return_probing=True)
            
            contact_loss, probing_loss = self.loss_func(y_pred, y, probing_pred, probing_target)
            # Combine losses before backward
            total_loss = contact_loss + probing_loss
        
            loss_acum += total_loss.item()
            contact_loss_acum += contact_loss.item()
            probing_loss_acum += probing_loss.item()

            f1_acum += contact_f1(y.cpu(), y_pred.detach().cpu(), batch["Ls"], method="triangular")
            f1_probing_acum += probing_f1(probing_target.cpu(), probing_pred.detach().cpu())
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
        loss_acum /= len(loader)
        contact_loss_acum /= len(loader)
        probing_loss_acum /= len(loader)
        f1_acum /= len(loader)
        f1_probing_acum /= len(loader)

        return {
            "loss": loss_acum,
            "f1": f1_acum,
            "contact_loss": contact_loss_acum,
            "probing_loss": probing_loss_acum,
            "f1_probing": f1_probing_acum
        }

    def test(self, loader):
        """Evaluate the model on a dataset"""
        self.eval()
        loss_acum = 0
        f1_acum = 0
        contact_loss_acum = 0
        probing_loss_acum = 0
        f1_probing_acum = 0
        # X=torch.zeros(BATCH_SIZE, 510, 4)
        # y=-torch.ones((BATCH_SIZE, 510, 510), dtype=torch.long)
        # probing_target=torch.zeros(BATCH_SIZE, 510)
        for batch in loader:
            X = batch["seq_embs_pad"].to(self.device)
            y = batch["contacts"].to(self.device)
            probing_target = batch["probings"].to(self.device)

            with torch.no_grad():
                y_pred, probing_pred = self(X, return_probing=True)
                contact_loss, probing_loss = self.loss_func(y_pred, y, probing_pred, probing_target)
                total_loss = contact_loss + probing_loss
        
            loss_acum += total_loss.item()
            contact_loss_acum += contact_loss.item()
            probing_loss_acum += probing_loss.item()
            
            f1_acum += contact_f1(y.cpu(), y_pred.detach().cpu(), batch["Ls"], method="triangular")
            f1_probing_acum += probing_f1(probing_target.cpu(), probing_pred.detach().cpu())

        loss_acum /= len(loader)
        f1_acum /= len(loader)
        contact_loss_acum /= len(loader)
        probing_loss_acum /= len(loader)
        f1_probing_acum /= len(loader)

        return {
            "loss": loss_acum,
            "f1": f1_acum,
            "contact_loss": contact_loss_acum,
            "probing_loss": probing_loss_acum,
            "f1_probing": f1_probing_acum
        }

    # def pred(self, loader):
    #     """Make predictions on a dataset"""
    #     self.eval()
    #     predictions = []
    #     for batch in loader:
    #         Ls = batch["Ls"]
    #         seq_ids = batch["seq_ids"]
    #         sequences = batch["sequences"]
    #         X = batch["seq_embs_pad"].to(self.device)
    #         with torch.no_grad():
    #             y_pred = self(X)

    #         for k in range(len(y_pred)):
    #             predictions.append((
    #                 seq_ids[k],
    #                 sequences[k],
    #                 mat2bp(
    #                     y_pred[k, : Ls[k], : Ls[k]].squeeze().cpu()
    #                 )
    #             ))
    #     predictions = pd.DataFrame(predictions, columns=["id", "sequence", "base_pairs"])
    #     return predictions
