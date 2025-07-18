# Model definitions

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from tqdm import tqdm
import pandas as pd
import math

from utils import contact_f1, outer_concat, mat2bp, probing_f1

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

        self.probing_predictor_1 = nn.Sequential(
            *self.resnet1d,
            self.convrank1)

        self.probing_predictor_2 = nn.Sequential(
            nn.Linear(128, 32),  # Adjust max_sequence_length
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
          )        
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
        # 1D processing
        x_1d = self.linear_in(x)
        x_1d = x_1d.permute(0, 2, 1)
        embedding_1d = self.probing_predictor_1(x_1d) if return_probing else None
        embedding_1d = embedding_1d.permute(0, 2, 1)

        x_2d = outer_concat(embedding_1d, embedding_1d)
        x_2d = x_2d.permute(0, 3, 1, 2)
        x_2d = self.resnet(x_2d)

        # Probing prediction branch
        x_2d_mean = torch.mean(x_2d, 3) # columns
        x_2d_mean = x_2d_mean.permute(0, 2, 1)
        probing_pred = self.probing_predictor_2(x_2d_mean)

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

        for batch in tqdm(loader):
            X = batch["seq_embs_pad"].to(self.device)
            y = batch["contacts"].to(self.device)
            probing_target = batch["probings"].to(self.device)
            
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
