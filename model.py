# Model definitions

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from tqdm import tqdm
import pandas as pd

from utils import contact_f1, outer_concat, mat2bp

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
        self.resnet = ResNet2D(conv_dim, num_blocks, kernel_size)
        self.conv_out = nn.Conv2d(conv_dim, 1, kernel_size=kernel_size, padding="same")
        self.device = device
        self.class_weight = torch.tensor([negative_weight, 1.0]).float().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.to(device)

    def loss_func(self, yhat, y):
        """Calculate loss between predicted and ground truth contact maps"""
        y = y.view(y.shape[0], -1)
        yhat = yhat.view(yhat.shape[0], -1)
        yhat = yhat.unsqueeze(1)
        yhat = torch.cat((-yhat, yhat), dim=1)
        loss = cross_entropy(yhat, y, ignore_index=-1, weight=self.class_weight)
        return loss

    def forward(self, x):
        """Forward pass through the network"""
        x = self.linear_in(x)
        x = outer_concat(x, x)
        x = x.permute(0, 3, 1, 2)
        x = self.resnet(x)
        x = self.conv_out(x)
        x = x.squeeze(-3)
        x = torch.triu(x, diagonal=1)
        x = x + x.transpose(-1, -2)
        return x.squeeze(-1)

    def fit(self, loader):
        """Train the model for one epoch"""
        self.train()
        loss_acum = 0
        f1_acum = 0
        for batch in tqdm(loader):
            X = batch["seq_embs_pad"].to(self.device)
            y = batch["contacts"].to(self.device)
            y_pred = self(X)
            loss = self.loss_func(y_pred, y)
            loss_acum += loss.item()
            f1_acum += contact_f1(y.cpu(), y_pred.detach().cpu(), batch["Ls"], method="triangular")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        loss_acum /= len(loader)
        f1_acum /= len(loader)
        return {"loss": loss_acum, "f1": f1_acum}

    def test(self, loader):
        """Evaluate the model on a dataset"""
        self.eval()
        loss_acum = 0
        f1_acum = 0
        for batch in loader:
            X = batch["seq_embs_pad"].to(self.device)
            y = batch["contacts"].to(self.device)
            with torch.no_grad():
                y_pred = self(X)
                loss = self.loss_func(y_pred, y)
            loss_acum += loss.item()
            f1_acum += contact_f1(y.cpu(), y_pred.detach().cpu(), batch["Ls"], method="triangular")
            
        loss_acum /= len(loader)
        f1_acum /= len(loader)
        return {"loss": loss_acum, "f1": f1_acum}

    def pred(self, loader):
        """Make predictions on a dataset"""
        self.eval()
        predictions = []
        for batch in loader:
            Ls = batch["Ls"]
            seq_ids = batch["seq_ids"]
            sequences = batch["sequences"]
            X = batch["seq_embs_pad"].to(self.device)
            with torch.no_grad():
                y_pred = self(X)

            for k in range(len(y_pred)):
                predictions.append((
                    seq_ids[k],
                    sequences[k],
                    mat2bp(
                        y_pred[k, : Ls[k], : Ls[k]].squeeze().cpu()
                    )
                ))
        predictions = pd.DataFrame(predictions, columns=["id", "sequence", "base_pairs"])
        return predictions
