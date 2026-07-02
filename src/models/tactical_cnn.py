"""
tactical_cnn.py — Model 2: Tactical Synergy 1D CNN

Architecture:
  1. Keep full (11, F) player matrix — no averaging
  2. Shared Team Encoder: 3× Conv1d layers scan adjacent position groups,
     learning tactical synergies (e.g., RB–CM–RW overlap runs)
  3. GlobalAveragePool → team representation vector
  4. Concatenate home + away + context → MLP head

The key improvement over baseline: the CNN sees *interactions* between
adjacent positions (CB–CB chemistry, CDM–CM coordination, etc.) rather
than treating each player as independent.

Weight sharing between home/away encoders prevents learning home/away bias.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


class TeamEncoder(nn.Module):
    """
    Shared 1D CNN encoder for a single team's player matrix.

    Input:  (B, 11, F) — 11 players × F features, ordered by canonical position
    Output: (B, out_channels) — team representation vector
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int = 64,
                 dropout: float = 0.25):
        super().__init__()

        # Input: (B, F, 11) after transpose — channels=F, length=11 positions
        # kernel_size=3: captures 3 adjacent positions (e.g., LB-CB1-CB2 or CM-CAM-ST)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 11, F) — ordered players
        Returns: (B, out_channels) — team representation
        """
        x = x.transpose(1, 2)      # (B, F, 11) — for Conv1d: (B, channels, length)
        x = self.encoder(x)         # (B, out_channels, 11)
        x = x.mean(dim=2)          # Global average pooling: (B, out_channels)
        return self.dropout(x)


class TacticalCNN(nn.Module):
    """
    Tactical Synergy 1D CNN for match outcome prediction.

    Input:
      home_players: (B, 11, F)
      away_players: (B, 11, F)
      context:      (B, C)

    Output: (B, 3) — raw logits for [Home Win, Draw, Away Win]
    """

    def __init__(self,
                 F: int = config.F,
                 C: int = config.C,
                 encoder_out: int = 64,
                 dropout: float = 0.3):
        super().__init__()

        # Single shared encoder — same weights for home and away
        self.team_encoder = TeamEncoder(in_channels=F, out_channels=encoder_out)

        match_input_dim = encoder_out * 2 + C  # home_repr + away_repr + context

        self.match_head = nn.Sequential(
            nn.Linear(match_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 3),  # output logits
        )

    def forward(self,
                home_players: torch.Tensor,
                away_players: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        """
        home_players: (B, 11, F)
        away_players: (B, 11, F)
        context:      (B, C)
        Returns:      (B, 3) logits
        """
        # Weight-shared encoding — same CNN for both teams
        home_repr = self.team_encoder(home_players)  # (B, encoder_out)
        away_repr = self.team_encoder(away_players)  # (B, encoder_out)

        x = torch.cat([home_repr, away_repr, context], dim=1)  # (B, 2*enc + C)
        return self.match_head(x)


if __name__ == "__main__":
    # Quick sanity check
    model = TacticalCNN()
    B = 4
    home = torch.randn(B, config.N_PLAYERS, config.F)
    away = torch.randn(B, config.N_PLAYERS, config.F)
    ctx  = torch.randn(B, config.C)
    out  = model(home, away, ctx)
    print(f"TacticalCNN output shape: {out.shape}")  # (4, 3)
    assert out.shape == (B, 3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print("✓ TacticalCNN OK")
