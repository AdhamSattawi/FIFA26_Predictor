"""
baseline_mlp.py — Model 1: Naive Average MLP (Baseline)

Architecture:
  1. Average all 11 player vectors → single team representation (loses player variance)
  2. Concatenate home_repr + away_repr + context
  3. 4-layer MLP with BatchNorm + Dropout

This is the baseline. Its weakness: a team with one superstar and
10 average players looks identical to an evenly talented team.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


class BaselineMLP(nn.Module):
    """
    Naive Average MLP for match outcome prediction.

    Input:
      home_players: (B, 11, F)
      away_players: (B, 11, F)
      context:      (B, C)

    Output: (B, 3) — raw logits for [Home Win, Draw, Away Win]
    """

    def __init__(self,
                 F: int = config.F,
                 C: int = config.C,
                 hidden_dims: list[int] = None,
                 dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        input_dim = F * 2 + C  # home_repr + away_repr + context

        layers = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, 3))  # output logits

        self.net = nn.Sequential(*layers)

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
        # Naive averaging — the key limitation of this model
        home_repr = home_players.mean(dim=1)   # (B, F)
        away_repr = away_players.mean(dim=1)   # (B, F)

        x = torch.cat([home_repr, away_repr, context], dim=1)  # (B, 2F+C)
        return self.net(x)


if __name__ == "__main__":
    # Quick sanity check
    model = BaselineMLP()
    B = 4
    home = torch.randn(B, config.N_PLAYERS, config.F)
    away = torch.randn(B, config.N_PLAYERS, config.F)
    ctx  = torch.randn(B, config.C)
    out  = model(home, away, ctx)
    print(f"BaselineMLP output shape: {out.shape}")  # (4, 3)
    assert out.shape == (B, 3)
    print("✓ BaselineMLP OK")
