"""
attention_cnn.py — Model 3: Attention CNN with Regularization

Architecture:
  1. Same Conv1d encoder as Model 2 (feature extraction)
  2. MultiheadAttention: learns which positions matter most
  3. Attention-weighted pooling (instead of GlobalAvgPool): learns α_i per player
  4. Class-weighted CrossEntropyLoss to force draw prediction
  5. Dropout + weight_decay for strong regularization

Key innovations vs. Model 2:
  - The model learns which PLAYERS carry predictive signal (not just patterns)
  - The goalkeeper may be irrelevant for goal scoring; the striker is central
  - Draw class gets extra weight in the loss function
  - The attention weights α_i are extractable for analysis (which positions matter?)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over the 11 player positions.
    Computes α_i = softmax(W · h_i) then weighted sum.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn_score = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, 11, embed_dim)
        Returns:
          pooled:  (B, embed_dim) — attention-weighted team representation
          weights: (B, 11)        — attention weights (interpretable!)
        """
        scores  = self.attn_score(x).squeeze(-1)      # (B, 11)
        weights = torch.softmax(scores, dim=1)         # (B, 11) — sum to 1
        pooled  = (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, embed_dim)
        return pooled, weights


class AttentionTeamEncoder(nn.Module):
    """
    Team encoder with CNN feature extraction + self-attention + attention pooling.

    Input:  (B, 11, F) — players ordered by canonical position
    Output:
      repr:    (B, out_dim) — team representation
      weights: (B, 11)     — attention weights per player position
    """

    def __init__(self,
                 in_channels: int,
                 cnn_out: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()

        # CNN feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, cnn_out, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out),
            nn.ReLU(),
        )

        # Self-attention over the 11 position slots
        self.self_attn   = nn.MultiheadAttention(
            embed_dim=cnn_out, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm  = nn.LayerNorm(cnn_out)
        self.attn_dropout = nn.Dropout(dropout)

        # Attention-weighted pooling
        self.pool = AttentionPooling(cnn_out)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, 11, F)
        Returns: (repr (B, cnn_out), weights (B, 11))
        """
        # CNN: (B, F, 11) → (B, cnn_out, 11)
        h = self.cnn(x.transpose(1, 2))
        h = h.transpose(1, 2)  # (B, 11, cnn_out) — back to seq format

        # Self-attention with residual + LayerNorm
        attn_out, _ = self.self_attn(h, h, h)        # (B, 11, cnn_out)
        h = self.layer_norm(h + self.attn_dropout(attn_out))

        # Attention-weighted pooling
        repr_, weights = self.pool(h)    # (B, cnn_out), (B, 11)
        return repr_, weights


class AttentionCNN(nn.Module):
    """
    Attention CNN for match outcome prediction.

    Input:
      home_players: (B, 11, F)
      away_players: (B, 11, F)
      context:      (B, C)

    Output: (B, 3) — raw logits for [Home Win, Draw, Away Win]

    Also exposes:
      home_attn_weights: (B, 11) — attention per player in home team
      away_attn_weights: (B, 11) — attention per player in away team
    """

    def __init__(self,
                 F: int = config.F,
                 C: int = config.C,
                 cnn_out: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()

        # Single shared encoder for home and away (weight sharing)
        self.team_encoder = AttentionTeamEncoder(
            in_channels=F, cnn_out=cnn_out, n_heads=n_heads, dropout=dropout
        )

        match_input_dim = cnn_out * 2 + C

        self.match_head = nn.Sequential(
            nn.Linear(match_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),   # was hardcoded 0.4 — now uses dropout param

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 3),  # output logits
        )

        # Store attention weights for analysis (populated on each forward pass)
        self.home_attn_weights: torch.Tensor | None = None
        self.away_attn_weights: torch.Tensor | None = None

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
        home_repr, home_w = self.team_encoder(home_players)  # (B, cnn_out), (B, 11)
        away_repr, away_w = self.team_encoder(away_players)

        # Store for external analysis
        self.home_attn_weights = home_w.detach()
        self.away_attn_weights = away_w.detach()

        x = torch.cat([home_repr, away_repr, context], dim=1)
        return self.match_head(x)

    def get_attention_weights(self) -> dict[str, torch.Tensor]:
        """Return the attention weights from the last forward pass."""
        return {
            "home": self.home_attn_weights,
            "away": self.away_attn_weights,
        }


def compute_class_weights(targets: torch.Tensor) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequency.
    w_c = N_total / (n_classes × N_c)

    Draw class will typically get the highest weight.
    """
    n_classes = 3
    n_total   = len(targets)
    weights   = torch.zeros(n_classes)
    for c in range(n_classes):
        n_c = (targets == c).sum().item()
        weights[c] = n_total / (n_classes * max(n_c, 1))
    return weights


if __name__ == "__main__":
    # Quick sanity check
    model = AttentionCNN()
    B = 4
    home = torch.randn(B, config.N_PLAYERS, config.F)
    away = torch.randn(B, config.N_PLAYERS, config.F)
    ctx  = torch.randn(B, config.C)
    out  = model(home, away, ctx)
    print(f"AttentionCNN output shape: {out.shape}")  # (4, 3)
    assert out.shape == (B, 3)

    # Check attention weights
    attn = model.get_attention_weights()
    print(f"Home attention weights shape: {attn['home'].shape}")  # (4, 11)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print("✓ AttentionCNN OK")
