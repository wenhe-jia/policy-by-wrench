import torch
import torch.nn as nn


class ForceEncoder(nn.Module):
    """
    力传感器信号编码器 - 支持 MLP/GRU 两种结构
    """

    def __init__(
        self,
        force_dim: int = 12,
        history_frames: int = 10,
        hidden_dim: int = 1536,
        intermediate_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        encoder_type: str = "mlp",
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 1,
        gru_bidirectional: bool = False,
    ):
        super().__init__()

        self.force_dim = force_dim
        self.history_frames = history_frames
        self.hidden_dim = hidden_dim
        self.encoder_type = str(encoder_type).lower()

        if self.encoder_type == "mlp":
            flattened_dim = force_dim * history_frames
            layers = []
            input_dim = flattened_dim
            for i in range(num_layers):
                output_dim = intermediate_dim if i < num_layers - 1 else hidden_dim
                layers.append(nn.Linear(input_dim, output_dim))
                if i < num_layers - 1:
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                input_dim = output_dim
            self.mlp = nn.Sequential(*layers)
            self.gru = None
            self.temporal_proj = None
        elif self.encoder_type == "gru":
            self.gru = nn.GRU(
                input_size=force_dim,
                hidden_size=gru_hidden_dim,
                num_layers=gru_num_layers,
                batch_first=True,
                dropout=dropout if gru_num_layers > 1 else 0.0,
                bidirectional=gru_bidirectional,
            )
            gru_out_dim = gru_hidden_dim * (2 if gru_bidirectional else 1)
            self.temporal_proj = nn.Sequential(
                nn.Linear(gru_out_dim, intermediate_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, hidden_dim),
            )
            self.mlp = None
        else:
            raise ValueError(
                f"Unsupported force encoder_type={encoder_type!r}, expected 'mlp' or 'gru'"
            )

    def forward(self, force_signals: torch.Tensor) -> torch.Tensor:
        batch_size = force_signals.shape[0]
        expected = (self.force_dim, self.history_frames)
        assert (
            force_signals.shape[1:] == expected
        ), f"Expected [B, {expected[0]}, {expected[1]}], got {force_signals.shape}"

        if self.encoder_type == "mlp":
            force_flattened = force_signals.reshape(batch_size, -1)
            assert self.mlp is not None
            force_encoded = self.mlp(force_flattened)
        else:
            assert self.gru is not None and self.temporal_proj is not None
            force_seq = force_signals.transpose(1, 2)
            _, hidden = self.gru(force_seq)
            if self.gru.bidirectional:
                temporal_feat = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            else:
                temporal_feat = hidden[-1]
            force_encoded = self.temporal_proj(temporal_feat)

        force_features = force_encoded.unsqueeze(1)
        return force_features

    def initialize_weights(self):
        if self.mlp is not None:
            for module in self.mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

        if self.gru is not None:
            for name, param in self.gru.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        if self.temporal_proj is not None:
            for module in self.temporal_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)


class CategorySpecificForceEncoder(nn.Module):
    """
    支持多机器人 (embodiment-aware) 的力编码器 - 支持时间序列
    """

    def __init__(
        self,
        force_dim: int = 12,
        hidden_dim: int = 1536,
        intermediate_dim: int = 512,
        num_embodiments: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
        history_frames: int = 10,
        encoder_type: str = "mlp",
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 1,
        gru_bidirectional: bool = False,
    ):
        super().__init__()
        self.force_dim = force_dim
        self.hidden_dim = hidden_dim
        self.num_embodiments = num_embodiments
        self.history_frames = history_frames

        self.encoders = nn.ModuleList(
            [
                ForceEncoder(
                    force_dim=force_dim,
                    history_frames=history_frames,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    encoder_type=encoder_type,
                    gru_hidden_dim=gru_hidden_dim,
                    gru_num_layers=gru_num_layers,
                    gru_bidirectional=gru_bidirectional,
                )
                for _ in range(num_embodiments)
            ]
        )

    def forward(self, force_signals: torch.Tensor, embodiment_ids: torch.Tensor) -> torch.Tensor:
        batch_size = force_signals.shape[0]
        expected = (self.force_dim, self.history_frames)
        assert (
            force_signals.shape[1:] == expected
        ), f"Expected [B, {expected[0]}, {expected[1]}], got {force_signals.shape}"

        force_features_list = []
        for b in range(batch_size):
            emb_id = embodiment_ids[b].item()
            encoder = self.encoders[emb_id]
            force_feat = encoder(force_signals[b : b + 1])
            force_features_list.append(force_feat)

        force_features = torch.cat(force_features_list, dim=0)
        return force_features

    def initialize_weights(self):
        for encoder in self.encoders:
            encoder.initialize_weights()
