# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.experiment.expt_config import ExptConfig
from gr00t.model.action_head.action_encoder import SinusoidalPositionalEncoding, swish
from gr00t.model.force.force_encoder import CategorySpecificForceEncoder, ForceEncoder

from .cross_attention_dit import DiT, SelfAttentionTransformer


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.force_config = ExptConfig().force_config()
        self.condition_source = "vl"

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        if self.force_config.get("force_input", {}).get("enabled", False):
            self.use_force_encoder = True
            force_encoder_cfg = self.force_config.get("force_input", {}).get("force_encoder", {})
            self.condition_source = str(force_encoder_cfg.get("condition_source", "force")).lower()
            if self.condition_source not in ("vl", "force"):
                raise ValueError(
                    f"Unsupported force_encoder.condition_source={self.condition_source!r}, "
                    "expected 'vl' or 'force'"
                )
            force_encoder_type = str(force_encoder_cfg.get("encoder_type", "mlp")).lower()
            selected_force_dims = force_encoder_cfg.get("selected_dims", None)
            if selected_force_dims is None:
                self.force_dim = int(force_encoder_cfg.get("force_dim", 12))
            else:
                if isinstance(selected_force_dims, int):
                    selected_force_dims = [selected_force_dims]
                if not isinstance(selected_force_dims, (list, tuple)):
                    raise ValueError(
                        "force_encoder.selected_dims must be a list/tuple of integer indices"
                    )
                self.force_dim = len(selected_force_dims)
                if self.force_dim == 0:
                    raise ValueError("force_encoder.selected_dims should not be empty")
            self.history_frames = int(force_encoder_cfg.get("history_frames", 10))

            if force_encoder_cfg.get("embodiment_aware", False):
                self.force_encoder = CategorySpecificForceEncoder(
                    force_dim=self.force_dim,
                    history_frames=self.history_frames,
                    hidden_dim=self.input_embedding_dim,
                    intermediate_dim=force_encoder_cfg.get("intermediate_dim", 512),
                    num_embodiments=config.max_num_embodiments,
                    num_layers=force_encoder_cfg.get("num_layers", 2),
                    dropout=force_encoder_cfg.get("dropout", 0.1),
                    encoder_type=force_encoder_type,
                    gru_hidden_dim=force_encoder_cfg.get("gru_hidden_dim", 128),
                    gru_num_layers=force_encoder_cfg.get("gru_num_layers", 1),
                    gru_bidirectional=force_encoder_cfg.get("gru_bidirectional", False),
                )
            else:
                self.force_encoder = ForceEncoder(
                    force_dim=self.force_dim,
                    history_frames=self.history_frames,
                    hidden_dim=self.input_embedding_dim,
                    intermediate_dim=force_encoder_cfg.get("intermediate_dim", 512),
                    num_layers=force_encoder_cfg.get("num_layers", 2),
                    dropout=force_encoder_cfg.get("dropout", 0.1),
                    encoder_type=force_encoder_type,
                    gru_hidden_dim=force_encoder_cfg.get("gru_hidden_dim", 128),
                    gru_num_layers=force_encoder_cfg.get("gru_num_layers", 1),
                    gru_bidirectional=force_encoder_cfg.get("gru_bidirectional", False),
                )

            self.force_fusion_mode = force_encoder_cfg.get("fusion_mode", "concat")
            if self.force_fusion_mode == "fuse":
                self.state_force_fusion = nn.Sequential(
                    nn.Linear(2 * self.hidden_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                )
            elif self.force_fusion_mode == "vl_concat":
                if self.input_embedding_dim == config.backbone_embedding_dim:
                    self.force_to_vl_proj = nn.Identity()
                else:
                    self.force_to_vl_proj = nn.Linear(
                        self.input_embedding_dim, config.backbone_embedding_dim
                    )
            if self.input_embedding_dim == config.backbone_embedding_dim:
                self.force_to_condition_proj = nn.Identity()
            else:
                self.force_to_condition_proj = nn.Linear(
                    self.input_embedding_dim, config.backbone_embedding_dim
                )
        else:
            self.use_force_encoder = False

        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if self.force_config.get("force_input", {}).get("enabled", {}):
            self.force_encoder.initialize_weights()
            if (
                self.force_fusion_mode == "vl_concat"
                and hasattr(self, "force_to_vl_proj")
                and isinstance(self.force_to_vl_proj, nn.Linear)
            ):
                init.xavier_uniform_(self.force_to_vl_proj.weight)
                init.zeros_(self.force_to_vl_proj.bias)
            if (
                hasattr(self, "force_to_condition_proj")
                and isinstance(self.force_to_condition_proj, nn.Linear)
            ):
                init.xavier_uniform_(self.force_to_condition_proj.weight)
                init.zeros_(self.force_to_condition_proj.bias)
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def _encode_force_features(
        self, action_input: BatchFeature, embodiment_id: torch.Tensor
    ) -> torch.Tensor | None:
        if not self.use_force_encoder or not hasattr(action_input, "force_signal"):
            return None

        force_signal = action_input.force_signal
        expected = (self.force_dim, self.history_frames)
        assert (
            force_signal.shape[1:] == expected
        ), f"Expected [B, {expected[0]}, {expected[1]}], got {force_signal.shape}"

        if hasattr(self.force_encoder, "encoders"):
            return self.force_encoder(force_signal, embodiment_id)
        return self.force_encoder(force_signal)

    def _build_force_condition(
        self, action_input: BatchFeature, embodiment_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        force_features = self._encode_force_features(action_input, embodiment_id)
        if force_features is None:
            raise ValueError(
                "force_signal is required when force_encoder.condition_source is set to 'force'"
            )
        cond_embs = self.force_to_condition_proj(force_features)
        cond_mask = torch.ones(
            (cond_embs.size(0), cond_embs.size(1)),
            dtype=torch.long,
            device=cond_embs.device,
        )
        return cond_embs, cond_mask

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        force_features = self._encode_force_features(action_input, embodiment_id)
        if (
            self.condition_source == "vl"
            and force_features is not None
            and self.force_fusion_mode == "fuse"
        ):
            combined = torch.cat([state_features, force_features], dim=-1)
            state_features = self.state_force_fusion(combined)
            if state_features.ndim == 2:
                state_features = state_features.unsqueeze(1)

        if self.condition_source == "force":
            cond_embs, cond_attn_mask = self._build_force_condition(action_input, embodiment_id)
        else:
            backbone_output = self.process_backbone_output(backbone_output)
            cond_embs = backbone_output.backbone_features
            cond_attn_mask = backbone_output.backbone_attention_mask
            if force_features is not None and self.force_fusion_mode == "vl_concat":
                force_vl_features = self.force_to_vl_proj(force_features).to(dtype=cond_embs.dtype)
                cond_embs = torch.cat([cond_embs, force_vl_features], dim=1)
                force_attn_mask = torch.ones(
                    (force_vl_features.size(0), force_vl_features.size(1)),
                    dtype=cond_attn_mask.dtype,
                    device=cond_attn_mask.device,
                )
                cond_attn_mask = torch.cat([cond_attn_mask, force_attn_mask], dim=1)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        device = action_features.device
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Joint hidden input = state_token + future_token + action_token.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(state_features.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=cond_embs,
            encoder_attention_mask=cond_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        force_features = self._encode_force_features(action_input, embodiment_id)
        if (
            self.condition_source == "vl"
            and force_features is not None
            and self.force_fusion_mode == "fuse"
        ):
            combined = torch.cat([state_features, force_features], dim=-1)
            state_features = self.state_force_fusion(combined)
            if state_features.ndim == 2:
                state_features = state_features.unsqueeze(1)

        if self.condition_source == "force":
            cond_embs, cond_attn_mask = self._build_force_condition(action_input, embodiment_id)
        else:
            backbone_output = self.process_backbone_output(backbone_output)
            cond_embs = backbone_output.backbone_features
            cond_attn_mask = backbone_output.backbone_attention_mask
            if force_features is not None and self.force_fusion_mode == "vl_concat":
                force_vl_features = self.force_to_vl_proj(force_features).to(dtype=cond_embs.dtype)
                cond_embs = torch.cat([cond_embs, force_vl_features], dim=1)
                force_attn_mask = torch.ones(
                    (force_vl_features.size(0), force_vl_features.size(1)),
                    dtype=cond_attn_mask.dtype,
                    device=cond_attn_mask.device,
                )
                cond_attn_mask = torch.cat([cond_attn_mask, force_attn_mask], dim=1)

        # Set initial actions as the sampled noise.
        batch_size = state_features.shape[0]
        device = state_features.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=state_features.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Joint hidden input = state_token + future_token + action_token.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(state_features.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=cond_embs,
                encoder_attention_mask=cond_attn_mask,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
