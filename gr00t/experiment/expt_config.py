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

from abc import ABC, abstractmethod


class BaseExptConfig(ABC):
    @abstractmethod
    def backbone_config(self) -> dict:
        pass

    @abstractmethod
    def action_config(self) -> dict:
        pass


class ExptConfig(BaseExptConfig):
    def backbone_config(self) -> dict:
        # Keep this as a compatibility placeholder in policy-by-wrench.
        return {}

    def action_config(self) -> dict:
        # Keep this as a compatibility placeholder in policy-by-wrench.
        return {}

    def force_config(self) -> dict:
        return {
            "force_input": {
                "enabled": True,
                "force_encoder": {
                    "enabled": True,
                    "use_force_embedding_as_dit_condition": True,
                    "force_dim": 12,
                    "selected_dims": [6],
                    "history_frames": 10,
                    "encoder_type": "mlp",
                    "force_embedding_dim": 1536,
                    "gru_hidden_dim": 128,
                    "gru_num_layers": 1,
                    "gru_bidirectional": False,
                    "embodiment_aware": False,
                    "condition_source": "force",
                    "fusion_mode": "concat",
                    "intermediate_dim": 512,
                    "num_layers": 2,
                    "dropout": 0.0,
                },
            },
        }
