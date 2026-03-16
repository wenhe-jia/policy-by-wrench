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

from typing import Any

import numpy as np
import torch
from pydantic import Field, PrivateAttr, field_validator, model_validator

from gr00t.data.schema import DatasetMetadata, StateActionMetadata
from gr00t.data.transform.base import InvertibleModalityTransform
from gr00t.data.transform.state_action import Normalizer


class ForceToTensor(InvertibleModalityTransform):
    """
    Transforms force signals to tensors.
    """

    input_dtypes: dict[str, np.dtype] = Field(
        default_factory=dict, description="The input dtypes for each force key."
    )
    output_dtypes: dict[str, torch.dtype] = Field(
        default_factory=dict, description="The output dtypes for each force key."
    )

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("input_dtypes", "output_dtypes", mode="before")
    def validate_dtypes(cls, v):
        for key, dtype in v.items():
            if isinstance(dtype, str):
                if dtype.startswith("torch."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = getattr(torch, dtype_split)
                elif dtype.startswith("np.") or dtype.startswith("numpy."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = np.dtype(dtype_split)
                else:
                    raise ValueError(f"Invalid dtype: {dtype}")
        return v

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, np.ndarray
            ), f"Unexpected input type: {type(value)}. Expected type: {np.ndarray}"
            data[key] = torch.from_numpy(value)
            if key in self.output_dtypes:
                data[key] = data[key].to(self.output_dtypes[key])
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, torch.Tensor
            ), f"Unexpected input type: {type(value)}. Expected type: {torch.Tensor}"
            data[key] = value.numpy()
            if key in self.input_dtypes:
                data[key] = data[key].astype(self.input_dtypes[key])
        return data


class ForceTransform(InvertibleModalityTransform):
    """
    Class for force signal transform.
    """

    apply_to: list[str] = Field(..., description="The keys in the modality to load and transform.")
    normalization_modes: dict[str, str] = Field(
        default_factory=dict, description="The normalization modes for each force key."
    )
    normalization_statistics: dict[str, dict] = Field(
        default_factory=dict, description="The statistics for each force key."
    )
    modality_metadata: dict[str, StateActionMetadata] = Field(
        default_factory=dict, description="The modality metadata for each force key."
    )

    _normalizers: dict[str, Normalizer] = PrivateAttr(default_factory=dict)
    _input_dtypes: dict[str, np.dtype | torch.dtype] = PrivateAttr(default_factory=dict)

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "normalization_modes"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("modality_metadata", mode="before")
    def validate_modality_metadata(cls, v):
        for modality_key, config in v.items():
            if isinstance(config, dict):
                config = StateActionMetadata.model_validate(config)
            else:
                assert isinstance(
                    config, StateActionMetadata
                ), f"Invalid source rotation config: {config}"
            v[modality_key] = config
        return v

    @model_validator(mode="after")
    def validate_normalization_statistics(self):
        for modality_key, normalization_statistics in self.normalization_statistics.items():
            if modality_key in self.normalization_modes:
                normalization_mode = self.normalization_modes[modality_key]
                if normalization_mode == "min_max":
                    assert (
                        "min" in normalization_statistics and "max" in normalization_statistics
                    ), f"Min and max statistics are required for min_max normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["min"]) == len(
                        normalization_statistics["max"]
                    ), f"Min and max statistics must have the same length, but got {normalization_statistics['min']} and {normalization_statistics['max']}"
                elif normalization_mode == "mean_std":
                    assert (
                        "mean" in normalization_statistics and "std" in normalization_statistics
                    ), f"Mean and std statistics are required for mean_std normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["mean"]) == len(
                        normalization_statistics["std"]
                    ), f"Mean and std statistics must have the same length, but got {normalization_statistics['mean']} and {normalization_statistics['std']}"
                elif normalization_mode == "q99":
                    assert (
                        "q01" in normalization_statistics and "q99" in normalization_statistics
                    ), f"q01 and q99 statistics are required for q99 normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["q01"]) == len(
                        normalization_statistics["q99"]
                    ), f"q01 and q99 statistics must have the same length, but got {normalization_statistics['q01']} and {normalization_statistics['q99']}"
                elif normalization_mode == "binary":
                    assert (
                        len(normalization_statistics) == 1
                    ), f"Binary normalization should only have one value, but got {normalization_statistics}"
                    assert normalization_statistics[0] in [
                        0,
                        1,
                    ], f"Binary normalization should only have 0 or 1, but got {normalization_statistics[0]}"
                else:
                    raise ValueError(f"Invalid normalization mode: {normalization_mode}")
        return self

    def _resolve_modality(self, modality: str) -> str:
        return "state" if modality == "force" else modality

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        dataset_statistics = dataset_metadata.statistics
        modality_metadata = dataset_metadata.modalities

        for key in self.apply_to:
            split_key = key.split(".")
            assert len(split_key) == 2, "Force keys should have two parts: 'modality.key'"
            if key not in self.modality_metadata:
                modality, state_key = split_key
                resolved_modality = self._resolve_modality(modality)
                assert hasattr(modality_metadata, resolved_modality), f"{modality} config not found"
                assert state_key in getattr(
                    modality_metadata, resolved_modality
                ), f"{state_key} config not found"
                self.modality_metadata[key] = getattr(modality_metadata, resolved_modality)[
                    state_key
                ]

        for key in self.normalization_modes:
            split_key = key.split(".")
            assert len(split_key) == 2, "Force keys should have two parts: 'modality.key'"
            modality, state_key = split_key
            resolved_modality = self._resolve_modality(modality)
            assert hasattr(dataset_statistics, resolved_modality), f"{modality} statistics not found"
            assert state_key in getattr(
                dataset_statistics, resolved_modality
            ), f"{state_key} statistics not found"
            assert (
                len(getattr(modality_metadata, resolved_modality)[state_key].shape) == 1
            ), f"{getattr(modality_metadata, resolved_modality)[state_key].shape=}"
            self.normalization_statistics[key] = getattr(dataset_statistics, resolved_modality)[
                state_key
            ].model_dump()

        for key in self.normalization_modes:
            if (
                not self.modality_metadata[key].continuous
                and self.normalization_modes[key] != "binary"
            ):
                raise ValueError(
                    f"{key} is not continuous, so it should be normalized using `binary` mode"
                )
            statistics = self.normalization_statistics[key]
            self._normalizers[key] = Normalizer(
                mode=self.normalization_modes[key], statistics=statistics
            )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            if key not in self._input_dtypes:
                input_dtype = data[key].dtype
                assert isinstance(
                    input_dtype, torch.dtype
                ), f"Unexpected input dtype: {input_dtype}. Expected type: {torch.dtype}"
                self._input_dtypes[key] = input_dtype
            else:
                assert (
                    data[key].dtype == self._input_dtypes[key]
                ), f"All forces corresponding to the same key must be of the same dtype, input dtype: {data[key].dtype}, expected dtype: {self._input_dtypes[key]}"
            state = data[key]
            if key in self._normalizers:
                state = self._normalizers[key].forward(state)
            data[key] = state
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            state = data[key]
            assert isinstance(
                state, torch.Tensor
            ), f"Unexpected state type: {type(state)}. Expected type: {torch.Tensor}"
            if key in self._normalizers:
                state = self._normalizers[key].inverse(state)
            assert isinstance(
                state, torch.Tensor
            ), f"State should be tensor after unapplying transformations, but got {type(state)}"
            if key in self._input_dtypes:
                original_dtype = self._input_dtypes[key]
                if isinstance(original_dtype, np.dtype):
                    state = state.numpy().astype(original_dtype)
                else:
                    state = state.to(original_dtype)
            data[key] = state
        return data
