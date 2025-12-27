# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Sharding manager to implement HybridEngine
"""

import torch.distributed as dist
from verl import DataProto


class BaseShardingManager:
    def __init__(self):
        self.timing = {}

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def preprocess_data(self, data: DataProto) -> DataProto:
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        return data


class HFRolloutShardingManager(BaseShardingManager):
    def __init__(self):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def preprocess_data(self, data: DataProto) -> DataProto:
        return data.chunk(chunks = self.world_size)[self.rank]

    def postprocess_data(self, data: DataProto) -> DataProto:
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, data)
        return DataProto.concat(gathered)

