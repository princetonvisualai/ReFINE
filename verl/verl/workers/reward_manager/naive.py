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

from collections import defaultdict

import torch

from verl import DataProto
#from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        #self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score 
        self.reward_fn_key = reward_fn_key


    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        reward_tensor = torch.zeros_like(data.batch["responses"]).float()
        reward_extra_info = {} 

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            response: str = self.tokenizer.decode(data_item.batch["responses"], skip_special_tokens=True)
            answers: List[str] = data_item.non_tensor_batch["answers"].split("|") # split by | to get the list of answers
            data_source: str = data_item.non_tensor_batch[self.reward_fn_key]
            score = self.compute_score(
                response=response,
                answers=answers,
                data_source=data_source
            )

            if score > 0.2:
                print(f"response: {response} | answer: {answers} | score: {score}")


            reward_tensor[i, -1] = score
       
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

        
