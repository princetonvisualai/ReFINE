# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.py_functional import append_to_dict

WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]

def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.reweight_method,
                config.pf_ppo.weight_pow,
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data

def compute_reward_for_ttt(data: DataProto, reward = "cosine_similarity", flatten = True):
    '''
    Compute the reward for the TTT data.
    Args:
        data: DataProto object containing the input data.
        reward: The reward function to use.
        flatten: Whether to flatten the reward.
    Returns:
        The reward tensor.
    '''
    if reward == "cosine_similarity":
        response_hidden_states = data.batch["response_hidden_states"]
        ground_truth_hidden_states = data.batch["ground_truth_hidden_states"]
        sim = F.cosine_similarity(response_hidden_states, ground_truth_hidden_states, dim = -1)
    elif reward == "binary":
        response_ids = data.batch["responses"]
        ground_truth_ids = data.batch["ground_truth_ids"]
        sim = (ground_truth_ids == response_ids).float()
    elif reward == "hybrid":
        response_ids = data.batch["responses"]
        ground_truth_ids = data.batch["ground_truth_ids"]
        response_hidden_states = data.batch["response_hidden_states"]
        ground_truth_hidden_states = data.batch["ground_truth_hidden_states"]
        cs_score = F.cosine_similarity(response_hidden_states, ground_truth_hidden_states, dim = -1)
        em_score = (ground_truth_ids == response_ids).float()
        sim = cs_score + em_score
    else:
        raise NotImplementedError(f"Reward function {reward} is not supported")
    
    if flatten: 
        scores = torch.zeros_like(data.batch["response_mask"]).float()
        scores[:, -1] = sim.mean(dim = -1)
    else:
        scores = sim

    return scores

def prepare_ppo_batch_for_ttt(
        data: DataProto, 
        ttt_n_chunks: int, 
        ttt_k: int, 
        ttt_n: int, 
        pad_token_id: int,
        ttt_sampling: str
    ) -> DataProto:
    
    input_ids = data.batch['input_ids']
    attention_mask = data.batch['attention_mask']
    hidden_states = data.batch['hidden_states']
    uuids = np.array([str(uuid.uuid4()) for _ in range(len(data.batch))], dtype=object)

    entropys = data.batch['entropys']
    if ttt_sampling == "entropy": 
        print("entropy sampling")
        if ttt_k > 1:
            # entropy pooling
            if ttt_k % 2 == 0: # if even, use one up odd kernel size
                kernel_size = ttt_k + 1
            else: # if odd, use the same kernel size
                kernel_size = ttt_k
            entropys = entropys.unsqueeze(1)  # [B, 1, S]
            smoothed_entropys = F.avg_pool1d(
                entropys,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            )
            smoothed_entropys = smoothed_entropys.squeeze(1) # [B, S]
        else: # ttt_k == 1
            smoothed_entropys = entropys
    elif ttt_sampling == "uniform": # uniform sampling
        print("uniform sampling")
        smoothed_entropys = torch.ones_like(entropys, dtype = entropys.dtype)
    else: # argmax, argmin
        smoothed_entropys = entropys

    # sampling response
    B, S = attention_mask.shape
    new_input_ids = []
    new_attention_mask = []
    new_ground_truth_ids = []
    new_ground_truth_hidden_states = []
    new_uuids = []
    for i in range(B):
        seq_input_ids = input_ids[i, :]
        seq_attention_mask = attention_mask[i, :]
        seq_smoothed_entropys = smoothed_entropys[i, :]
        seq_hidden_states = hidden_states[i, :, :]
        seq_uuids = uuids[i]

        seq_start_idx = seq_attention_mask.argmax(dim = -1).item() # first nonzero
        seq_len = seq_attention_mask.sum(dim = -1).item()
        seq_chunk_size = seq_len // ttt_n_chunks

        for j in range(ttt_n_chunks):
            chunk_start_idx = seq_start_idx + j * seq_chunk_size
            chunk_smoothed_entropys = seq_smoothed_entropys[chunk_start_idx: chunk_start_idx + seq_chunk_size]
            if ttt_sampling in ["entropy", "uniform"]:
                chunk_smoothed_entropys_softmax = F.softmax(chunk_smoothed_entropys, dim = -1)
                target_idx = torch.distributions.Categorical(probs = chunk_smoothed_entropys_softmax).sample()
            elif ttt_sampling == "argmax": 
                target_idx = torch.argmax(chunk_smoothed_entropys)
            elif ttt_sampling == "argmin":
                target_idx = torch.argmin(chunk_smoothed_entropys)
            else:
                raise NotImplementedError(f"TTT sampling method {ttt_sampling} is not supported")
  
            response_start_idx = chunk_start_idx + target_idx + 1 # start from the next token
            response_start_idx = torch.clamp(response_start_idx, min = seq_start_idx + 20, max = S - 20)
            response_end_idx = response_start_idx + ttt_k

            seq_new_input_ids = torch.full((S,), pad_token_id, dtype = seq_input_ids.dtype)
            seq_new_input_ids[-response_start_idx:] = seq_input_ids[:response_start_idx]

            seq_new_attention_mask = torch.zeros((S,), dtype = seq_attention_mask.dtype)
            seq_new_attention_mask[-response_start_idx:] = seq_attention_mask[:response_start_idx]
            
            seq_new_ground_truth_ids = seq_input_ids[response_start_idx:response_end_idx]
            seq_new_ground_truth_hidden_states = seq_hidden_states[response_start_idx:response_end_idx, :]

            new_input_ids.append(seq_new_input_ids)
            new_attention_mask.append(seq_new_attention_mask)
            new_ground_truth_ids.append(seq_new_ground_truth_ids)
            new_ground_truth_hidden_states.append(seq_new_ground_truth_hidden_states)
            
            new_uuids.append(seq_uuids)
   
    new_input_ids = torch.stack(new_input_ids)
    new_attention_mask = torch.stack(new_attention_mask)
    new_ground_truth_ids = torch.stack(new_ground_truth_ids)
    new_ground_truth_hidden_states = torch.stack(new_ground_truth_hidden_states)
    new_uuids = np.array(new_uuids)
    tensor_batch = {
        "input_ids": new_input_ids,
        "attention_mask": new_attention_mask,
        "ground_truth_ids": new_ground_truth_ids,
        "ground_truth_hidden_states": new_ground_truth_hidden_states,
    }
    non_tensor_batch = {
        "uid": new_uuids,
    }

    output = DataProto.from_dict(
        tensors=tensor_batch, 
        non_tensors=non_tensor_batch, 
    )

    # get new uuids for each chunk if ttt_n > 1
    if ttt_n > 1:
        output.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(output.batch))], dtype=object
        )

    output = output.repeat(repeat_times=ttt_n, interleave=True)  

    return output 

def compute_reward_metrics(batch: DataProto):

    flattened_cs_reward = compute_reward_for_ttt(batch, reward="cosine_similarity", flatten = True).sum(dim = -1)
    flattened_binary_reward = compute_reward_for_ttt(batch, reward="binary", flatten = True).sum(dim = -1)

    combined_flattened_reward = flattened_cs_reward + flattened_binary_reward
    
    reward_metrics = {
        "reward/cs_reward_mean": flattened_cs_reward.mean(),
        "reward/cs_reward_std": flattened_cs_reward.std(),
        "reward/binary_reward_mean": flattened_binary_reward.mean(),
        "reward/binary_reward_std": flattened_binary_reward.std(),
        "reward/hybrid_reward_mean": combined_flattened_reward.mean(),
        "reward/hybrid_reward_std": combined_flattened_reward.std()
    }

    return reward_metrics



class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GPG,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            if config.algorithm.get("task_ppo_update", False):
            
                check_mutually_exclusive(
                    config.actor_rollout_ref.actor.ppo_micro_batch_size,
                    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                    "actor_rollout_ref.actor",
                )

                if self.use_reference_policy:
                    # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                    check_mutually_exclusive(
                        config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                        config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                        "actor_rollout_ref.ref",
                    )

                #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                    config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.rollout",
                )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if config.algorithm.get("task_ppo_update", False):
                assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
                sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
                if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                    assert (
                        config.actor_rollout_ref.actor.ppo_mini_batch_size
                        % config.actor_rollout_ref.actor.ppo_micro_batch_size
                        == 0
                    )
                    assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

            # TTT PPO update
            if config.algorithm.get("ttt_ppo_update", False):
                assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ttt_ppo_mini_batch_size
            # TTT PPO validation update
            if config.algorithm.get("val_ttt_ppo_update", False):
                assert config.data.val_batch_size >= config.actor_rollout_ref.actor.ttt_ppo_mini_batch_size

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", False),
            #drop_last=False,
            drop_last=True,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _validate(self):
    
        val_metrics = defaultdict(list)
        
        for test_data in self.val_dataloader:
            batch_timing = {}

            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )
            
            if self.config.algorithm.get("val_get_loss", False):
                # mid-training mode
                with marked_timer("process_sequence_for_val", batch_timing):
                    output = self.actor_rollout_wg.process_input_for_validation(test_batch)
                assert output.batch['loss'].shape == (len(test_batch), 1)
                val_metrics['val_loss'].append(output.batch['loss'].mean().item())

                input_ids = test_batch.batch["input_ids"]
                reward_mask = test_batch.batch["attention_mask"]
                reward_mask[:, -1] = 0

                responses = output.batch.pop("responses")
                ground_truth = torch.roll(input_ids, shifts = -1, dims = -1)
                
                em_score = (responses == ground_truth)[reward_mask == 1].float().mean()
                val_metrics['val_em_score'].append(em_score)
            else:
                # inner training loop for validation
                if self.config.algorithm.get("val_ttt_sft_update", False) or self.config.algorithm.get("val_ttt_ppo_update", False):
                    print("inner training loop -- ttt update for validation")
                    ttt_batch = test_batch.select(
                        batch_keys=["input_ids", "attention_mask"],
                        non_tensor_batch_keys=[],
                    )
                    with marked_timer("ttt_update_val", batch_timing):
                        self._inner_training_loop(ttt_batch, validation=True)

                batch_keys_to_pop = ["input_ids", "attention_mask"]
                test_gen_batch = test_batch.pop(
                    batch_keys=batch_keys_to_pop,
                )
                test_gen_batch.meta_info["validate"] = True
                with marked_timer("gen_val", batch_timing):
                    test_gen_batch_output = self.actor_rollout_wg.generate_sequences(test_gen_batch)    
                test_batch = test_batch.union(test_gen_batch_output)
                print("generation time: ", batch_timing["gen_val"])
                
                test_batch.batch["response_mask"] = compute_response_mask(test_batch)

                test_reward_tensor, test_reward_extra_infos_dict = compute_reward(test_batch, self.val_reward_fn)
                val_metrics['val_reward'] += test_reward_tensor.sum(-1).tolist()

        val_metrics = reduce_metrics(val_metrics)
        return val_metrics

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)


        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")
 
    def _inner_training_loop(self, batch: DataProto, validation: bool = False):
        """
        RL to update the fast weights. Should be able to call during train or validation.
        """
        inner_metrics = {}
        inner_timing = {}
      
        sft_update = self.config.algorithm.get("ttt_sft_update" if not validation else "val_ttt_sft_update", False)
        ppo_update = self.config.algorithm.get("ttt_ppo_update" if not validation else "val_ttt_ppo_update", False)

        if sft_update:
            sft_batch = batch.select(
                batch_keys=["input_ids", "attention_mask"], 
                non_tensor_batch_keys=[],
            )
            sft_batch.meta_info["ttt_global_token_num"] = torch.sum(sft_batch.batch["attention_mask"], dim=-1).tolist()

        if ppo_update:
            ppo_batch = batch.select(
                batch_keys=["input_ids", "attention_mask"], 
                non_tensor_batch_keys=[],
            )
            with marked_timer("forward_ttt", inner_timing):
                outputs = self.actor_rollout_wg.process_input_for_ttt(ppo_batch) # hidden_states, responses, entropys
   
            ppo_batch = ppo_batch.union(outputs)

            ttt_sampling = self.config.algorithm.get("ttt_sampling", "entropy")
            print(f"ttt_sampling: {ttt_sampling}")
            ppo_batch = prepare_ppo_batch_for_ttt(
                    data=ppo_batch,  
                    ttt_n_chunks=self.config.actor_rollout_ref.actor.ttt_n_chunks, 
                    ttt_k=self.config.actor_rollout_ref.rollout.ttt_response_length, 
                    ttt_n=self.config.actor_rollout_ref.actor.ttt_n, 
                    pad_token_id=self.tokenizer.pad_token_id,
                    ttt_sampling=ttt_sampling 
                ) # input_ids, attention_mask, ground_truth_ids, ground_truth_hidden_states, uid
            
            gen_batch = ppo_batch.pop(
                batch_keys=["input_ids", "attention_mask"]
            )
            with marked_timer("gen_ttt", inner_timing):
                output = self.actor_rollout_wg.generate_sequences_for_ttt(gen_batch) # responses, response_mask, updated input_ids, updated attention_mask, temperature
            ppo_batch = ppo_batch.union(output)
            print("generation time: ", inner_timing["gen_ttt"])
            
            ppo_batch.meta_info["ttt_global_token_num"] = torch.sum(ppo_batch.batch["attention_mask"], dim=-1).tolist()

            # recompute old_log_probs
            with marked_timer("old_log_prob_ttt", inner_timing):
                output = self.actor_rollout_wg.compute_log_prob_for_ttt(ppo_batch) # old_log_probs, hidden_states
            ppo_batch = ppo_batch.union(output)

            reward_metrics = compute_reward_metrics(ppo_batch)
            inner_metrics.update(reward_metrics)
            
            ppo_batch.batch["token_level_scores"] = compute_reward_for_ttt(ppo_batch, reward=self.config.actor_rollout_ref.actor.ttt_reward)

            # TODO: KL penalty for ttt
            ppo_batch.batch["token_level_rewards"] = ppo_batch.batch["token_level_scores"]
                
            # compute advantages, executed on the driver process
            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor
            ppo_batch = compute_advantage(
                ppo_batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.actor.ttt_n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )
        else:
            ppo_batch = None
        
        # update actor
        with marked_timer("update_actor_ttt", inner_timing):
            if sft_update:
                if ppo_update:
                    ttt_actor_output = self.actor_rollout_wg.update_actor_ppo_ttt(sft_data=sft_batch, ppo_data=ppo_batch)
                else:
                    ttt_actor_output = self.actor_rollout_wg.update_actor_sft_ttt(sft_data=sft_batch)
            else:
                raise ValueError("sft_update and ppo_update cannot be False at the same time")
            inner_metrics.update(reduce_metrics(ttt_actor_output.meta_info["metrics"]))
       
        return ppo_batch, inner_metrics, inner_timing

    def _outer_training_loop(self, batch: DataProto):

        outer_metrics = {}
        outer_timing = {}

        sft_update = self.config.algorithm.get("task_sft_update", False)
        ppo_update = self.config.algorithm.get("task_ppo_update", False)
        assert sft_update != ppo_update, "sft_update and ppo_update cannot be the same"
        
        if sft_update:
            batch.meta_info["global_token_num"] = (torch.sum(batch.batch["attention_mask"], dim=-1) + torch.sum(batch.batch["ground_truth_attention_mask"], dim=-1)).tolist()
            with marked_timer("update_actor_task", outer_timing):
                task_actor_output = self.actor_rollout_wg.update_actor_sft(batch)
            outer_metrics.update(reduce_metrics(task_actor_output.meta_info["metrics"]))

        if ppo_update: 
            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask"]
            )
            # generate a batch
            with marked_timer("gen_task", outer_timing, color="red"):
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)    
            batch = batch.union(gen_batch_output)

            if "response_mask" not in batch.batch.keys():
                batch.batch["response_mask"] = compute_response_mask(batch)

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
            batch.batch["token_level_scores"] = reward_tensor

            # recompute old_log_probs
            with marked_timer("old_log_prob_task", outer_timing, color="blue"):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch) # compute old_log_probs for temperature
            batch = batch.union(old_log_prob)

            batch.batch["token_level_rewards"] = reward_tensor

            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                "norm_adv_by_std_in_grpo", True
            )  # GRPO adv normalization factor

            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )

            with marked_timer("update_actor_task", outer_timing, color="red"):
                task_actor_output = self.actor_rollout_wg.update_actor(batch)
            outer_metrics.update(reduce_metrics(task_actor_output.meta_info["metrics"]))
        
        return batch, outer_metrics, outer_timing

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        #if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress", position=0)

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # inner training loop 
                    ttt_sft_update = self.config.algorithm.get("ttt_sft_update", False)
                    ttt_ppo_update = self.config.algorithm.get("ttt_ppo_update", False)
                    if ttt_sft_update or ttt_ppo_update:
                        print("inner training loop -- ttt update")
                        ttt_batch = batch.select(
                            batch_keys=["input_ids", "attention_mask"],
                            non_tensor_batch_keys=[],
                        )
                        with marked_timer("ttt_update", timing_raw):
                            ttt_batch, inner_metrics, inner_timing = self._inner_training_loop(ttt_batch)
                        metrics.update(inner_metrics)
                        timing_raw.update(inner_timing)
                        
                        if ttt_ppo_update:
                            metrics.update(compute_data_metrics(batch=ttt_batch, use_critic=self.use_critic, ttt=True))


                    task_sft_update = self.config.algorithm.get("task_sft_update", False)
                    task_ppo_update = self.config.algorithm.get("task_ppo_update", False)
                    if task_sft_update or task_ppo_update:
                        print("outer training loop -- task update")
                        with marked_timer("task_update", timing_raw):
                            task_batch, outer_metrics, outer_timing = self._outer_training_loop(batch)
                        metrics.update(outer_metrics)
                        timing_raw.update(outer_timing)

                        if task_ppo_update:
                            metrics.update(compute_data_metrics(batch=task_batch, use_critic=self.use_critic, ttt=False))
                
                    

                    # validate
                    if (
                        self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step
                    or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()
            
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_timing_metrics(timing_raw=timing_raw))
                
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return


