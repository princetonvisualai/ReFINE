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
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available, get_device_id
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            #position_ids = micro_batch["position_ids"]
            position_ids = None
            entropy = None
            #if position_ids.dim() == 3:  # qwen2vl mrope
            #    position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
            
            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _forward_micro_batch_for_input_ttt(
        self, micro_batch, get_hidden_states=True, get_response=True, get_entropy=True, get_loss=True
    ):
        """
        Returns:
            hidden_states: # (bs, response_len, hidden_size)
            response_ids: # (bs, response_len)
            entropys: # (bs, response_len)
            loss: # (1,)
        """
        
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            #position_ids = micro_batch["position_ids"]
            position_ids = None

            extra_args = {}
            if get_loss:
                extra_args["labels"] = input_ids
            if get_hidden_states:
                extra_args["output_hidden_states"] = True

            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **extra_args,
            )  # prevent model thinks we are generating
            logits = output.logits

            hidden_states = None
            if get_hidden_states:
                hidden_states = output.hidden_states[-1] # (mbsz, seq_len, hidden_size)

            response_ids = None
            if get_response:
                response_ids = torch.distributions.Categorical(logits = logits / 1.0).sample() # (mbsz, seq_len)

            entropy = None
            if get_entropy:
                entropy = verl_F.entropy_from_logits(logits)  # (mbsz, seq_len)

            loss = None 
            if get_loss:
                loss = output.loss   

            return hidden_states, response_ids, entropy, loss

    def _forward_micro_batch_for_generation_ttt(self, micro_batch, temperature):
        """
        Returns:
            response_ids: # (bs, response_len)
            input_ids: # (bs, seq_len)
            attention_mask: # (bs, seq_len)
        """
        
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            #position_ids = micro_batch["position_ids"]
            position_ids = None

            for i in range(self.config.ttt_k):
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                
                step_logits = output.logits[:, -1, :]
                step_response = torch.distributions.Categorical(logits = step_logits / temperature).sample() # (mbsz,)
                input_ids = torch.cat([input_ids, step_response.unsqueeze(-1)], dim=-1).to(input_ids.dtype).to(input_ids.device)
                attention_mask = torch.cat([attention_mask, torch.ones_like(step_response).unsqueeze(-1)], dim=-1).to(attention_mask.dtype).to(attention_mask.device)

            response_ids = input_ids[:, -self.config.ttt_k:]

            return response_ids, input_ids, attention_mask

    def _forward_micro_batch_for_ttt(self, micro_batch, temperature, get_hidden_states=True):
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            response_ids = micro_batch["responses"] 
            response_length = response_ids.size(-1)

            extra_args = {}
            if get_hidden_states:
                extra_args["output_hidden_states"] = True

            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                **extra_args,
            )  # prevent model thinks we are generating
            
            logits = output.logits    
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = logprobs_from_logits(logits, response_ids)
            
            hidden_states = None
            if get_hidden_states:   
                hidden_states = output.hidden_states[-1]
                hidden_states = hidden_states[:, -response_length:, :]
            return hidden_states, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        #select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        select_keys = ["responses", "input_ids", "attention_mask"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            if entropy is not None:
                entropy = entropy.to("cpu")
            
            log_probs_lst.append(log_probs.to("cpu"))
            if calculate_entropy:
                entropy_lst.append(entropy.to("cpu"))

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def process_input_for_ttt(self, data: DataProto):
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        
        select_keys = ["input_ids", "attention_mask"]
        data = data.select(batch_keys=select_keys)
        micro_batches = data.split(micro_batch_size)

        hidden_states_lst = []
        response_ids_lst = []
        entropys_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                hidden_states, response_ids, entropys, _ = self._forward_micro_batch_for_input_ttt(model_inputs, get_hidden_states=True, get_response=True, get_entropy=True, get_loss=False)
                hidden_states = hidden_states.to("cpu")
                response_ids = response_ids.to("cpu")
                entropys = entropys.to("cpu")
            hidden_states_lst.append(hidden_states)
            response_ids_lst.append(response_ids)
            entropys_lst.append(entropys)

        hidden_states = torch.concat(hidden_states_lst, dim=0)
        response_ids = torch.concat(response_ids_lst, dim=0)
        entropys = torch.concat(entropys_lst, dim=0)

        return hidden_states, response_ids, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def process_sequence_for_validation(self, data: DataProto):
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        
        select_keys = ["input_ids", "attention_mask"]
        data = data.select(batch_keys=select_keys)
        micro_batches = data.split(micro_batch_size)

        response_ids_lst = []
        loss_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                _, response_ids, _, loss = self._forward_micro_batch_for_input_ttt(model_inputs, get_hidden_states=False, get_response=True, get_entropy=False, get_loss=True)
            response_ids_lst.append(response_ids.to("cpu"))
            loss_lst.append(loss.detach().to("cpu").item())

        response_ids = torch.concat(response_ids_lst, dim=0)
        loss = loss_lst

        return response_ids, loss

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def generate_sequences_for_ttt(self, data: DataProto):

        # set to eval
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]

        select_keys = ["input_ids", "attention_mask"]
        data = data.select(batch_keys=select_keys)
        micro_batches = data.split(micro_batch_size)

        response_ids_lst = []
        input_ids_lst = []
        attention_mask_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                response_ids, input_ids, attention_mask = self._forward_micro_batch_for_generation_ttt(model_inputs, temperature=temperature)
                response_ids = response_ids.to("cpu")
                input_ids = input_ids.to("cpu")
                attention_mask = attention_mask.to("cpu")
            response_ids_lst.append(response_ids)
            input_ids_lst.append(input_ids)
            attention_mask_lst.append(attention_mask)
        response_ids = torch.cat(response_ids_lst, dim=0)
        input_ids = torch.cat(input_ids_lst, dim=0)
        attention_mask = torch.cat(attention_mask_lst, dim=0)
        return response_ids, input_ids, attention_mask

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob_for_ttt(self, data: DataProto):
        # set to eval
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]

        select_keys = ["input_ids", "attention_mask", "responses"]
        batch = data.select(batch_keys=select_keys)
        micro_batches = batch.split(micro_batch_size)

        hidden_states_lst = []
        log_probs_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                hidden_states, log_probs = self._forward_micro_batch_for_ttt(model_inputs, temperature=temperature, get_hidden_states=True)
                hidden_states = hidden_states.to("cpu")
                log_probs = log_probs.to("cpu")
            log_probs_lst.append(log_probs)
            hidden_states_lst.append(hidden_states)
        
        hidden_states = torch.concat(hidden_states_lst, dim=0)
        log_probs = torch.concat(log_probs_lst, dim=0)
        return hidden_states, log_probs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            #"position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            #"actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            #"actor/ppo_kl": ppo_kl.detach().item(),
                            #"actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy_ppo_ttt(self, sft_data: DataProto, ppo_data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = ppo_data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        ppo_select_keys = ["responses", "response_mask", "input_ids", "attention_mask", "old_log_probs", "advantages"]
        #if self.config.use_kl_loss:
        #    ppo_select_keys.append("ref_log_prob")
        sft_select_keys = ["input_ids", "attention_mask"]

        ppo_data = ppo_data.select(batch_keys=ppo_select_keys)
        sft_data = sft_data.select(batch_keys=sft_select_keys) 

        ppo_mini_batch_size = self.config.ttt_mini_batch_size
        sft_mini_batch_size = self.config.ttt_mini_batch_size // self.config.ttt_n_chunks // self.config.ttt_n 
        ppo_mini_batches = ppo_data.split(ppo_mini_batch_size) 
        sft_mini_batches = sft_data.split(sft_mini_batch_size) 

        self.sft_gradient_accumulation = sft_mini_batch_size // self.config.ttt_micro_batch_size_per_gpu 
        self.ppo_gradient_accumulation = ppo_mini_batch_size // self.config.ttt_micro_batch_size_per_gpu


        metrics = {}
        for sft_mini_batch, ppo_mini_batch in zip(sft_mini_batches, ppo_mini_batches):

            sft_micro_batches = sft_mini_batch.split(self.config.ttt_micro_batch_size_per_gpu)
            ppo_micro_batches = ppo_mini_batch.split(self.config.ttt_micro_batch_size_per_gpu)
            
            self.actor_optimizer.zero_grad()

            for sft_micro_batch in sft_micro_batches:
                print("sft_micro_batches")
                sft_micro_batch = sft_micro_batch.to(get_device_id())

                # compute sft loss
                sft_inputs = {**sft_micro_batch.batch, **sft_micro_batch.non_tensor_batch}
                _, _, _, sft_loss = self._forward_micro_batch_for_input_ttt(sft_inputs, get_hidden_states=False, get_response=False, get_entropy=False, get_loss=True)
                sft_loss = sft_loss * self.config.ttt_sft_loss_coef / self.sft_gradient_accumulation
                sft_loss.backward() 
                
                sft_metrics = {
                    "actor/sft_loss_ttt": sft_loss.detach().item(),
                }
                append_to_dict(metrics, sft_metrics)

            for ppo_micro_batch in ppo_micro_batches:
                print("ppo_micro_batches")
                ppo_micro_batch = ppo_micro_batch.to(get_device_id())
                
                ppo_inputs = {**ppo_micro_batch.batch, **ppo_micro_batch.non_tensor_batch}
                response_mask = ppo_inputs["response_mask"]
                old_log_prob = ppo_inputs["old_log_probs"]
                advantages = ppo_inputs["advantages"]

                clip_ratio = self.config.clip_ratio
                clip_ratio_low = (
                    self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                )
                clip_ratio_high = (
                    self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                )
                clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                entropy_coeff = self.config.entropy_coeff
                loss_agg_mode = self.config.loss_agg_mode

                _, log_prob = self._forward_micro_batch_for_ttt(
                    ppo_inputs, temperature=temperature, get_hidden_states=False
                )

                loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                if self.config.policy_loss.loss_mode == "vanilla":
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                else:
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                    )

                if entropy_coeff != 0:
                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff
                else:
                    policy_loss = pg_loss
                '''
                if self.config.use_kl_loss:
                    ref_log_prob = model_inputs["ref_log_prob"]
                    # compute kl loss
                    kld = kl_penalty(
                        logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                    )
                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                    micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                '''
                policy_loss = policy_loss * self.config.ttt_ppo_loss_coef / self.ppo_gradient_accumulation 
                policy_loss.backward()

                ppo_metrics = {
                            "actor/pg_loss_ttt": policy_loss.detach().item(),
                            #"actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            #"actor/ppo_kl": ppo_kl.detach().item(),
                            #"actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                            }
                
                append_to_dict(metrics, ppo_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy_sft_ttt(self, sft_data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        sft_select_keys = ["input_ids", "attention_mask"]
        sft_data = sft_data.select(batch_keys=sft_select_keys)
        print("sft data length: ", len(sft_data))
        sft_micro_batches = sft_data.split(self.config.ppo_micro_batch_size_per_gpu) # we changed the normalization for this

        metrics = {}
        self.actor_optimizer.zero_grad()
        for sft_micro_batch in sft_micro_batches:
            print("sft_micro_batches")
            sft_micro_batch = sft_micro_batch.to(get_device_id())
            sft_inputs = {**sft_micro_batch.batch, **sft_micro_batch.non_tensor_batch}
            _, _, _, sft_loss = self._forward_micro_batch_for_input_ttt(sft_inputs, get_hidden_states=False, get_response=False, get_entropy=False, get_loss=True)
            sft_loss.backward() 
            
            sft_metrics = {
                "actor/sft_loss_ttt": sft_loss.detach().item(),
            }
            append_to_dict(metrics, sft_metrics)
                
        grad_norm = self._optimizer_step()
        batch_metrics = {"actor/grad_norm_ttt": grad_norm.detach().item()}
        append_to_dict(metrics, batch_metrics)

        self.actor_optimizer.zero_grad()
        return metrics
