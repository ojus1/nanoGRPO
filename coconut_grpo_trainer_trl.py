import random
import warnings
from typing import Union, Any, Optional, Callable

from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch

import torch
import torch.utils.data
from accelerate.utils import broadcast_object_list, gather, gather_object
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_peft_available

from ..import_utils import is_vllm_available
from ..models import unwrap_model_for_generation

from .grpo_trainer import GRPOTrainer, defaultdict, is_deepspeed_zero3_enabled, is_peft_model, create_reference_model
from .coconut_grpo_config import CoconutGRPOConfig
from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..models import unwrap_model_for_generation
from datasets import Dataset, IterableDataset

# New trainer that applies GRPO but where the model is taught to reason in latent space (Coconut).

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

class CoconutGRPOTrainer(GRPOTrainer):
    """
    A GRPOTrainer that integrates Coconut-style (latent/continuous thought) reasoning.
    At each rollout step, a per-example coin flip (guided by a latent probability that increases during training)
    decides whether to use latent reasoning or traditional token-based reasoning.
    
    When latent reasoning is chosen, the prompt is modified to append a special marker token ("<bot>"),
    and a number `c` (sampled from [1, max_continuous_tokens] with a ramped-up maximum) is used to set the
    number of continuous thought tokens the model should generate.
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: CoconutGRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        prompt_preprocess_prehook: Optional[Callable[[str], str]] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs
        )
        
        # Store the prompt preprocessing hook if provided
        if prompt_preprocess_prehook is not None:
            self.prompt_preprocess_prehook = prompt_preprocess_prehook

        # Reference model handling
        self.beta = args.beta
        if self.beta == 0.0 or is_peft_available() and is_peft_model(self.model):
            # Skip reference model if beta=0 or using PEFT
            self.ref_model = None
        else:
            # Load reference model as before
            if is_deepspeed_zero3_enabled():
                raise NotImplementedError("DeepSpeed Zero 3 is not supported for coconut grpo trainer.")
            else:
                self.ref_model = create_reference_model(model)

        # Initialize metrics with specific keys for Coconut
        self._metrics = {
            "train": defaultdict(list),
            "eval": defaultdict(list)
        }
        
        # Pre-initialize all the metric keys we'll use
        for mode in ["train", "eval"]:
            self._metrics[mode]["latent_reward"] = []
            self._metrics[mode]["latent_reward_std"] = []
            self._metrics[mode]["token_reward"] = []
            self._metrics[mode]["token_reward_std"] = []
            self._metrics[mode]["policy_loss"] = []
            self._metrics[mode]["clip_ratio"] = []  # Add metric for clip ratio
            if self.beta > 0.0 and self.ref_model is not None:
                self._metrics[mode]["kl"] = []

        self.log_completions = args.log_completions
        
        # Multi-step optimization parameters
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon = args.epsilon
        # Tracks the number of iterations (forward + backward passes)
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

    def _sample_continuous_tokens(self, current_step: int) -> int:
        """
        Sample the number c of continuous tokens from [1, max_continuous_tokens]. The maximum allowed value
        increases gradually over training.
        """
        rampup = min(1.0, current_step / self.args.continuous_tokens_rampup_steps)
        # Current maximum allowed continuous tokens (at least 1)
        current_max = 1 + (self.args.max_continuous_tokens - 1) * rampup
        current_max_int = max(1, int(round(current_max)))
        return random.randint(1, current_max_int)

    def _current_latent_prob(self, current_step: int) -> float:
        """
        Compute the current probability p_latent to use the latent reasoning branch.
        This probability is linearly increased from latent_initial_prob to latent_final_prob.
        """
        rampup = min(1.0, current_step / self.args.latent_prob_rampup_steps)
        return self.args.latent_initial_prob + (self.args.latent_final_prob - self.args.latent_initial_prob) * rampup

    def _generate_completion_for_prompt(self, prompt_text: str, device: torch.device) -> tuple:
        """
        Generate a completion for one prompt.
        Depending on a coin flip (with probability p_latent) perform latent (continuous thought) generation.
        In latent branch, the prompt is appended with "<bot>", and a number `c` (sampled from [1, max_continuous_tokens] with a ramped-up maximum) is used to set the
        number of continuous thought tokens the model should generate.
        In the standard branch, the generation proceeds as in GRPOTrainer.
        Returns:
            A tuple (completion, mode) where mode is "latent" or "token".
        """
        current_step = self.state.global_step if hasattr(self.state, "global_step") else 0
        p_latent = self._current_latent_prob(current_step)
        
        if random.random() < p_latent:
            # --- Latent reasoning branch ---
            c = self._sample_continuous_tokens(current_step)
            # Append special tokens with latent length specification
            latent_prompt = f"{prompt_text} <bot><num_thoughts={c}>"
            latent_prompt_tokens = self.processing_class(
                latent_prompt, return_tensors="pt", add_special_tokens=False
            ).to(device)

            # Force generation to return latent states (which include continuous embeddings)
            latent_output = self.model.generate(
                latent_prompt_tokens["input_ids"],
                input_text=latent_prompt,
                eos_token_id=self.model.eot_token_id,
                generation_config=self.generation_config,
                return_latent_states=True
            )
            return latent_output, "latent"
        else:
            # --- Standard (token) reasoning branch ---
            token_prompt_tokens = self.processing_class(
                prompt_text, return_tensors="pt", add_special_tokens=False
            ).to(device)
            gen_kwargs = {"max_new_tokens": self.generation_config.max_new_tokens,
                          "attention_mask": token_prompt_tokens["attention_mask"],
                          "input_text": prompt_text,
                          "eos_token_id": self.processing_class.eos_token_id}   # Pass input_text and eos_token_id to satisfy assertion
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                token_generated = unwrapped_model.generate(token_prompt_tokens["input_ids"], **gen_kwargs)
            token_completion = token_generated[0][token_prompt_tokens["input_ids"].size(1):]
            return token_completion, "token"

    def _prepare_inputs(self, inputs):
        """
        Prepares a batch of inputs for the GRPO update.
        In multi-step optimization, this method will either generate new completions
        or reuse previously stored ones based on the current iteration.
        """
        mode = "eval" if self.control.should_evaluate else "train"
        
        if mode == "train":
            # If we're at the start of a new multi-step cycle, generate new completions
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._prepare_sample_with_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                # Otherwise, reuse the stored completions from the previous step
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, always generate new completions
            inputs = self._prepare_sample_with_completions(inputs)
        
        return inputs

    def _prepare_sample_with_completions(self, inputs):
        """
        Helper method that handles the core completion generation and processing.
        This separates the completion generation logic from the reuse logic.
        """
        device = self.accelerator.device
        # Extract prompts from inputs.
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        # Tokenize prompts for later use in computing loss.
        prompt_ids_list = []
        prompt_masks_list = []
        for p_text in prompts_text:
            pt = self.processing_class(p_text, return_tensors="pt", padding=False, add_special_tokens=False).to(device)
            prompt_ids_list.append(pt["input_ids"][0])
            prompt_masks_list.append(pt["attention_mask"][0])
        max_prompt_length = max(p.size(0) for p in prompt_ids_list)
        padded_prompts = []
        padded_prompt_masks = []
        for p_ids, mask in zip(prompt_ids_list, prompt_masks_list):
            pad_len = max_prompt_length - p_ids.size(0)
            padding = torch.full((pad_len,), self.processing_class.pad_token_id, dtype=p_ids.dtype, device=device)
            mask_padding = torch.zeros(pad_len, dtype=mask.dtype, device=device)
            padded_prompts.append(torch.cat([p_ids, padding], dim=0))
            padded_prompt_masks.append(torch.cat([mask, mask_padding], dim=0))
        prompt_ids_tensor = torch.stack(padded_prompts, dim=0)  # (B, L_prompt)
        prompt_mask_tensor = torch.stack(padded_prompt_masks, dim=0)

        # ------------------------------
        # Generate completions for each prompt individually.
        completions_list = []
        modes = []  # record the branch ("latent" or "token") for each example
        for p_text in prompts_text:
            comp, mode = self._generate_completion_for_prompt(p_text, device)
            completions_list.append(comp)
            modes.append(mode)
        # Pad all completions to the same length.
        # For latent branch examples, use the continuous embeddings length to adjust answer_token_ids.
        comp_lengths = []
        for i, comp in enumerate(completions_list):
            if modes[i] == "latent":
                # Use the length of continuous embeddings for latent branch.
                latent_length = comp["input_embeds"].size(1)
                comp_lengths.append(latent_length)
            else:
                token_ids = comp if not isinstance(comp, dict) else comp["answer_token_ids"]
                if token_ids.dim() == 2 and token_ids.size(0) == 1:
                    token_ids = token_ids.squeeze(0)
                comp_lengths.append(token_ids.size(0))
        max_completion_length = max(comp_lengths)
        padded_completions = []
        for i, comp in enumerate(completions_list):
            if modes[i] == "latent":
                token_ids = comp["answer_token_ids"]
                if token_ids.dim() == 2 and token_ids.size(0) == 1:
                    token_ids = token_ids.squeeze(0)
                latent_length = comp["input_embeds"].size(1)
                # Adjust token_ids to match the continuous embeddings length.
                if token_ids.size(0) < latent_length:
                    pad_len = latent_length - token_ids.size(0)
                    padding = torch.full((pad_len,), self.processing_class.pad_token_id, dtype=token_ids.dtype, device=device)
                    token_ids = torch.cat([token_ids, padding], dim=0)
                else:
                    token_ids = token_ids[:latent_length]
                # Then pad to max_completion_length if needed.
                if token_ids.size(0) < max_completion_length:
                    pad_len = max_completion_length - token_ids.size(0)
                    padding = torch.full((pad_len,), self.processing_class.pad_token_id, dtype=token_ids.dtype, device=device)
                    token_ids = torch.cat([token_ids, padding], dim=0)
                padded_completions.append(token_ids)
            else:
                token_ids = comp if not isinstance(comp, dict) else comp["answer_token_ids"]
                if token_ids.dim() == 2 and token_ids.size(0) == 1:
                    token_ids = token_ids.squeeze(0)
                pad_len = max_completion_length - token_ids.size(0)
                padding = torch.full((pad_len,), self.processing_class.pad_token_id, dtype=token_ids.dtype, device=device)
                padded_completions.append(torch.cat([token_ids, padding], dim=0))
    
        completion_ids = torch.stack(padded_completions, dim=0)  # (B, L_completion)

        # Concatenate prompt and completion tokens.
        prompt_completion_ids = torch.cat([prompt_ids_tensor, completion_ids], dim=1)

        # Create a completion mask, masking out tokens after the first EOS.
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        for i in range(is_eos.size(0)):
            if is_eos[i].any():
                eos_idx[i] = int(is_eos[i].nonzero(as_tuple=False)[0].item())
        seq_idx = torch.arange(is_eos.size(1), device=device).unsqueeze(0).expand(is_eos.size(0), -1)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()

        # Concatenate the prompt mask and completion mask.
        attention_mask = torch.cat([prompt_mask_tensor, completion_mask], dim=1)

        # Build input embeddings per example:
        batch_embeds = []
        batch_size = prompt_ids_tensor.size(0)
        for i in range(batch_size):
            # Compute prompt embeddings from discrete tokens.
            prompt_embeds = self.model.get_input_embeddings()(prompt_ids_list[i].unsqueeze(0))
            if modes[i] == "latent":
                # Use the continuous embeddings from the latent generation output.
                # Assume the generation output (for latent branch) is a dict containing "input_embeds"
                completion_embeds = completions_list[i]["input_embeds"]
            else:
                # In token mode, convert discrete tokens to embeddings.
                completion_ids = padded_completions[i].unsqueeze(0)
                completion_embeds = self.model.get_input_embeddings()(completion_ids)
            full_embeds = torch.cat([prompt_embeds, completion_embeds], dim=1)
            batch_embeds.append(full_embeds)
        inputs_embeds = torch.cat(batch_embeds, dim=0)
        inputs_embeds.requires_grad_(True)
        # Decode the generated completions.
        completions_text = [
            self.processing_class.decode(comp, skip_special_tokens=True)
            for comp in padded_completions
        ]
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": ct}] for ct in completions_text]
        else:
            completions = completions_text

        logits_to_keep = completion_ids.size(1)  # number of tokens in completions

        with torch.inference_mode():
            # When using num_iterations > 1, compute and store old_per_token_logps
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None
            
            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # --- New Reward Computation Logic ---
        # Split examples based on mode.
        import numpy as np
        modes_np = np.array(modes)
        latent_indices = torch.tensor(np.where(modes_np == "latent")[0], dtype=torch.long, device=device)
        token_indices = torch.tensor(np.where(modes_np == "token")[0], dtype=torch.long, device=device)

        # Compute rewards for latent branch.
        if latent_indices.numel() > 0:
            latent_prompts = [prompts[i] for i in latent_indices.tolist()]
            latent_completions = [completions_text[i] for i in latent_indices.tolist()]
            # For latent completions, use text after <eot> for reward computation.
            processed_latent_completions = [
                text.split("<eot>", 1)[-1] if "<eot>" in text else text
                for text in latent_completions
            ]
            latent_rewards_per_func = torch.zeros(len(latent_prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                if isinstance(reward_func, nn.Module):
                    if is_conversational(inputs[0]):
                        # The conversation mode: wrap as messages.
                        messages = [
                            {"messages": p + [{"role": "assistant", "content": c}]}
                            for p, c in zip(latent_prompts, processed_latent_completions)
                        ]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(latent_prompts, processed_latent_completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    ).to(device)
                    with torch.inference_mode():
                        latent_rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=latent_prompts, completions=processed_latent_completions, **reward_kwargs)
                    latent_rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
            latent_rewards = latent_rewards_per_func.sum(dim=1)
            latent_rewards_std = latent_rewards.std()
        else:
            latent_rewards = torch.tensor([], device=device)
            latent_rewards_std = torch.tensor([], device=device)

        # Compute rewards for token branch.
        if token_indices.numel() > 0:
            token_prompts = [prompts[i] for i in token_indices.tolist()]
            token_completions = [completions_text[i] for i in token_indices.tolist()]
            token_rewards_per_func = torch.zeros(len(token_prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                if isinstance(reward_func, nn.Module):
                    if is_conversational(inputs[0]):
                        messages = [
                            {"messages": p + [{"role": "assistant", "content": c}]}
                            for p, c in zip(token_prompts, token_completions)
                        ]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(token_prompts, token_completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    ).to(device)
                    with torch.inference_mode():
                        token_rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=token_prompts, completions=token_completions, **reward_kwargs)
                    token_rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
            token_rewards = token_rewards_per_func.sum(dim=1)
            token_rewards_std = token_rewards.std()
        else:
            token_rewards = torch.tensor([], device=device)
            token_rewards_std = torch.tensor([], device=device)

        # Combine rewards from both branches.
        rewards = torch.zeros(len(prompts), device=device)
        if latent_indices.numel() > 0:
            for j, latent_idx in enumerate(latent_indices.tolist()):
                rewards[latent_idx] = latent_rewards[j]
        if token_indices.numel() > 0:
            for j, token_idx in enumerate(token_indices.tolist()):
                rewards[token_idx] = token_rewards[j]

        # Determine current mode ("eval" if evaluating, else "train")
        mode = "eval" if self.control.should_evaluate else "train"

        # Log aggregated per-branch rewards for analysis.
        self._metrics[mode]["latent_reward"].append(
            latent_rewards.mean().item() if latent_rewards.numel() > 0 else 0.0
        )
        self._metrics[mode]["latent_reward_std"].append(
            latent_rewards_std.item() if latent_rewards_std.numel() > 0 else 0.0
        )
        self._metrics[mode]["token_reward"].append(
            token_rewards.mean().item() if token_rewards.numel() > 0 else 0.0
        )
        self._metrics[mode]["token_reward_std"].append(
            token_rewards_std.item() if token_rewards_std.numel() > 0 else 0.0
        )

        # Log overall rewards for each branch.
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)
            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "answer_token_ids": completion_ids,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": rewards,
        } 

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute GRPO loss using embeddings and latent states with multi-step optimization."""
        if return_outputs:
            raise ValueError("The CoconutGRPOTrainer does not support returning outputs")

        # Get the current mode for metrics
        mode = "eval" if self.control.should_evaluate else "train"
        
        # Ensure inputs_embeds has gradients enabled
        inputs_embeds = inputs["inputs_embeds"]
        if not inputs_embeds.requires_grad:
            inputs_embeds.requires_grad_(True)

        # Forward pass with embeddings
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            use_cache=False
        )
        
        logits = outputs.logits
        
        # Get logits only for answer tokens
        answer_tokens = inputs["answer_token_ids"]
        answer_mask = (answer_tokens != self.processing_class.pad_token_id).float()
        
        # Check if answer_mask is not all zeros or empty
        if answer_mask.sum() == 0:
            # If we have no valid tokens, use a dummy loss
            device = inputs_embeds.device
            
            # Add dummy values to metrics to avoid division by zero in logging
            self._metrics[mode]["policy_loss"].append(0.0)
            self._metrics[mode]["clip_ratio"].append(0.0)
            if self.beta > 0.0 and self.ref_model is not None:
                self._metrics[mode]["kl"].append(0.0)
            
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute log probabilities for answer tokens
        log_probs = torch.log_softmax(logits, dim=-1)
        answer_log_probs = torch.gather(log_probs, -1, answer_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Mask out padding tokens
        masked_log_probs = answer_log_probs * answer_mask
        
        # Get advantages
        advantages = inputs["advantages"].unsqueeze(-1)
        
        # Implement multi-step optimization - importance sampling and clipping
        if self.num_iterations > 1 and "old_per_token_logps" in inputs:
            old_per_token_logps = inputs["old_per_token_logps"]
            # Ensure shapes are compatible
            if old_per_token_logps.shape != masked_log_probs.shape:
                # Resize or recreate old_per_token_logps to match masked_log_probs shape
                print(f"Shape mismatch: old_per_token_logps {old_per_token_logps.shape}, masked_log_probs {masked_log_probs.shape}")
                
                # If sequence dimensions don't match, we need to handle this
                if masked_log_probs.size(1) == 0:
                    # If masked_log_probs has empty sequence dimension, use a dummy value
                    device = inputs_embeds.device
                    masked_log_probs = torch.zeros_like(old_per_token_logps)
                else:
                    # Reshape old_per_token_logps to match masked_log_probs
                    old_per_token_logps = old_per_token_logps[:, :masked_log_probs.size(1)]
        else:
            old_per_token_logps = masked_log_probs.detach()
        
        # Importance sampling ratio
        coef_1 = torch.exp(masked_log_probs - old_per_token_logps)
        
        # Clipped importance ratio
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        
        # Compute unclipped and clipped objectives
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        
        # Take minimum to implement pessimistic bound
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        # Add KL penalty only if beta > 0 and ref_model exists
        if self.beta > 0.0 and self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    inputs_embeds=inputs_embeds.detach(),  # Detach for ref model
                    attention_mask=inputs["attention_mask"]
                )
                ref_log_probs = torch.log_softmax(ref_outputs.logits, dim=-1)
                ref_answer_log_probs = torch.gather(ref_log_probs, -1, answer_tokens.unsqueeze(-1)).squeeze(-1)
                
            kl_div = (masked_log_probs - ref_answer_log_probs.detach()) * answer_mask
            kl_penalty = self.beta * kl_div.sum(dim=1) / (answer_mask.sum(dim=1) + 1e-8)  # Add epsilon to avoid div by zero
            
            loss = per_token_loss.sum(dim=1) / (answer_mask.sum(dim=1) + 1e-8) + kl_penalty
            loss = loss.mean()
            
            # Log KL metrics only when using KL penalty
            mean_kl = ((kl_div * answer_mask).sum(dim=1) / (answer_mask.sum(dim=1) + 1e-8)).mean()
            mode = "eval" if self.control.should_evaluate else "train"
            self._metrics[mode]["kl"].append(mean_kl.item())
        else:
            # Normalize by token count and take mean across batch
            loss = (per_token_loss * answer_mask).sum(dim=1) / (answer_mask.sum(dim=1) + 1e-8)
            loss = loss.mean()

        # Log metrics
        mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[mode]["policy_loss"].append(per_token_loss.mean().item())
        
        # Log clip ratio metric
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * answer_mask).sum() / (answer_mask.sum() + 1e-8)
        self._metrics[mode]["clip_ratio"].append(clip_ratio.item())

        return loss

    def _get_per_token_logps(self, model, prompt_completion_ids, attention_mask, logits_to_keep):
        """
        Override _get_per_token_logps to use inputs_embeds rather than input_ids.
        This is needed because the Qwen2ForCausalLM model does not support input_ids.
        Instead, we compute the embeddings from prompt_completion_ids and pass them
        to the model so that the latent tokens (and token-based decoded tokens) are used.
        """
        # Convert token IDs to embeddings
        inputs_embeds = model.get_input_embeddings()(prompt_completion_ids)
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        
        # Get logits only for the completion part
        logits = outputs.logits[:, -logits_to_keep:, :]
        
        # Get token IDs for the completion part
        completion_ids = prompt_completion_ids[:, -logits_to_keep:]
        
        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Gather the log probabilities for each token
        batch_size, seq_length = completion_ids.size()
        token_log_probs = torch.zeros((batch_size, seq_length), device=logits.device)
        
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_length):
                token_id = completion_ids[batch_idx, seq_idx]
                if token_id < log_probs.size(2):  # Ensure token_id is within vocab size
                    token_log_probs[batch_idx, seq_idx] = log_probs[batch_idx, seq_idx, token_id]
        
        return token_log_probs 

    def log(self, logs, start_time):
        """
        Override log method to safely handle empty metrics lists.
        This prevents division by zero errors when computing metric averages.
        """
        # Get the current mode (train or eval)
        mode = "eval" if self.control.should_evaluate else "train"
        
        # Safely compute averages, avoiding division by zero
        metrics = {}
        for key, val in self._metrics[mode].items():
            if val:  # Check if list is non-empty
                metrics[key] = sum(val) / len(val)
            else:
                # Use a default value of 0.0 for empty metrics
                metrics[key] = 0.0
                print(f"Warning: Empty metric list for {key} in {mode} mode")
        
        # Clear metrics for the current mode
        for val in self._metrics[mode].values():
            val.clear()
        
        # Add metrics to logs
        logs.update(metrics)
        
        # Call parent's log method (skip GRPOTrainer's implementation)
        super(GRPOTrainer, self).log(logs, start_time) 