import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model

DEFAULT_LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj"


def parse_target_modules(target_modules: str):
    if target_modules is None:
        return []
    if isinstance(target_modules, str):
        modules = [module.strip() for module in target_modules.split(",") if module.strip()]
    elif isinstance(target_modules, (list, tuple, set)):
        modules = [str(module).strip() for module in target_modules if str(module).strip()]
    else:
        modules = [str(target_modules).strip()] if str(target_modules).strip() else []
    return modules or ["all-linear"]


def build_lora_config(model_args, target_modules: str = None):
    selected_modules = parse_target_modules(target_modules or model_args.lora_target_modules)
    return {
        "r": model_args.lora_r,
        "lora_alpha": model_args.lora_alpha,
        "lora_dropout": model_args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": selected_modules if len(selected_modules) > 1 else selected_modules[0],
    }

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    merge_size: int = field(default=16, metadata={"help": "tokens per memory slot (COMI-style pooling)"})
    merge_sizes: str = field(default="16,32", metadata={"help": "candidate merge sizes for random sampling, comma separated"})
    is_random: bool = field(default=False, metadata={"help": "if true, sample merge size from merge_sizes for each batch"})
    segment_size: int = field(default=10000, metadata={"help": "segment size for long context compression"})
    compress_ratio: int = field(default=0, metadata={"help": "deprecated alias of merge_size"})
    coarse_grained_on: bool = field(default=True, metadata={"help": "enable coarse-grained group reallocation"})
    fine_grained_on: bool = field(default=True, metadata={"help": "enable fine-grained weighted token merge"})
    redun_coarse: bool = field(default=True, metadata={"help": "use redundancy in coarse-grained MIG scoring"})
    redun_fine: bool = field(default=True, metadata={"help": "use redundancy in fine-grained MIG scoring"})
    lamda_select: bool = field(default=False, metadata={"help": "learn coarse alpha/beta parameters"})
    lamda_merge: bool = field(default=False, metadata={"help": "learn fine alpha/beta parameters"})
    alpha_select_init: float = field(default=1.0)
    beta_select_init: float = field(default=1.0)
    alpha_merge_init: float = field(default=1.0)
    beta_merge_init: float = field(default=1.0)
    use_flash_attention_2: bool = field(default=False, metadata={"help": "enable flash attention 2"})
    use_transform_layer: bool = field(default=True, metadata={"help": "use decoder block(s) as LSA memory fusion layer"})
    num_mem_fusion_layers: int = field(default=1, metadata={"help": "number of decoder blocks used as LSA"})
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_alpha: int = field(default=32, metadata={"help": "lora alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    lora_target_modules: str = field(default=DEFAULT_LORA_TARGET_MODULES, metadata={"help": "comma separated LoRA target modules"})
    encoder_lora_target_modules: str = field(default="", metadata={"help": "optional encoder-specific LoRA target modules"})
    decoder_lora_target_modules: str = field(default="", metadata={"help": "optional decoder-specific LoRA target modules"})
    lsa_lora_target_modules: str = field(default="", metadata={"help": "optional LSA-specific LoRA target modules"})
    full_finetune_encoder: bool = field(default=True, metadata={"help": "full finetune encoder backbone"})
    lora_encoder: bool = field(default=False, metadata={"help": "apply LoRA to encoder"})
    full_finetune_decoder: bool = field(default=False, metadata={"help": "full finetune decoder backbone"})
    lora_decoder: bool = field(default=True, metadata={"help": "apply LoRA to decoder"})
    full_finetune_lsa: bool = field(default=True, metadata={"help": "full finetune LSA transformer blocks"})
    lora_lsa: bool = field(default=False, metadata={"help": "apply LoRA to LSA memory fusion layer"})
    train: bool = field(default=False, metadata={"help": "if true, the model ckpt will be initialized for training; else, it's for inference"})
    encoder_layers: int = field(default=8, metadata={"help": "number of encoder layers"})
@dataclass
class DataArguments:
    train_file: str = field(default="/path/to/train.jsonl", metadata={"help": "Path to the training data."})
    test_file: str = field(default="/path/to/test.jsonl", metadata={"help": "Path to the test data."})
    debug_data: bool = field(default=False, metadata={"help": "Enable debug dataset"})
    test_sample_index: int = field(default=0, metadata={"help": "Index of the sample to test in the dataset."})
    num_samples: int = field(default=0, metadata={"help": "Number of eval samples to use; 0 means all"})
    shuffle_samples: bool = field(default=False, metadata={"help": "Shuffle test samples before slicing"})
@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=28000,
        metadata={"help": "Maximum sequence length."},
    )
    report_to: Optional[str] = field(default="wandb")
    project_name: Optional[str] = field(default="cluster")
    max_steps: int = field(default=10000, metadata={"help": "max steps of training"})
    save_strategy: Optional[str] = field(default="steps")
    save_steps: int = field(default=10000, metadata={"help": "interval of saving checkpoints in steps"})
    eval_strategy: Optional[str] = field(default="steps")
    eval_steps: int = field(default=20000, metadata={"help": "interval of evaluation in steps"})
    num_train_epochs: int = field(default=1)
    add_special_token_for_lm: bool = field(default=False)
    restore_from: str = field(default="", metadata={"help": "checkpoint to restore from"})
    overwrite_output_dir: bool = field(default=True)
    logging_steps: int = field(default=100)
    deepspeed: str = field(default="")
    bf16: bool = field(default=True, metadata={"help": "Use bfloat16"})
    gradient_accumulation_steps: int = field(default=1)
    optim: str = field(default="adamw_torch")
    per_device_train_batch_size: int = field(default=1)
    lr_scheduler_type: str = field(default="cosine")
    learning_rate: float = field(default=1e-5)
    gradient_checkpointing: bool = field(default=True)
    warmup_ratio: float = field(default=0.1)
    weight_decay: float = field(default=0.01)
    seed: int = field(default=42)

def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.2f}")

def load_checkpoint_file(path):
    if os.path.isdir(path):
        candidate_files = [
            os.path.join(path, "model.safetensors"),
            os.path.join(path, "adapter_model.safetensors"),
            os.path.join(path, "pytorch_model.bin"),
        ]
        ckpt_file = next((candidate for candidate in candidate_files if os.path.exists(candidate)), None)
        if ckpt_file is None:
            raise FileNotFoundError(f"No supported checkpoint file found in directory: {path}")
    else:
        ckpt_file = path

    if ckpt_file.endswith(".safetensors"):
        state_dict = load_file(ckpt_file)
    else:
        state_dict = torch.load(ckpt_file, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    state_dict = {
        "encoder." + key[len("icae."):] if key.startswith("icae.") else key: value
        for key, value in state_dict.items()
    }
    return state_dict, ckpt_file

class GroupReallocator:
    def __init__(self, alpha_select, beta_select, redun_coarse=True):
        self.alpha_select = alpha_select
        self.beta_select = beta_select
        self.redun_coarse = redun_coarse

    def reallocate_group_size(self, group_size, context_embeddings, query_embedding):
        n_tokens, _ = context_embeddings.shape
        n_group = math.ceil(n_tokens / group_size)
        if n_group <= 1:
            return torch.tensor(
                [n_tokens], device=context_embeddings.device, dtype=torch.long
            )
        context_normalized = F.normalize(context_embeddings, p=2, dim=1)
        query_normalized = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1).squeeze(0)
        relevance = context_normalized @ query_normalized
        padded_length = n_group * group_size
        padded_relevance = F.pad(
            relevance,
            (0, padded_length - n_tokens),
            value=-torch.inf,
        ).view(n_group, group_size)
        group_relevance, local_max_indices = padded_relevance.max(dim=1)
        group_offsets = torch.arange(
            n_group, device=context_embeddings.device, dtype=torch.long
        ) * group_size
        representative_indices = group_offsets + local_max_indices
        pooled = context_embeddings[representative_indices]

        if self.redun_coarse:
            pooled_normalized = F.normalize(pooled, p=2, dim=1)
            cosine_sim_matrix = pooled_normalized @ pooled_normalized.T
            cosine_sim_matrix.fill_diagonal_(-float("inf"))
            redundancy = cosine_sim_matrix.max(dim=1).values
        else:
            redundancy = torch.zeros_like(group_relevance)
        scores = self.alpha_select * group_relevance - self.beta_select * redundancy
        probs = F.softmax(-scores, dim=0)
        quotas = n_tokens * probs
        ints = torch.floor(quotas).long()
        remainder = n_tokens - ints.sum()
        frac = quotas - ints.to(quotas.dtype)
        ranked_indices = torch.topk(frac, k=n_group).indices
        ranked_positions = torch.arange(
            n_group, device=context_embeddings.device, dtype=torch.long
        )
        bonuses = torch.zeros_like(ints).scatter(
            0,
            ranked_indices,
            (ranked_positions < remainder).to(ints.dtype),
        )
        return ints + bonuses

class TKDR:
    def __init__(
        self,
        alpha_select,
        beta_select,
        alpha_merge,
        beta_merge,
        coarse_grained_on=True,
        fine_grained_on=True,
        redun_coarse=True,
        redun_fine=True,
    ):
        self.alpha_select = alpha_select
        self.beta_select = beta_select
        self.alpha_merge = alpha_merge
        self.beta_merge = beta_merge
        self.coarse_grained_on = coarse_grained_on
        self.fine_grained_on = fine_grained_on
        self.redun_coarse = redun_coarse
        self.redun_fine = redun_fine

    def batched_weighted_pooling_mrmr(self, group_embs, group_mask, query_norm):
        if group_embs.shape[0] == 0:
            return torch.empty((0, group_embs.shape[-1]), device=group_embs.device, dtype=group_embs.dtype)

        embs_norm = F.normalize(group_embs, p=2, dim=-1)
        rel2query = torch.matmul(embs_norm, query_norm)

        if self.redun_fine:
            sim_mat = torch.bmm(embs_norm, embs_norm.transpose(1, 2))
            sim_mat.diagonal(dim1=-2, dim2=-1).fill_(-float("inf"))
            mask_2d = group_mask.unsqueeze(1).expand(-1, group_mask.shape[1], -1)
            sim_mat.masked_fill_(~mask_2d, 0)
            max_redundancy = sim_mat.max(dim=-1).values
            max_redundancy = max_redundancy.masked_fill(max_redundancy == -float("inf"), 0.0)
        else:
            max_redundancy = torch.zeros_like(rel2query)

        mrmr_scores = self.alpha_merge * rel2query - self.beta_merge * max_redundancy
        mrmr_scores.masked_fill_(~group_mask, -torch.inf)
        weights = F.softmax(mrmr_scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * group_embs).sum(dim=1)
        return pooled.to(group_embs.dtype)

    def compress_by_mrmr(self, context_embeddings, query_embedding, compress_rate):
        n_tokens, hidden_size = context_embeddings.shape
        target_length = max(1, math.ceil(n_tokens / compress_rate))
        query_norm = F.normalize(query_embedding, dim=-1)

        if self.coarse_grained_on:
            reallocator = GroupReallocator(self.alpha_select, self.beta_select, redun_coarse=self.redun_coarse)
            reallocated_sizes = reallocator.reallocate_group_size(
                group_size=compress_rate,
                context_embeddings=context_embeddings,
                query_embedding=query_embedding,
            )
        else:
            num_groups = max(1, round(n_tokens / compress_rate))
            base_size = n_tokens // num_groups
            remainder = n_tokens % num_groups
            group_indices = torch.arange(
                num_groups, device=context_embeddings.device, dtype=torch.long
            )
            reallocated_sizes = torch.full(
                (num_groups,),
                base_size,
                device=context_embeddings.device,
                dtype=torch.long,
            ) + (group_indices < remainder).long()
        host_sizes = reallocated_sizes.detach().cpu()
        host_sizes = host_sizes[host_sizes > 0]
        num_groups = host_sizes.shape[0]
        max_group_len = host_sizes.max().item()
        reallocated_sizes = host_sizes.to(context_embeddings.device)
        group_starts = reallocated_sizes.cumsum(dim=0) - reallocated_sizes
        group_ids = torch.repeat_interleave(
            torch.arange(num_groups, device=context_embeddings.device),
            reallocated_sizes,
            output_size=n_tokens,
        )
        positions = torch.arange(n_tokens, device=context_embeddings.device) - torch.repeat_interleave(
            group_starts,
            reallocated_sizes,
            output_size=n_tokens,
        )
        group_embs_padded = torch.zeros(num_groups, max_group_len, hidden_size, device=context_embeddings.device, dtype=context_embeddings.dtype)
        group_mask = torch.zeros(num_groups, max_group_len, device=context_embeddings.device, dtype=torch.bool)
        group_embs_padded[group_ids, positions] = context_embeddings
        group_mask[group_ids, positions] = True

        if self.fine_grained_on:
            pooled_embs = self.batched_weighted_pooling_mrmr(group_embs_padded, group_mask, query_norm)
        else:
            masked_embs = group_embs_padded * group_mask.unsqueeze(-1)
            pooled_embs = masked_embs.sum(dim=1) / group_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return pooled_embs[:target_length]

class COMI(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config=None):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.lora_config = lora_config
        self.model_name = model_args.model_name_or_path
        encoder_config = AutoConfig.from_pretrained(self.model_name)
        encoder_config.num_hidden_layers = model_args.encoder_layers
        self.encoder = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=encoder_config,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            # use_flash_attention_2=model_args.use_flash_attention_2,
            # resume_download=True,
            trust_remote_code=True
        )
        self.training_mode = model_args.train
        self.decoder = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            # use_flash_attention_2=model_args.use_flash_attention_2,
            # resume_download=True,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.dim = self.encoder.config.hidden_size
        self.merge_size = model_args.compress_ratio if model_args.compress_ratio > 0 else model_args.merge_size
        self.merge_sizes = [int(x.strip()) for x in model_args.merge_sizes.split(",") if x.strip()]
        self.is_random = model_args.is_random
        self.segment_size = model_args.segment_size
        self.use_transform_layer = model_args.use_transform_layer
        self.coarse_grained_on = model_args.coarse_grained_on
        self.fine_grained_on = model_args.fine_grained_on
        self.redun_coarse = model_args.redun_coarse
        self.redun_fine = model_args.redun_fine

        decoder_hidden_size = self.decoder.config.hidden_size
        self.semantic_alignment_layer = nn.Linear(self.dim, decoder_hidden_size).to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16
        )
        if self.use_transform_layer:
            fusion_config = AutoConfig.from_pretrained(self.model_name)
            fusion_config.num_hidden_layers = model_args.num_mem_fusion_layers
            self.memory_fusion_layer = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=fusion_config,
                torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
                # use_flash_attention_2=model_args.use_flash_attention_2,
                # resume_download=True,
                trust_remote_code=True,
            )

        self.encoder_uses_lora = model_args.lora_encoder and self.lora_config is not None
        self.decoder_uses_lora = model_args.lora_decoder and self.lora_config is not None
        self.lsa_uses_lora = self.use_transform_layer and model_args.lora_lsa and self.lora_config is not None
        self._apply_lora_adapters()

        self.cos = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        if model_args.lamda_select:
            self.alpha_select = nn.Parameter(torch.tensor(model_args.alpha_select_init, dtype=torch.float32))
            self.beta_select = nn.Parameter(torch.tensor(model_args.beta_select_init, dtype=torch.float32))
        else:
            self.register_buffer("alpha_select", torch.tensor(model_args.alpha_select_init, dtype=torch.float32))
            self.register_buffer("beta_select", torch.tensor(model_args.beta_select_init, dtype=torch.float32))

        if model_args.lamda_merge:
            self.alpha_merge = nn.Parameter(torch.tensor(model_args.alpha_merge_init, dtype=torch.float32))
            self.beta_merge = nn.Parameter(torch.tensor(model_args.beta_merge_init, dtype=torch.float32))
        else:
            self.register_buffer("alpha_merge", torch.tensor(model_args.alpha_merge_init, dtype=torch.float32))
            self.register_buffer("beta_merge", torch.tensor(model_args.beta_merge_init, dtype=torch.float32))

        self.tkdr = TKDR(
            self.alpha_select,
            self.beta_select,
            self.alpha_merge,
            self.beta_merge,
            coarse_grained_on=self.coarse_grained_on,
            fine_grained_on=self.fine_grained_on,
            redun_coarse=self.redun_coarse,
            redun_fine=self.redun_fine,
        )
        self._configure_trainability()

        if self.training_mode:
            self.init()

    def _set_requires_grad(self, module, flag):
        for param in module.parameters():
            param.requires_grad = flag

    def _wrap_with_lora(self, module, override_target_modules: str = ""):
        if self.lora_config is None:
            return module
        if isinstance(self.lora_config, dict):
            config_kwargs = copy.deepcopy(self.lora_config)
            chosen_modules = parse_target_modules(override_target_modules or config_kwargs.get("target_modules", self.model_args.lora_target_modules))
            config_kwargs["target_modules"] = chosen_modules if len(chosen_modules) > 1 else chosen_modules[0]
            config = LoraConfig(**config_kwargs)
        else:
            config_kwargs = copy.deepcopy(self.lora_config.to_dict())
            chosen_modules = parse_target_modules(override_target_modules or config_kwargs.get("target_modules", self.model_args.lora_target_modules))
            config_kwargs["target_modules"] = chosen_modules if len(chosen_modules) > 1 else chosen_modules[0]
            config = LoraConfig(**config_kwargs)
        return get_peft_model(module, config)

    def _apply_lora_adapters(self):
        if self.encoder_uses_lora:
            self.encoder = self._wrap_with_lora(self.encoder, self.model_args.encoder_lora_target_modules)
        if self.decoder_uses_lora:
            self.decoder = self._wrap_with_lora(self.decoder, self.model_args.decoder_lora_target_modules)
        if self.lsa_uses_lora and hasattr(self, "memory_fusion_layer"):
            self.memory_fusion_layer = self._wrap_with_lora(self.memory_fusion_layer, self.model_args.lsa_lora_target_modules)

    def _enable_lora_parameters(self, module):
        for name, param in module.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def _configure_trainability(self):
        if not self.training_mode:
            self.encoder.eval()
            self.decoder.eval()
            if self.use_transform_layer and hasattr(self, "memory_fusion_layer"):
                self.memory_fusion_layer.eval()
            return

        if self.use_transform_layer and hasattr(self, "memory_fusion_layer"):
            if self.model_args.full_finetune_lsa:
                self._set_requires_grad(self.memory_fusion_layer, True)
            elif self.lsa_uses_lora:
                self._set_requires_grad(self.memory_fusion_layer, False)
                self._enable_lora_parameters(self.memory_fusion_layer)
            else:
                self._set_requires_grad(self.memory_fusion_layer, False)
            self._set_requires_grad(self.semantic_alignment_layer, False)
        else:
            self._set_requires_grad(self.semantic_alignment_layer, True)

        if self.model_args.full_finetune_encoder:
            self._set_requires_grad(self.encoder, True)
        else:
            self._set_requires_grad(self.encoder, False)
        if self.encoder_uses_lora:
            self._enable_lora_parameters(self.encoder)

        if self.model_args.full_finetune_decoder:
            self._set_requires_grad(self.decoder, True)
        elif self.decoder_uses_lora:
            self._set_requires_grad(self.decoder, False)
            self._enable_lora_parameters(self.decoder)
        else:
            self._set_requires_grad(self.decoder, False)
            tuned_keys = ("q_proj", "k_proj", "v_proj", "o_proj")
            for name, param in self.decoder.named_parameters():
                if any(key in name for key in tuned_keys):
                    param.requires_grad = True

    def init(self):
        print_trainable_parameters(self)
        if self.training_args.restore_from is not None and self.training_args.restore_from != "":
            print(f"Loading from the pretrained checkpoint: {self.training_args.restore_from}...")
            state_dict, ckpt_file = load_checkpoint_file(self.training_args.restore_from)
            self.load_state_dict(state_dict, strict=False)
            print(f"Finished loading from {self.training_args.restore_from}")
        # print("Enabling gradient checkpointing...")
        # if any(p.requires_grad for p in self.encoder.parameters()):
        #     self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # if any(p.requires_grad for p in self.decoder.parameters()):
        #     self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # if self.use_transform_layer and hasattr(self, "memory_fusion_layer"):
        #     if any(p.requires_grad for p in self.memory_fusion_layer.parameters()):
        #         self.memory_fusion_layer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def tokens_to_embeddings(self, token_ids):
        if hasattr(self.encoder, "get_base_model"):
            base_embeds = self.encoder.get_base_model().model.embed_tokens(token_ids)
        else:
            base_embeds = self.encoder.model.embed_tokens(token_ids)
        return base_embeds

    def generate_merge_size(self):
        if not self.merge_sizes:
            return self.merge_size
        return self.merge_sizes[torch.randint(0, len(self.merge_sizes), (1,)).item()]

    def split_to_segments(self, input_ids, attention_mask):
        total_len = input_ids.shape[1]
        num_segments = math.ceil(total_len / self.segment_size)
        if num_segments <= 1:
            return [input_ids], [attention_mask]
        segments = []
        segment_masks = []
        for i in range(num_segments):
            start = i * self.segment_size
            end = min((i + 1) * self.segment_size, total_len)
            segments.append(input_ids[:, start:end])
            segment_masks.append(attention_mask[:, start:end])
        return segments, segment_masks

    def generate_tkdr_memorys(
        self,
        input_ids,
        input_mask,
        merge_size,
        query_ids,
        query_input_mask,
    ):
        device = input_ids.device
        if self.is_random:
            merge_size = self.generate_merge_size()

        all_input_ids = torch.cat((input_ids, query_ids), dim=1)
        all_input_mask = torch.cat((input_mask, query_input_mask), dim=1)

        encoder_embeds = self.tokens_to_embeddings(all_input_ids)
        last_hidden_state = self.encoder(
            inputs_embeds=encoder_embeds,
            attention_mask=all_input_mask,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]
        batch_size = input_ids.shape[0]
        context_len = input_ids.shape[1]
        query_len = query_ids.shape[1]
        memorys_list = []
        memory_lengths = []
        for i in range(batch_size):
            context_mask = input_mask[i]
            context_hidden_state = last_hidden_state[i][:context_len, :]
            query_mask = query_input_mask[i]
            query_hidden_state = last_hidden_state[i][-query_len:, :]
            select_context_hidden_state = context_hidden_state[context_mask.bool()]
            select_query_hidden_state = query_hidden_state[query_mask.bool()]
            if select_context_hidden_state.shape[0] == 0:
                select_context_hidden_state = torch.zeros((1, context_hidden_state.shape[-1]), device=device, dtype=context_hidden_state.dtype)
            if select_query_hidden_state.shape[0] == 0:
                select_query_hidden_state = select_context_hidden_state
            query_embeds = torch.mean(select_query_hidden_state, dim=0)
            memory_embeds = self.tkdr.compress_by_mrmr(
                select_context_hidden_state,
                query_embeds,
                merge_size,
            )
            memorys_list.append(memory_embeds.unsqueeze(0))
            memory_lengths.append(memory_embeds.shape[0])

        max_len = max(e.shape[1] for e in memorys_list)
        final_memorys_list = []
        for idx, e in enumerate(memorys_list):
            pad_len = max_len - e.shape[1]
            if pad_len == 0:
                pad_memorys = e
            else:
                pad_embeds = torch.zeros(1, pad_len, e.shape[2], device=device, dtype=e.dtype)
                pad_memorys = torch.cat((e, pad_embeds), dim=1)
            final_memorys_list.append(pad_memorys)

        final_memorys = torch.cat(final_memorys_list, dim=0).to(last_hidden_state.dtype)
        memory_lengths = torch.tensor(memory_lengths, device=device)
        att_mask = torch.arange(max_len, device=device).unsqueeze(0) < memory_lengths.unsqueeze(1)
        nonzero_memory = (final_memorys != 0).reshape(batch_size, -1).any(dim=1)
        att_mask = att_mask & nonzero_memory.unsqueeze(1)

        if self.use_transform_layer and hasattr(self, "memory_fusion_layer"):
            aligned_memorys = self.memory_fusion_layer(
                inputs_embeds=final_memorys,
                attention_mask=att_mask,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[-1]
        else:
            decoder_hidden_size = self.decoder.config.hidden_size
            if final_memorys.shape[-1] != decoder_hidden_size:
                aligned_memorys = self.semantic_alignment_layer(final_memorys)
            else:
                aligned_memorys = final_memorys

        return aligned_memorys, att_mask

    def build_comi_memory(self, input_ids, attention_mask, query_ids, query_attention_mask):
        segments, segment_masks = self.split_to_segments(input_ids, attention_mask)
        all_memory = []
        all_masks = []
        for seg_ids, seg_mask in zip(segments, segment_masks):
            seg_memory, seg_memory_mask = self.generate_tkdr_memorys(
                input_ids=seg_ids,
                input_mask=seg_mask,
                merge_size=self.merge_size,
                query_ids=query_ids,
                query_input_mask=query_attention_mask,
            )
            all_memory.append(seg_memory)
            all_masks.append(seg_memory_mask)

        total_mem = torch.cat(all_memory, dim=1)
        total_mask = torch.cat(all_masks, dim=1)
        total_mem = total_mem.to(self.decoder.get_input_embeddings().weight.dtype)
        return total_mem, total_mask




    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        query_ids=None,
        query_attention_mask=None,
        labels=None,
    ):
        if query_ids is None or query_attention_mask is None:
            raise ValueError("`query_ids` and `query_attention_mask` are required for COMI decoding.")

        compressed_embeds, compressed_mask = self.build_comi_memory(
            input_ids=input_ids,
            attention_mask=attention_mask,
            query_ids=query_ids,
            query_attention_mask=query_attention_mask,
        )
        decoder_dtype = self.decoder.get_input_embeddings().weight.dtype
        compressed_embeds = compressed_embeds.to(decoder_dtype)
        query_embeds = self.decoder.get_input_embeddings()(query_ids).to(decoder_dtype)
        query_mask = query_attention_mask.bool()
        if labels is not None:
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = self.tokenizer.pad_token_id
            label_embeds = self.decoder.get_input_embeddings()(safe_labels)
            label_embeds = label_embeds.to(decoder_dtype)

            full_embeds = torch.cat([compressed_embeds, query_embeds, label_embeds], dim=1)
            label_mask = (labels != -100)
            full_mask = torch.cat([compressed_mask, query_mask, label_mask], dim=1)
            ignore_mem_labels = torch.full(compressed_mask.shape, -100, device=labels.device, dtype=labels.dtype)
            ignore_query_labels = torch.full(query_ids.shape, -100, device=labels.device, dtype=labels.dtype)
            full_labels = torch.cat([ignore_mem_labels, ignore_query_labels, labels], dim=1)
            decoder_outputs = self.decoder(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                labels=full_labels,
                return_dict=True
            )
            if self.training:
                return {"loss": decoder_outputs.loss}
            return decoder_outputs
        else:
            decoder_inputs_embeds = torch.cat([compressed_embeds, query_embeds], dim=1)
            decoder_attention_mask = torch.cat([compressed_mask, query_mask], dim=1).long()
            outputs = self.decoder.generate(
                inputs_embeds=decoder_inputs_embeds,
                attention_mask=decoder_attention_mask,
                max_new_tokens=20,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_id,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.3,
                top_p=0.9,
                use_cache=True
            )
            return outputs

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.encoder.gradient_checkpointing_enable(*args, **kwargs)
        self.decoder.gradient_checkpointing_enable(*args, **kwargs)
        if self.use_transform_layer and hasattr(self, "memory_fusion_layer"):
            self.memory_fusion_layer.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        self.encoder.gradient_checkpointing_disable(*args, **kwargs)
        self.decoder.gradient_checkpointing_disable(*args, **kwargs)
        if self.use_transform_layer and hasattr(self, "memory_fusion_layer"):
            self.memory_fusion_layer.gradient_checkpointing_disable(*args, **kwargs)
