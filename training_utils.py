import os

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint
import safetensors.torch

class InstructFTTokenizeFunction:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        required = ("input", "prompt", "answer")
        missing = [name for name in required if name not in examples]
        if missing:
            raise ValueError("Each batch must contain input, prompt, and answer columns.")

        all_context_ids = []
        all_query_ids = []
        all_labels = []
        eos_token = self.tokenizer.eos_token
        num_inputs = len(examples["input"])
        num_prompts = len(examples["prompt"])
        num_answers = len(examples["answer"])
        if not (num_inputs == num_prompts == num_answers):
            raise ValueError(
                "Mismatched example columns in InstructFTTokenizeFunction: "
                f"input={num_inputs}, prompt={num_prompts}, answer={num_answers}"
            )
        for i, p, a in zip(examples["input"], examples["prompt"], examples["answer"]):
            context_ids = self.tokenizer.encode(f"{i}", add_special_tokens=False)
            query_ids = self.tokenizer.encode(f"{p}", add_special_tokens=False)
            if isinstance(a, list):
                if len(a) == 0:
                    raise ValueError("Encountered empty answer list in InstructFTTokenizeFunction.")
                answer_ids = self.tokenizer.encode(f"{a[0]}{eos_token}", add_special_tokens=False)
            else:
                answer_ids = self.tokenizer.encode(f"{a}{eos_token}", add_special_tokens=False)

            all_context_ids.append(torch.tensor(context_ids, dtype=torch.long))
            all_query_ids.append(torch.tensor(query_ids, dtype=torch.long))
            all_labels.append(torch.tensor(answer_ids, dtype=torch.long))

        return {
            "input_ids": all_context_ids,
            "query_ids": all_query_ids,
            "labels": all_labels,
        }


class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = []
        query_ids = []
        labels = []
        for feature in features:
            input_ids.append(torch.as_tensor(feature["input_ids"], dtype=torch.long))
            query_ids.append(torch.as_tensor(feature["query_ids"], dtype=torch.long))
            labels.append(torch.as_tensor(feature["labels"], dtype=torch.long))

        batch_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        batch_query_ids = pad_sequence(query_ids, batch_first=True, padding_value=self.pad_token_id)
        batch_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (batch_input_ids != self.pad_token_id).long()
        query_attention_mask = (batch_query_ids != self.pad_token_id).long()

        return {
            "input_ids": batch_input_ids,
            "query_ids": batch_query_ids,
            "labels": batch_labels,
            "attention_mask": attention_mask,
            "query_attention_mask": query_attention_mask,
        }

class CustomTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        deduped = {}
        seen_data_ptrs = {}
        for k, v in state_dict.items():
            ptr = v.data_ptr()
            if ptr in seen_data_ptrs:
                deduped[k] = v.clone()
            else:
                seen_data_ptrs[ptr] = k
                deduped[k] = v
        safetensors.torch.save_file(deduped, os.path.join(output_dir, "model.safetensors"))
        if hasattr(self.model, "config"):
            self.model.config.save_pretrained(output_dir)

def train_model(model, train_dataset, eval_dataset, training_args, tokenizer):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(training_args)
    training_args.remove_unused_columns = False
    training_args.safe_serialization = False
    data_collator = DataCollatorForDynamicPadding(tokenizer.pad_token_id)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        print(f"Loaded from the checkpoint: {checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
