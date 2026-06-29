import transformers
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer

from modeling_comi import COMI, DataArguments, ModelArguments, TrainingArguments, build_lora_config
from training_utils import InstructFTTokenizeFunction, train_model


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files={"train": data_args.train_file, "eval": data_args.test_file},
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    if data_args.debug_data:
        train_dataset = train_dataset.select(range(min(32, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(10, len(eval_dataset))))

    print(f"Dataset size: train={len(train_dataset)}, eval={len(eval_dataset)}")

    tokenize = InstructFTTokenizeFunction(tokenizer)
    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        batch_size=1000,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        tokenize,
        batched=True,
        batch_size=1000,
        remove_columns=eval_dataset.column_names,
    )

    model = COMI(
        model_args,
        training_args,
        lora_config=LoraConfig(**build_lora_config(model_args)),
    )
    train_model(model, train_dataset, eval_dataset, training_args, tokenizer)


if __name__ == "__main__":
    main()
