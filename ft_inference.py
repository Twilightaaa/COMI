import json
import math
import os
import random
from pathlib import Path
import torch
import torch.distributed as dist
from tqdm import tqdm
from peft import LoraConfig
from modeling_comi import COMI, build_lora_config, load_checkpoint_file
from eval_utils import compute_exact, compute_f1, normalize_answer


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return distributed, rank, world_size, local_rank, device


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def chunked(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def load_eval_samples(test_file, num_samples, shuffle_samples, seed):
    samples = []
    with open(test_file, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            missing = {"input", "prompt", "answer"} - sample.keys()
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"Missing fields at line {line_number}: {names}")
            samples.append(sample)
    if shuffle_samples:
        random.Random(seed).shuffle(samples)
    return samples if num_samples <= 0 else samples[:num_samples]


def make_output_paths(output_dir, num_samples, merge_size):
    eval_dir = Path(output_dir) / "eval_result"
    eval_dir.mkdir(parents=True, exist_ok=True)
    sample_tag = "all" if num_samples == 0 else str(num_samples)
    output_path = eval_dir / f"nq_inference_results_{sample_tag}_{merge_size}.jsonl"
    metrics_path = eval_dir / f"nq_inference_metrics_{sample_tag}_{merge_size}.json"
    tmp_dir = eval_dir / f".tmp_rank_outputs_{sample_tag}_{merge_size}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return output_path, metrics_path, tmp_dir


def prepare_model(model_args, training_args, device):
    model_args.train = False
    lora_config = LoraConfig(**build_lora_config(model_args))
    model = COMI(model_args, training_args, lora_config=lora_config)
    if training_args.restore_from:
        state_dict, ckpt_file = load_checkpoint_file(training_args.restore_from)
        print(f"Loading checkpoint from {ckpt_file}")
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def infer_rank_batches(model, samples, batch_size, training_args, device, rank, world_size):
    rank_samples = samples[rank::world_size]
    generated_rows = []
    iterator = chunked(rank_samples, batch_size)
    progress = tqdm(iterator, total=math.ceil(len(rank_samples) / batch_size), disable=rank != 0, desc="Inference")

    with torch.inference_mode():
        for batch in progress:
            contexts = [item.get("input", "") for item in batch]
            questions = [item.get("prompt", "") for item in batch]

            context_inputs = model.tokenizer(
                contexts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            query_inputs = model.tokenizer(
                questions,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )

            generated_ids = model(
                input_ids=context_inputs["input_ids"].to(device),
                attention_mask=context_inputs["attention_mask"].to(device),
                query_ids=query_inputs["input_ids"].to(device),
                query_attention_mask=query_inputs["attention_mask"].to(device),
                labels=None,
            )
            responses = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for sample, response in zip(batch, responses):
                generated_rows.append(
                    {
                        "context": sample.get("input", ""),
                        "question": sample.get("prompt", ""),
                        "model_output": response,
                        "ground_truth": sample.get("answer", ""),
                        "sample_id": sample.get("sample_id"),
                    }
                )

    return generated_rows


def write_rank_results(rank_rows, tmp_dir, rank):
    rank_file = tmp_dir / f"rank_{rank:05d}.jsonl"
    with open(rank_file, "w", encoding="utf-8") as f:
        for row in rank_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rank_file


def merge_rank_outputs(tmp_dir, world_size, final_output_path):
    merged_rows = []
    for rank in range(world_size):
        rank_file = tmp_dir / f"rank_{rank:05d}.jsonl"
        if not rank_file.exists():
            continue
        with open(rank_file, "r", encoding="utf-8") as f:
            for line in f:
                merged_rows.append(json.loads(line))

    merged_rows.sort(key=lambda row: row.get("sample_id", 0))
    with open(final_output_path, "w", encoding="utf-8") as out_f:
        for row in merged_rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return merged_rows


def compute_metrics(rows):
    em_scores = []
    f1_scores = []

    for item in rows:
        prediction = item["model_output"]
        ground_truth = item["ground_truth"]

        if isinstance(ground_truth, list):
            em_score = max(compute_exact(normalize_answer(gt), normalize_answer(prediction)) for gt in ground_truth)
            f1_score = max(compute_f1(normalize_answer(prediction), normalize_answer(gt)) for gt in ground_truth)
        else:
            em_score = compute_exact(normalize_answer(ground_truth), normalize_answer(prediction))
            f1_score = compute_f1(normalize_answer(prediction), normalize_answer(ground_truth))

        em_scores.append(em_score)
        f1_scores.append(f1_score)

    return {
        "total_samples": len(em_scores),
        "avg_em": sum(em_scores) / len(em_scores) if em_scores else 0,
        "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
    }


def main():
    from transformers import HfArgumentParser

    from modeling_comi import DataArguments, ModelArguments, TrainingArguments

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    distributed, rank, world_size, local_rank, device = setup_distributed()
    random.seed(training_args.seed + rank)
    torch.manual_seed(training_args.seed + rank)

    effective_merge_size = model_args.compress_ratio if model_args.compress_ratio > 0 else model_args.merge_size
    output_path, metrics_path, tmp_dir = make_output_paths(training_args.output_dir, data_args.num_samples, effective_merge_size)

    if rank == 0:
        print(f"Initializing model from {model_args.model_name_or_path}")
    model = prepare_model(model_args, training_args, device)

    samples = load_eval_samples(
        data_args.test_file,
        num_samples=data_args.num_samples,
        shuffle_samples=data_args.shuffle_samples,
        seed=training_args.seed,
    )
    for idx, sample in enumerate(samples):
        sample["sample_id"] = idx

    if rank == 0:
        print(
            f"Running inference on {len(samples)} samples with world_size={world_size}, "
            f"per_device_eval_batch_size={training_args.per_device_eval_batch_size}"
        )

    rank_rows = infer_rank_batches(
        model=model,
        samples=samples,
        batch_size=training_args.per_device_eval_batch_size,
        training_args=training_args,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    write_rank_results(rank_rows, tmp_dir, rank)

    barrier()

    if rank == 0:
        merged_rows = merge_rank_outputs(tmp_dir, world_size, output_path)
        final_metrics = compute_metrics(merged_rows)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=4)

        print(f"Inference finished. Results saved to: {output_path}")
        print(f"Metrics saved to: {metrics_path}")
        print(
            f"Final metrics: EM={final_metrics['avg_em']:.4f}, "
            f"F1={final_metrics['avg_f1']:.4f}"
        )

    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
