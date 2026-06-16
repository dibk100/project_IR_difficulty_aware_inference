import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


"""
Phase03-A: Sample-wise Token Hidden Matrix Extraction

GSM8K problem
↓
Same prompt as Phase02
↓
Model forward pass
↓
Extract token-level hidden matrix for every layer
↓
Save one .npz file per sample

Each sample file contains:
- id
- question
- label
- correct_count
- token_len
- layer01_hidden: (T, D)
- layer02_hidden: (T, D)
...


실행 예시
python a_effective_rank/extract_token_hidden_matrices.py \
  --model_name microsoft/Phi-3.5-mini-instruct \
  --input_path ../phase01_difficulty_verification/output_phi_1000/gsm8k_main_rollouts.jsonl \
  --output_dir output_phi_1000/effective_rank/token_hidden_matrices
  
python a_effective_rank/extract_token_hidden_matrices.py \
  --model_name microsoft/Phi-3.5-mini-instruct \
  --input_path ../phase01_difficulty_verification/output_phi_500/gsm8k_main_rollouts.jsonl \
  --output_dir output_phi_500/effective_rank/token_hidden_matrices_debug \
  --max_samples 3 \
  --overwrite
"""


def build_prompt(question: str) -> str:
    return (
        "Solve the following math problem. "
        "Give the final answer after 'Final Answer:'.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or local model path.",
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to phase01 gsm8k_main_rollouts.jsonl.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save sample-wise token hidden matrices.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional maximum number of samples for debugging.",
    )

    parser.add_argument(
        "--save_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Dtype used when saving hidden matrices.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sample files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("[Phase03-A] Extract token hidden matrices")
    print("=" * 80)
    print("Model:", args.model_name)
    print("Input:", args.input_path)
    print("Output dir:", args.output_dir)
    print("Save dtype:", args.save_dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    with open(args.input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    if args.max_samples is not None:
        records = records[: args.max_samples]

    print("Number of records:", len(records))

    manifest = []

    for idx, record in enumerate(tqdm(records)):
        sample_id = record["id"]
        question = record["question"]
        label = record["label"]
        correct_count = record.get("correct_count", None)

        output_path = os.path.join(
            args.output_dir,
            f"sample_{idx:06d}.npz",
        )

        if os.path.exists(output_path) and not args.overwrite:
            manifest.append(
                {
                    "index": idx,
                    "id": sample_id,
                    "label": label,
                    "path": output_path,
                    "status": "skipped_existing",
                }
            )
            continue

        prompt = build_prompt(question)

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=False,
        ).to(model.device)

        token_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        num_layers = len(hidden_states) - 1

        if idx == 0:
            print("> hidden state count:", len(hidden_states))
            print("> hidden_states[0] shape:", hidden_states[0].shape)
            print("> hidden_states[-1] shape:", hidden_states[-1].shape)
            print("> transformer layer count:", num_layers)
            print("> token_len:", token_len)
            print("> decoded input example:")
            print(tokenizer.decode(input_ids[0]))

        save_dict = {
            "index": np.array(idx),
            "id": np.array(sample_id),
            "question": np.array(question),
            "label": np.array(label),
            "correct_count": np.array(-1 if correct_count is None else correct_count),
            "token_len": np.array(token_len),
        }

        for layer_idx in range(1, num_layers + 1):
            # hidden_states[layer_idx]: [batch=1, seq_len=T, hidden_dim=D]
            h = hidden_states[layer_idx][0].detach().cpu()

            if args.save_dtype == "float16":
                h_np = h.to(torch.float16).numpy()
            else:
                h_np = h.float().numpy()

            save_dict[f"layer{layer_idx:02d}_hidden"] = h_np

        np.savez_compressed(output_path, **save_dict)

        manifest.append(
            {
                "index": idx,
                "id": sample_id,
                "label": label,
                "token_len": int(token_len),
                "path": output_path,
                "status": "saved",
            }
        )

    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("=" * 80)
    print("Done.")
    print("Saved manifest:", manifest_path)
    print("Saved files:", args.output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()