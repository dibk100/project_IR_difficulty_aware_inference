import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

"""
GSM8K 문제
↓
모델에 입력
↓
Layer 1~N의 hidden state 추출
↓
각 레이어마다
  1. last-token representation
  2. mean-pooling representation
저장
↓
나중에 layerwise_probe.py에서 난이도 분류 실험
"""


# 경로를 스크립트 파일 기준으로 고정 → 어느 cwd에서 실행해도 안전
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../phase02_layerwise_decodability
PROJECT_ROOT = os.path.dirname(BASE_DIR)                       # .../project_IR_difficulty_aware_inference

# "meta-llama/Llama-3.1-8B-Instruct"
# "Qwen/Qwen2.5-7B-Instruct"
# "microsoft/Phi-3.5-mini-instruct"
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "phase01_difficulty_verification", "output_phi_1000", "gsm8k_main_rollouts.jsonl",
)
OUTPUT_PATH = "gsm8k_layerwise_hidden_states.npz"


def build_prompt(question: str):
    return (
        "Solve the following math problem. "
        "Give the final answer after 'Final Answer:'.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    layerwise_last = {}
    layerwise_mean = {}

    labels = []
    ids = []
    questions = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
        # 실험? records = records[:10]  -> Layer 01 | last: (10, 4096) | mean: (10, 4096)

    for idx, record in enumerate(tqdm(records)):
        question = record["question"]
        prompt = build_prompt(question)

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=False,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states

        if idx == 0:
            print("> hidden state 개수:", len(hidden_states))
            print("> hidden_states[0] shape:", hidden_states[0].shape)
            print("> hidden_states[-1] shape:", hidden_states[-1].shape)
            print("> decoded input example:")
            print(tokenizer.decode(input_ids[0]))

            num_layers = len(hidden_states) - 1
            print("> transformer layer 개수:", num_layers)

            for layer_idx in range(1, num_layers + 1):
                layerwise_last[layer_idx] = []
                layerwise_mean[layer_idx] = []

        num_layers = len(hidden_states) - 1

        for layer_idx in range(1, num_layers + 1):
            h = hidden_states[layer_idx][0]  # [seq_len, hidden_dim]

            rep_last = h[-1, :].detach().float().cpu().numpy()
            rep_mean = h.mean(dim=0).detach().float().cpu().numpy()

            layerwise_last[layer_idx].append(rep_last)
            layerwise_mean[layer_idx].append(rep_mean)

        labels.append(record["label"])
        ids.append(record["id"])
        questions.append(question)

    save_dict = {
        "labels": np.array(labels),
        "ids": np.array(ids),
        "questions": np.array(questions),
    }

    for layer_idx in layerwise_last.keys():
        save_dict[f"layer{layer_idx:02d}_last"] = np.stack(layerwise_last[layer_idx])
        save_dict[f"layer{layer_idx:02d}_mean"] = np.stack(layerwise_mean[layer_idx])

    np.savez(OUTPUT_PATH, **save_dict)

    print("Saved:", OUTPUT_PATH)
    print("Labels shape:", np.array(labels).shape)

    for layer_idx in layerwise_last.keys():
        print(
            f"Layer {layer_idx:02d} | "
            f"last: {save_dict[f'layer{layer_idx:02d}_last'].shape} | "
            f"mean: {save_dict[f'layer{layer_idx:02d}_mean'].shape}"
        )


if __name__ == "__main__":
    main()