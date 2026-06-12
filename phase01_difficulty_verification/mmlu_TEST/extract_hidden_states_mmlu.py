"""extract_hidden_states_mmlu.py — Step 2 (MMLU-Pro 버전): Hidden State Extraction

run_rollouts_mmlu.py 가 만든 라벨(mmlu_pro_rollouts.jsonl)을 입력으로,
각 질문의 입력 표현(hidden representation)을 추출해 npz로 저장한다.

GSM8K용 extract_hidden_states.py 와 다른 점:
  - 객관식 프롬프트(options 포함)를 사용한다. run_rollouts_mmlu.py 의 build_prompt 와
    동일한 형식이어야 모델이 본 입력과 일치한다.
  - 입력/출력 파일명이 mmlu_pro_* 이다.
"""

import string
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
INPUT_PATH = "mmlu_pro_rollouts.jsonl"
OUTPUT_PATH = "mmlu_pro_hidden_states.npz"

LETTERS = string.ascii_uppercase


def build_prompt(question: str, options: list):
    """run_rollouts_mmlu.py 와 동일한 객관식 프롬프트."""
    option_lines = "\n".join(
        f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options)
    )
    return (
        "Answer the following multiple choice question. "
        "Give only the letter of the correct option after 'Final Answer:'.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_lines}\n\n"
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

    embeddings_last = []
    embeddings_mean = []
    labels = []
    ids = []
    questions = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    for idx, record in enumerate(tqdm(records)):
        question = record["question"]
        options = record["options"]
        prompt = build_prompt(question, options)

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

        last_hidden = outputs.hidden_states[-1]

        if idx == 0:
            print("> hidden state 개수:", len(outputs.hidden_states))
            print("> last_hidden shape [batch_size, seq_len, hidden_dim]:", last_hidden.shape)
            print("> decoded input example:")
            print(tokenizer.decode(input_ids[0]))

        # 1. Last layer, last token representation
        rep_last = last_hidden[0, -1, :].detach().float().cpu().numpy()

        # 2. Last layer, mean pooling over all input tokens
        rep_mean = last_hidden[0].mean(dim=0).detach().float().cpu().numpy()

        embeddings_last.append(rep_last)
        embeddings_mean.append(rep_mean)
        labels.append(record["label"])
        ids.append(record["id"])
        questions.append(question)

    np.savez(
        OUTPUT_PATH,
        embeddings_last=np.stack(embeddings_last),
        embeddings_mean=np.stack(embeddings_mean),
        labels=np.array(labels),
        ids=np.array(ids),
        questions=np.array(questions),
    )

    print("Saved:", OUTPUT_PATH)
    print("Last-token embedding shape:", np.stack(embeddings_last).shape)
    print("Mean-pooling embedding shape:", np.stack(embeddings_mean).shape)


if __name__ == "__main__":
    main()
