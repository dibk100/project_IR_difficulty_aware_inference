"""run_rollouts_mmlu.py — Step 1 (MMLU-Pro 버전): Difficulty Label Generation

MMLU-Pro 객관식 문제를 모델로 NUM_ROLLOUTS회 독립적으로 풀게 한 뒤,
정답률에 따라 Easy / Hard 라벨을 매겨 jsonl로 저장한다.

GSM8K용 run_rollouts.py 와 다른 점:
  - 객관식(A~J, 최대 10지선다)이므로 프롬프트에 options 를 포함한다.
  - 정답은 answer_index를 기준으로 gold letter를 재구성한다.
  - 채점은 모델 출력에서 선택지 글자를 파싱해 gold letter와 비교한다.
"""

import os
import re
import json
import string

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== 실험 설정 =====
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
NUM_SAMPLES = 500
NUM_ROLLOUTS = 3

DATASET_PATH = os.path.join(BASE_DIR, "data", "mmlu_pro_test.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "mmlu_pro_rollouts.jsonl")

LETTERS = string.ascii_uppercase


def build_prompt(question: str, options: list[str]) -> str:
    """MMLU-Pro 객관식 프롬프트 구성."""
    option_lines = "\n".join(
        f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options)
    )

    return (
        "Answer the following multiple choice question.\n"
        "Choose the single best option.\n"
        "Give only the letter of the correct option after 'Final Answer:'.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Final Answer:"
    )


def get_gold_letter(item: dict) -> str:
    """answer_index를 기준으로 gold letter를 반환한다."""
    if "answer_index" in item and item["answer_index"] is not None:
        gold_index = int(item["answer_index"])
        gold = LETTERS[gold_index]

        # answer 필드와 answer_index가 불일치하는 경우 경고
        if item.get("answer") is not None and item["answer"] != gold:
            print(
                f"[WARN] answer mismatch: "
                f"answer={item.get('answer')} "
                f"answer_index={gold_index} -> {gold}"
            )

        return gold

    # fallback
    return item["answer"]


def extract_pred(text: str, num_options: int) -> str | None:
    """모델 출력에서 예측 선택지 글자(A~J)를 파싱한다."""
    valid = set(LETTERS[:num_options])

    # 1) Final Answer: C / Final Answer: (C)
    m = re.search(r"Final Answer:\s*\(?([A-Z])\)?", text)
    if m and m.group(1) in valid:
        return m.group(1)

    # 2) Answer: C / Answer: (C)
    m = re.search(r"Answer:\s*\(?([A-Z])\)?", text)
    if m and m.group(1) in valid:
        return m.group(1)

    # 3) fallback: 단독으로 등장하는 유효 글자 중 마지막 것
    candidates = re.findall(r"\b([A-Z])\b", text)
    for c in reversed(candidates):
        if c in valid:
            return c

    return None


def is_correct(pred_letter: str | None, gold_letter: str) -> bool:
    return pred_letter is not None and pred_letter == gold_letter


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f][:NUM_SAMPLES]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for idx, item in enumerate(tqdm(dataset)):
            question = item["question"]
            options = item["options"]
            num_options = len(options)

            gold = get_gold_letter(item)
            prompt = build_prompt(question, options)

            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(model.device)

            rollouts = []
            correct_count = 0

            for r in range(NUM_ROLLOUTS):
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=16,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                generated = tokenizer.decode(
                    output_ids[0][input_ids.shape[-1]:],
                    skip_special_tokens=True,
                )

                pred_letter = extract_pred(generated, num_options)
                correct = is_correct(pred_letter, gold)
                correct_count += int(correct)

                rollouts.append(
                    {
                        "rollout_id": r,
                        "output": generated,
                        "parsed_pred": pred_letter,
                        "parsed_gold": gold,
                        "correct": correct,
                    }
                )

            label = "easy" if correct_count == NUM_ROLLOUTS else "hard"

            record = {
                "id": idx,
                "question_id": item.get("question_id"),
                "category": item.get("category"),
                "src": item.get("src"),
                "question": question,
                "options": options,
                "gold": gold,
                "gold_index": LETTERS.index(gold),
                "original_answer": item.get("answer"),
                "original_answer_index": item.get("answer_index"),
                "correct_count": correct_count,
                "num_rollouts": NUM_ROLLOUTS,
                "label": label,
                "rollouts": rollouts,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()