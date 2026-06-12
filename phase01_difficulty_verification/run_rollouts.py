"""run_rollouts.py — Step 1: Difficulty Label Generation

각 GSM8K 문제를 모델로 NUM_ROLLOUTS회 독립적으로 풀게 한 뒤,
정답률에 따라 Easy / Hard 라벨을 매겨 jsonl로 저장한다.
(논문 "The LLM Already Knows"의 difficulty labeling 방식)
"""

import os
import re
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 스크립트 파일 위치 기준 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== 실험 설정 =====
# Hub 조회/네트워크 없이 로컬 캐시 스냅샷에서 직접 로드 (Qwen2.5-14B 메타데이터 불완전 우회)
MODEL_NAME = "/mnt/hdd/hf_cache/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
NUM_SAMPLES = 500
NUM_ROLLOUTS = 3
DATASET_PATH = os.path.join(BASE_DIR, "data", "gsm8k_main_test.jsonl")
OUTPUT_PATH = "gsm8k_main_rollouts.jsonl"


def extract_answer(text: str):
    """텍스트에서 마지막 숫자를 정답으로 간주해 추출한다.

    콤마(1,000 → 1000)를 제거한 뒤 모든 정수/소수를 찾고,
    가장 마지막 값을 반환한다.

    기존 정규식 r"-?\d+\.?\d*"는 '18.'처럼 마침표까지 숫자로 잡을 수 있다.
    수정된 정규식은 소수점 뒤에 숫자가 있을 때만 소수로 인정한다.
    """
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def normalize_gold(answer: str):
    """GSM8K 정답(gold) 문자열에서 최종 정답 숫자만 뽑아낸다.

    GSM8K answer는 풀이 과정 뒤에 '#### 18' 형태로 정답이 붙어 있으므로,
    '####' 뒤쪽만 잘라서 숫자를 추출한다.
    """
    if "####" in answer:
        answer = answer.split("####")[-1]
    return extract_answer(answer)


def is_correct(pred: str, gold: str):
    """모델 출력(pred)이 정답(gold)과 일치하는지 판정한다.

    문자열 그대로 비교하지 않고 float으로 변환해 비교한다.
    예: '18', '18.0'은 같은 정답으로 처리한다.
    """
    pred_ans = extract_answer(pred)
    gold_ans = normalize_gold(gold)

    if pred_ans is None or gold_ans is None:
        return False

    try:
        return float(pred_ans) == float(gold_ans)
    except ValueError:
        return pred_ans == gold_ans


def build_prompt(question: str):
    """모델에 전달할 프롬프트를 구성한다."""
    return (
        "Solve the following math problem. "
        "Give the final answer after 'Final Answer:'.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def main():
    # 로컬 jsonl(data/gsm8k_main_test.jsonl)에서 앞쪽 NUM_SAMPLES개 문제만 로드
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
            gold = item["answer"]
            prompt = build_prompt(question)

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
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                generated = tokenizer.decode(
                    output_ids[0][input_ids.shape[-1]:],
                    skip_special_tokens=True,
                )

                correct = is_correct(generated, gold)
                correct_count += int(correct)

                rollouts.append({
                    "rollout_id": r,
                    "output": generated,
                    "parsed_pred": extract_answer(generated),
                    "parsed_gold": normalize_gold(gold),
                    "correct": correct,
                })

            label = "easy" if correct_count == NUM_ROLLOUTS else "hard"

            record = {
                "id": idx,
                "question": question,
                "gold": gold,
                "parsed_gold": normalize_gold(gold),
                "correct_count": correct_count,
                "label": label,
                "rollouts": rollouts,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()