"""run_rollouts.py — Step 1: Difficulty Label Generation

각 GSM8K 문제를 모델로 NUM_ROLLOUTS회 독립적으로 풀게 한 뒤,
정답률에 따라 Easy / Hard 라벨을 매겨 jsonl로 저장한다.
(논문 "The LLM Already Knows"의 difficulty labeling 방식)
"""

import re
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 실험 설정 =====
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # 난이도를 측정할 대상 모델
NUM_SAMPLES = 100        # 사용할 문제 수 (먼저 100, 잘 되면 300~500으로 확장)
NUM_ROLLOUTS = 3         # 문제당 독립 생성 횟수 (3/3 정답 → easy)
OUTPUT_PATH = "gsm8k_rollouts.jsonl"  # 라벨링 결과 저장 경로


def extract_answer(text: str):
    """텍스트에서 '마지막 숫자'를 정답으로 간주해 추출한다.

    콤마(1,000 → 1000)를 제거한 뒤 모든 정수/소수를 찾고,
    가장 마지막 값을 반환 (모델 출력은 보통 마지막에 답을 적기 때문).
    숫자가 없으면 None.
    """
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
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
    """모델 출력(pred)이 정답(gold)과 일치하는지 판정 (문자열 비교)."""
    pred_ans = extract_answer(pred)        # 모델 출력의 마지막 숫자
    gold_ans = normalize_gold(gold)        # gold의 '####' 뒤 정답 숫자
    return pred_ans == gold_ans


def build_prompt(question: str):
    """모델에 전달할 프롬프트를 구성한다 (최종 답을 명시하도록 지시)."""
    return (
        "Solve the following math problem. "
        "Give the final answer after 'Final Answer:'.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def main():
    # GSM8K test split에서 앞쪽 NUM_SAMPLES개 문제만 로드
    dataset = load_dataset("gsm8k", "main", split=f"test[:{NUM_SAMPLES}]")

    # 토크나이저 + 모델 로드 (bf16, 가용 GPU에 자동 분산)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()  # 추론 모드 (dropout 등 비활성화)

    # 결과를 jsonl로 한 줄씩 기록
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for idx, item in enumerate(tqdm(dataset)):
            question = item["question"]
            gold = item["answer"]          # '#### 정답'이 포함된 gold 문자열
            prompt = build_prompt(question)

            # Instruct 모델용 chat template 적용 → 토큰 id 텐서로 변환
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,   # 어시스턴트 응답 시작 토큰 추가
            ).to(model.device)

            rollouts = []
            correct_count = 0

            # 같은 문제를 NUM_ROLLOUTS회 독립적으로 생성 (sampling으로 매번 다른 풀이)
            for r in range(NUM_ROLLOUTS):
                with torch.no_grad():  # 그래디언트 미계산 → 메모리/속도 절약
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=256,
                        do_sample=True,        # 확률적 샘플링 (deterministic 아님)
                        temperature=0.7,       # 다양성 조절
                        top_p=0.95,            # nucleus sampling
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # 입력 길이 이후의 토큰만 디코딩 → 모델이 생성한 부분만 추출
                generated = tokenizer.decode(
                    output_ids[0][input_ids.shape[-1]:],
                    skip_special_tokens=True,
                )

                # 정답 여부 판정 후 누적
                correct = is_correct(generated, gold)
                correct_count += int(correct)

                rollouts.append({
                    "rollout_id": r,
                    "output": generated,
                    "correct": correct,
                })

            # 난이도 라벨: 3회 모두 정답이면 easy, 그 외는 hard
            label = "easy" if correct_count == NUM_ROLLOUTS else "hard"

            # 문제별 결과 레코드 구성 후 한 줄 기록
            record = {
                "id": idx,
                "question": question,
                "gold": gold,
                "correct_count": correct_count,
                "label": label,
                "rollouts": rollouts,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()