import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
INPUT_PATH = "gsm8k_socratic_rollouts.jsonl"
OUTPUT_PATH = "gsm8k_socratic_hidden_states.npz"


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

    embeddings = []
    labels = []
    ids = []
    questions = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    for record in tqdm(records):
        question = record["question"]
        prompt = build_prompt(question)

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=False,             # True로 설정하면, 어시스턴트 응답 시작 토큰 추가됨. 내 연구에서는 질문의 마지막 토큰이 필요해서 False
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        
        print("> hidden state 사이즈 확인해보기(레이어수 추정) : ",len(outputs.hidden_states))
        last_hidden = outputs.hidden_states[-1]             # 마지막 레이어의 출력
        print("> last_hidden shape 확인해보기[batch_size, seq_len, hidden_dim] : ",last_hidden.shape)          

        # Last layer, last input token representation
        # last_hidden[0, -1, :] : 문제의 마지막 토큰의 dim차원 hidden representation(h_t를 가져옴)
        rep = last_hidden[0, -1, :].detach().float().cpu().numpy()

        embeddings.append(rep)
        labels.append(record["label"])
        ids.append(record["id"])
        questions.append(question)

    np.savez(
        OUTPUT_PATH,
        embeddings=np.stack(embeddings),
        labels=np.array(labels),
        ids=np.array(ids),
        questions=np.array(questions),
    )

    print("Saved:", OUTPUT_PATH)
    print("Embedding shape:", np.stack(embeddings).shape)


if __name__ == "__main__":
    main()