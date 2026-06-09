import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
INPUT_PATH = "gsm8k_rollouts.jsonl"
OUTPUT_PATH = "gsm8k_hidden_states.npz"


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
            add_generation_prompt=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )

        last_hidden = outputs.hidden_states[-1]

        # Last layer, last input token representation
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