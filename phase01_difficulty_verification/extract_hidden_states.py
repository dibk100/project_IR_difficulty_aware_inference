import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
INPUT_PATH = "gsm8k_main_rollouts.jsonl"
OUTPUT_PATH = "gsm8k_main_hidden_states.npz"


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

    embeddings_last = []
    embeddings_mean = []
    labels = []
    ids = []
    questions = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

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