#!/usr/bin/python3

import torch
import json
import re
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "/root/.cache/huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"


SYSTEM_PROMPT = """
You are an elite logical reasoning solver.

STRICT RULES:

- Output ONLY valid JSON.
- No thinking.
- No markdown.
- No explanation outside JSON.

FORMAT:

{
 "answer": "A",
 "reasoning": "brief reasoning under 100 words"
}

IMPORTANT:

- Answer MUST be one of A, B, C, or D.
- Solve carefully before answering.
"""


class AAgent:

    def __init__(self):

        print("🚀 Loading Answer Model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            padding_side="left",
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,   # ⭐ MI300X optimized
            device_map="auto",
            trust_remote_code=True
        )

        self.model.eval()

        print("✅ Answer model loaded!")

    # ⭐ STRONG JSON extractor
    def extract_json(self, text):

        match = re.search(r'\{.*\}', text, re.DOTALL)

        if not match:
            return None

        try:
            return json.loads(match.group())
        except:
            return None


    def solve(self, question, choices):

        user_prompt = f"""
Question:
{question}

Choices:
A) {choices['A']}
B) {choices['B']}
C) {choices['C']}
D) {choices['D']}

Return ONLY JSON.
"""

        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.model.device)

        start = time.time()

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,        # ⭐ README REQUIRED
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,            # ⭐ MUST when temperature used
                pad_token_id=self.tokenizer.eos_token_id
            )

        latency = round(time.time() - start, 2)

        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

        parsed = self.extract_json(response)

        # ⭐ fallback if model acts dumb
        if parsed is None:
            return {
                "answer": "A",
                "reasoning": "Model output parsing failed."
            }, latency

        # safety check
        if parsed.get("answer") not in ["A", "B", "C", "D"]:
            parsed["answer"] = "A"

        return parsed, latency


def main():

    agent = AAgent()

    with open("outputs/filtered_questions.json") as f:
        questions = json.load(f)

    answers = []

    for q in questions:

        result, latency = agent.solve(
            q["question"],
            q["choices"]
        )

        result["latency_sec"] = latency

        answers.append(result)

    with open("outputs/answers.json", "w") as f:
        json.dump(answers, f, indent=4)

    print("✅ Answers saved!")


if __name__ == "__main__":
    main()
