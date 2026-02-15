import torch
import time
import re
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "/root/.cache/huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"


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
            dtype=torch.float16,      # torch_dtype deprecated fix
            device_map="auto",
            trust_remote_code=True
        )

        self.model.eval()

        print("✅ Answer model loaded!")

    def extract_option(self, text):

        match = re.search(r"\b([ABCD])\b", text)

        if match:
            return match.group(1)

        return "A" 
    def generate_response(self, question):

        prompt = f"""
Solve the logical reasoning MCQ.

Return ONLY the correct option letter.

NO explanation.
NO thinking.
NO extra words.

Question:
{question}

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        start = time.time()

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=True,
                temperature=0.45,  
                top_p=0.9,

                pad_token_id=self.tokenizer.eos_token_id
            )

        latency = time.time() - start

        decoded = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

        letter = self.extract_option(decoded)

        return letter, latency
    def solve(self, question):

        ans1, _ = self.generate_response(question)
        ans2, latency = self.generate_response(question)
        if ans1 == ans2:
            return ans1
        ans3, _ = self.generate_response(question)

        return ans3
