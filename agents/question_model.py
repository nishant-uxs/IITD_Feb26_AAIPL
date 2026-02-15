import torch
import time
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/.cache/huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"


class QAgent:
    def __init__(self, **kwargs):

        print("🚀 Loading Question Model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            padding_side="left",
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        self.model.eval()
        print("✅ Question model loaded!")

    # ---------- Extract JSON ----------
    def extract_json(self, text):
        match = re.search(r'\{.*\}', text, re.DOTALL)

        if match:
            try:
                return json.loads(match.group())
            except:
                return None
        return None

    # ---------- FORCE HACKATHON FORMAT ----------
    def normalize_question(self, q):

        if not isinstance(q, dict):
            return None

        required = ["topic", "question", "choices", "answer", "explanation"]

        for k in required:
            if k not in q:
                return None

        # ---- FIX CHOICES ----
        choices = q["choices"]

        if isinstance(choices, list):
            if len(choices) != 4:
                return None

            choices = {
                "A": choices[0],
                "B": choices[1],
                "C": choices[2],
                "D": choices[3],
            }

        cleaned = {}
        for k, v in choices.items():
            key = k.strip().replace(")", "").replace(".", "").upper()

            if key not in ["A", "B", "C", "D"]:
                return None

            cleaned[key] = str(v).strip()

        if len(cleaned) != 4:
            return None

        q["choices"] = cleaned

        # ---- FIX ANSWER ----
        ans = str(q["answer"]).strip().upper()
        ans = ans.replace("OPTION ", "").replace(")", "")

        ans = ans[0]

        if ans not in ["A", "B", "C", "D"]:
            return None

        q["answer"] = ans

        # short explanation
        q["explanation"] = str(q["explanation"])[:300]

        return q

    # ---------- GENERATE ----------
    def generate_response(self, prompt, system_prompt, **kwargs):

        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
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
                max_new_tokens=1024,   # REQUIRED
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        latency = time.time() - start

        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

        parsed = self.extract_json(response)
        parsed = self.normalize_question(parsed)

        if parsed is None:
            return None, latency, None

        return parsed, latency, None
