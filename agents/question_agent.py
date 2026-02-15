#!/usr/bin/python3

from tqdm import tqdm
from pathlib import Path
from .question_model import QAgent

import random
import json


SYSTEM_PROMPT = """
You are an elite aptitude question setter.

STRICT RULES:

- Output ONLY valid JSON.
- Do NOT think.
- Do NOT write <think>.
- No markdown.
- No extra text.

FORMAT:

{
 "topic": "...",
 "question": "...",
 "choices": {
   "A": "...",
   "B": "...",
   "C": "...",
   "D": "..."
 },
 "answer": "A",
 "explanation": "under 100 words"
}

Make the question HARD but solvable.
"""


class QuestioningAgent:
    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)

    def build_prompt(self, topic):

        prompt = f"""
Generate ONE extremely challenging logical reasoning MCQ.

Topic: {topic}

Rules:

- Avoid easy patterns
- Require multi-step reasoning
- No tricks
- No ambiguity
- Must be logically solvable

Return ONLY JSON.
"""

        return prompt, SYSTEM_PROMPT

    # ---------- MAIN GENERATOR ----------
    def generate_batches(self, num_questions, topics):

        all_topics = [(t, st) for t in topics for st in topics[t]]

        raw_questions = []
        filtered_questions = []

        pbar = tqdm(total=num_questions, desc="Generating Questions")

        while len(filtered_questions) < num_questions:

            t, st = random.choice(all_topics)
            topic = f"{t}/{st}"

            prompt, sp = self.build_prompt(topic)

            resp, _, _ = self.agent.generate_response(prompt, sp)

            if resp:
                raw_questions.append(resp)

                # Basic quality filter
                if len(resp["question"]) > 25:
                    filtered_questions.append(resp)
                    pbar.update(1)

        pbar.close()

        return raw_questions, filtered_questions

    # ---------- SAVE BOTH ----------
    def save_questions(self, raw, filtered):

        Path("outputs").mkdir(exist_ok=True)

        with open("outputs/questions.json", "w") as f:
            json.dump(raw, f, indent=4)

        with open("outputs/filtered_questions.json", "w") as f:
            json.dump(filtered, f, indent=4)


# ================= RUN =================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=5)

    args = parser.parse_args()

    with open("assets/topics.json") as f:
        topics = json.load(f)

    agent = QuestioningAgent()

    raw, filtered = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics
    )

    agent.save_questions(raw, filtered)

    print(f"\n✅ Generated EXACTLY {len(filtered)} filtered questions!")
