"""
A-Agent: Answer Agent using trained Q-model for reasoning MCQs.

This module loads the trained Q-model and provides reasoning/answers
to given questions in the AAIPL format.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.q_m import QAgent


class AAgent:
    """Answer Agent: Provides reasoned answers to MCQ questions."""
    
    def __init__(self, lora_path: str = "./q_model_lora", use_llm_rewrite: bool = True):
        """Initialize A-Agent with trained Q-model."""
        self.model = QAgent(lora_path=lora_path, use_llm_rewrite=use_llm_rewrite)
        self.lora_path = lora_path
    
    def answer_question(self, question: Dict[str, object], temperature: float = 0.7) -> Dict[str, object]:
        """Generate answer and reasoning for a question.
        
        Input: {"topic": "Seating Arrangements", "question": "...", "choices": [...]}
        Output: {"answer": "C", "reasoning": "...", "confidence": 0.95, "latency_seconds": 0.234}
        """
        start_time = time.time()
        
        try:
            topic = question.get("topic", "")
            q_text = question.get("question", "")
            choices = question.get("choices", [])
            
            # Generate reasoning using the model
            prompt = f"Answer this reasoning MCQ:\n\nTopic: {topic}\nQuestion: {q_text}\nChoices:\n" + "\n".join(choices)
            response, _, _ = self.model.generate_response(
                prompt,
                temperature=temperature,
                max_new_tokens=500
            )
            
            # Extract answer letter from response
            answer_letter = self._extract_answer(response)
            confidence = self._estimate_confidence(response)
            
            latency = time.time() - start_time
            
            return {
                "answer": answer_letter,
                "reasoning": response[:200],
                "confidence": confidence,
                "latency_seconds": round(latency, 3)
            }
        except Exception as e:
            latency = time.time() - start_time
            return {
                "answer": "A",
                "reasoning": f"Error: {str(e)}",
                "confidence": 0.0,
                "latency_seconds": round(latency, 3)
            }
    
    def _extract_answer(self, response: str) -> str:
        """Extract answer letter (A, B, C, D) from response."""
        response_upper = response.upper()
        for letter in ["A", "B", "C", "D"]:
            if letter in response_upper:
                return letter
        return "A"
    
    def _estimate_confidence(self, response: str) -> float:
        """Estimate confidence score based on response quality."""
        length = len(response.split())
        if length < 10:
            return 0.3
        elif length < 30:
            return 0.6
        else:
            return 0.9


def answer_batch(questions_file: str, model_path: str = "./q_model_lora") -> None:
    """Answer a batch of questions from a JSON file."""
    print(f"\nA-Agent: Answering questions from {questions_file}\n")
    
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    agent = AAgent(lora_path=model_path)
    answers = []
    
    print(f"{'Question':<50} {'Answer':<10} {'Confidence':<12} {'Latency (s)'}")
    print("-" * 85)
    
    for q in questions:
        result = agent.answer_question(q)
        answers.append({
            "question": q.get('question', '')[:50],
            **result
        })
        print(f"{q.get('question', '')[:50]:<50} {result['answer']:<10} {result['confidence']:<12.2f} {result['latency_seconds']:<10.3f}")
    
    with open("answers.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Answers saved to answers.json\n")


def cmd_answer(args: object) -> None:
    """CLI command: Answer questions."""
    answer_batch(
        questions_file=getattr(args, "questions", "generated_questions.json"),
        model_path=getattr(args, "model", "./q_model_lora")
    )


def main() -> None:
    """Main CLI entry point for A-Agent."""
    parser = argparse.ArgumentParser(description="A-Agent: Answer agent for reasoning MCQs")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # answer command
    answer_parser = subparsers.add_parser("answer", help="Answer questions from a JSON file")
    answer_parser.add_argument("--questions", default="generated_questions.json", help="Input questions file")
    answer_parser.add_argument("--model", default="./q_model_lora", help="Trained model path")
    answer_parser.set_defaults(func=cmd_answer)
    
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
