import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set HuggingFace cache to writable directory BEFORE any transformers imports
_cache_root = Path(__file__).resolve().parents[1] / ".hf_cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_cache_root))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_cache_root / "transformers"))

import torch
from datasets import Dataset

DOMAIN_ALIASES = {
    "seating": "Seating Arrangements",
    "seating arrangements": "Seating Arrangements",
    "blood": "Blood Relations",
    "blood relations": "Blood Relations",
    "syllogism": "Syllogisms",
    "syllogisms": "Syllogisms",
    "series": "Mixed Series (Alphanumeric)",
    "mixed series": "Mixed Series (Alphanumeric)",
    "alphanumeric": "Mixed Series (Alphanumeric)",
}

DOMAINS = [
    "Seating Arrangements",
    "Blood Relations",
    "Syllogisms",
    "Mixed Series (Alphanumeric)",
]


@dataclass
class GeneratedItem:
    domain: str
    question: str
    choices: List[str]
    expected_answer: str
    explanation: str


class StructureGenerator:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def generate(self, domain: str) -> GeneratedItem:
        domain = self._normalize_domain(domain)
        if domain == "Seating Arrangements":
            return self._generate_seating()
        if domain == "Blood Relations":
            return self._generate_blood()
        if domain == "Syllogisms":
            return self._generate_syllogism()
        if domain == "Mixed Series (Alphanumeric)":
            return self._generate_series()
        raise ValueError(f"Unsupported domain: {domain}")

    def validate(self, item: GeneratedItem) -> Tuple[bool, str]:
        if item.domain not in DOMAINS:
            return False, "invalid_domain"
        if not item.question or not item.question.strip().endswith("?"):
            return False, "question_format"
        if len(item.choices) != 4:
            return False, "choices_count"
        if item.expected_answer not in ["A", "B", "C", "D"]:
            return False, "expected_answer"
        normalized = [c.strip().lower() for c in item.choices]
        if len(set(normalized)) != 4:
            return False, "duplicate_choices"
        return True, "ok"

    def to_json(self, item: GeneratedItem) -> str:
        """Convert to AAIPL competition format."""
        obj = {
            "topic": item.domain,
            "question": item.question,
            "choices": item.choices,
            "answer": item.expected_answer,
            "explanation": item.explanation,
        }
        return json.dumps(obj, ensure_ascii=False, indent=2)

    def _normalize_domain(self, domain: str) -> str:
        key = domain.strip().lower()
        return DOMAIN_ALIASES.get(key, domain)

    def _generate_seating(self) -> GeneratedItem:
        names = self._pick_names(6)
        positions = list(range(6))
        self.rng.shuffle(names)
        seating = dict(zip(positions, names))

        def left_of(pos: int) -> int:
            return (pos - 1) % 6

        def right_of(pos: int) -> int:
            return (pos + 1) % 6

        name_to_pos = {name: pos for pos, name in seating.items()}
        anchor = names[0]
        anchor_pos = name_to_pos[anchor]

        constraints = []
        constraints.append(("anchor", anchor, anchor_pos))

        for _ in range(4):
            a, b = self.rng.sample(names, 2)
            if self.rng.random() < 0.5:
                constraints.append(("left", a, b))
            else:
                constraints.append(("right", a, b))

        a, b, c = self.rng.sample(names, 3)
        constraints.append(("between", a, b, c))

        if not self._seating_unique(names, constraints):
            constraints.append(("opposite", names[1], names[4]))

        if not self._seating_unique(names, constraints):
            constraints.append(("second_left", names[2], names[5]))

        if not self._seating_unique(names, constraints):
            constraints.append(("second_right", names[3], names[0]))

        if not self._seating_unique(names, constraints):
            constraints = self._make_constraints_from_full_arrangement(seating, anchor)

        question, choices, answer = self._build_seating_question(seating, constraints)
        explanation = self._explain_seating(seating, constraints, question)
        return GeneratedItem("Seating Arrangements", question, choices, answer, explanation)

    def _seating_unique(self, names: List[str], constraints: List[Tuple]) -> bool:
        fixed_anchor = None
        for c in constraints:
            if c[0] == "anchor":
                fixed_anchor = (c[1], c[2])
                break
        if fixed_anchor is None:
            return False

        anchor_name, anchor_pos = fixed_anchor
        remaining = [n for n in names if n != anchor_name]
        positions = [p for p in range(6) if p != anchor_pos]

        count = 0
        for perm in self._permute(remaining):
            seating = {anchor_pos: anchor_name}
            for pos, name in zip(positions, perm):
                seating[pos] = name
            if self._check_seating_constraints(seating, constraints):
                count += 1
                if count > 1:
                    return False
        return count == 1

    def _check_seating_constraints(self, seating: Dict[int, str], constraints: List[Tuple]) -> bool:
        name_to_pos = {v: k for k, v in seating.items()}

        def left_of(pos: int) -> int:
            return (pos - 1) % 6

        def right_of(pos: int) -> int:
            return (pos + 1) % 6

        for c in constraints:
            if c[0] == "anchor":
                if name_to_pos.get(c[1]) != c[2]:
                    return False
            elif c[0] == "left":
                if name_to_pos[c[1]] != left_of(name_to_pos[c[2]]):
                    return False
            elif c[0] == "right":
                if name_to_pos[c[1]] != right_of(name_to_pos[c[2]]):
                    return False
            elif c[0] == "between":
                a, b, d = c[1], c[2], c[3]
                pos_a = name_to_pos[a]
                if not ((left_of(pos_a) == name_to_pos[b] and right_of(pos_a) == name_to_pos[d]) or
                        (left_of(pos_a) == name_to_pos[d] and right_of(pos_a) == name_to_pos[b])):
                    return False
            elif c[0] == "opposite":
                if (name_to_pos[c[1]] - name_to_pos[c[2]]) % 6 != 3:
                    return False
            elif c[0] == "second_left":
                if name_to_pos[c[1]] != (name_to_pos[c[2]] - 2) % 6:
                    return False
            elif c[0] == "second_right":
                if name_to_pos[c[1]] != (name_to_pos[c[2]] + 2) % 6:
                    return False
        return True

    def _make_constraints_from_full_arrangement(self, seating: Dict[int, str], anchor: str) -> List[Tuple]:
        name_to_pos = {v: k for k, v in seating.items()}
        constraints = [("anchor", anchor, name_to_pos[anchor])]
        for i, name in seating.items():
            left_name = seating[(i - 1) % 6]
            right_name = seating[(i + 1) % 6]
            constraints.append(("left", left_name, name))
            constraints.append(("right", right_name, name))
        return constraints

    def _build_seating_question(
        self,
        seating: Dict[int, str],
        constraints: List[Tuple],
    ) -> Tuple[str, List[str], str]:
        name_to_pos = {v: k for k, v in seating.items()}

        anchor = None
        anchor_pos = None
        for c in constraints:
            if c[0] == "anchor":
                anchor = c[1]
                anchor_pos = c[2]
                break

        statements = []
        statements.append(
            f"Six friends sit around a circular table facing the center. {anchor} sits at position {anchor_pos + 1} counting clockwise from the top."
        )

        for c in constraints:
            if c[0] == "left":
                statements.append(f"{c[1]} sits immediately to the left of {c[2]}.")
            elif c[0] == "right":
                statements.append(f"{c[1]} sits immediately to the right of {c[2]}.")
            elif c[0] == "between":
                statements.append(f"{c[1]} sits between {c[2]} and {c[3]}.")
            elif c[0] == "opposite":
                statements.append(f"{c[1]} sits opposite {c[2]}.")
            elif c[0] == "second_left":
                statements.append(f"{c[1]} sits second to the left of {c[2]}.")
            elif c[0] == "second_right":
                statements.append(f"{c[1]} sits second to the right of {c[2]}.")

        target = self.rng.choice(list(seating.values()))
        pos = name_to_pos[target]
        correct_name = seating[(pos - 1) % 6]
        question = " ".join(statements) + f" Who sits immediately to the left of {target}?"

        choices = self._make_choices(correct_name, list(seating.values()))
        answer = self._correct_letter(choices, correct_name)
        return question, choices, answer

    def _explain_seating(self, seating: Dict[int, str], constraints: List[Tuple], question: str) -> str:
        name_to_pos = {v: k for k, v in seating.items()}
        arrangement = [seating[i] for i in range(6)]
        target = question.split("Who sits immediately to the left of ")[-1].rstrip("?")
        if target in name_to_pos:
            pos = name_to_pos[target]
            correct = seating[(pos - 1) % 6]
            return f"The circular arrangement is {arrangement}. {correct} sits to the left of {target}."
        return "Arrangement follows all given constraints."

    def _generate_blood(self) -> GeneratedItem:
        people = self._pick_names(8)
        father, mother = people[0], people[1]
        son, daughter = people[2], people[3]
        uncle, aunt = people[4], people[5]
        cousin = people[6]
        grandparent = people[7]

        statements = [
            f"{grandparent} is the parent of {father}.",
            f"{grandparent} is the parent of {uncle}.",
            f"{father} is the father of {son}.",
            f"{mother} is the mother of {son}.",
            f"{father} is the father of {daughter}.",
            f"{mother} is the mother of {daughter}.",
            f"{uncle} is the father of {cousin}.",
        ]
        self.rng.shuffle(statements)

        target_pair = self.rng.choice([
            (son, cousin),
            (cousin, son),
            (daughter, cousin),
            (cousin, daughter),
            (son, grandparent),
            (daughter, grandparent),
            (cousin, grandparent),
        ])

        relation = self._compute_relation(
            subject=target_pair[0],
            obj=target_pair[1],
            father=father,
            mother=mother,
            son=son,
            daughter=daughter,
            uncle=uncle,
            cousin=cousin,
            grandparent=grandparent,
        )

        question = " ".join(statements) + f" What is {target_pair[0]}'s relation to {target_pair[1]}?"
        choices = self._make_relation_choices(relation)
        answer = self._correct_letter(choices, relation)
        explanation = f"Based on the family tree, {target_pair[0]} is the {relation} of {target_pair[1]}."
        return GeneratedItem("Blood Relations", question, choices, answer, explanation)

    def _compute_relation(
        self,
        subject: str,
        obj: str,
        father: str,
        mother: str,
        son: str,
        daughter: str,
        uncle: str,
        cousin: str,
        grandparent: str,
    ) -> str:
        if subject == son and obj == grandparent:
            return "grandson"
        if subject == daughter and obj == grandparent:
            return "granddaughter"
        if subject == cousin and obj == grandparent:
            return "grandson"
        if subject == son and obj == cousin:
            return "cousin"
        if subject == cousin and obj == son:
            return "cousin"
        if subject == daughter and obj == cousin:
            return "cousin"
        if subject == cousin and obj == daughter:
            return "cousin"
        return "cousin"

    def _make_relation_choices(self, correct: str) -> List[str]:
        pool = ["cousin", "brother", "sister", "uncle", "aunt", "grandson", "granddaughter", "nephew", "niece"]
        if correct not in pool:
            pool.append(correct)
        self.rng.shuffle(pool)
        distractors = [p for p in pool if p != correct][:3]
        options = [correct] + distractors
        self.rng.shuffle(options)
        return [f"{letter}) {opt}" for letter, opt in zip(["A", "B", "C", "D"], options)]

    def _generate_syllogism(self) -> GeneratedItem:
        terms = self.rng.sample(["artists", "painters", "poets", "designers", "writers", "dancers"], 3)
        a, b, c = terms

        premises = [
            ("all", a, b),
            ("some", b, c),
        ]

        conclusions = [
            ("some", a, c),
            ("all", c, a),
            ("no", a, c),
            ("some_not", a, c),
            ("all", b, a),
            ("some", c, a),
            ("no", b, c),
            ("some_not", b, c),
        ]

        entailed = []
        non_entailed = []
        for concl in conclusions:
            if self._entails([a, b, c], premises, concl):
                entailed.append(concl)
            else:
                non_entailed.append(concl)

        if not entailed:
            entailed = [conclusions[0]]
            non_entailed = [c for c in conclusions if c != entailed[0]]

        correct = entailed[0]
        self.rng.shuffle(non_entailed)
        distractors = non_entailed[:3]
        if len(distractors) < 3:
            extras = [c for c in conclusions if c != correct and c not in distractors]
            self.rng.shuffle(extras)
            distractors = (distractors + extras)[:3]

        question = (
            f"Statements: 1) All {a} are {b}. 2) Some {b} are {c}. "
            "Which conclusion logically follows?"
        )

        options = [self._render_statement(correct)] + [self._render_statement(d) for d in distractors]
        self.rng.shuffle(options)
        choices = [f"{letter}) {opt}" for letter, opt in zip(["A", "B", "C", "D"], options)]
        answer = self._correct_letter(choices, self._render_statement(correct))
        explanation = f"The conclusion '{self._render_statement(correct)}' logically follows from the given premises using set theory."
        return GeneratedItem("Syllogisms", question, choices, answer, explanation)

    def _entails(self, terms: List[str], premises: List[Tuple[str, str, str]], concl: Tuple[str, str, str]) -> bool:
        universe = [0, 1, 2, 3]
        all_sets = self._all_sets(universe)

        for set_a in all_sets:
            for set_b in all_sets:
                for set_c in all_sets:
                    model = {terms[0]: set_a, terms[1]: set_b, terms[2]: set_c}
                    if self._satisfies(model, premises):
                        if not self._statement_holds(model, concl):
                            return False
        return True

    def _all_sets(self, universe: List[int]) -> List[set]:
        sets = []
        n = len(universe)
        for mask in range(1 << n):
            subset = set()
            for i in range(n):
                if mask & (1 << i):
                    subset.add(universe[i])
            sets.append(subset)
        return sets

    def _satisfies(self, model: Dict[str, set], premises: List[Tuple[str, str, str]]) -> bool:
        return all(self._statement_holds(model, p) for p in premises)

    def _statement_holds(self, model: Dict[str, set], stmt: Tuple[str, str, str]) -> bool:
        kind, x, y = stmt
        set_x = model.get(x, set())
        set_y = model.get(y, set())
        if kind == "all":
            return set_x.issubset(set_y)
        if kind == "some":
            return len(set_x.intersection(set_y)) > 0
        if kind == "no":
            return len(set_x.intersection(set_y)) == 0
        if kind == "some_not":
            return len(set_x.difference(set_y)) > 0
        return False

    def _render_statement(self, stmt: Tuple[str, str, str]) -> str:
        kind, x, y = stmt
        if kind == "all":
            return f"All {x} are {y}."
        if kind == "some":
            return f"Some {x} are {y}."
        if kind == "no":
            return f"No {x} are {y}."
        if kind == "some_not":
            return f"Some {x} are not {y}."
        return ""

    def _generate_series(self) -> GeneratedItem:
        start_letter = self.rng.choice(["A", "B", "C", "D"])
        start_num = self.rng.randint(1, 5)
        letter_steps = [1, 2, 3, 4]
        num_steps = [2, 3, 4, 5]

        letters = [start_letter]
        nums = [start_num]
        for i in range(1, 5):
            letters.append(chr(ord(letters[-1]) + letter_steps[i - 1]))
            nums.append(nums[-1] + num_steps[i - 1])

        terms = [f"{l}{n}" for l, n in zip(letters, nums)]
        correct = terms[-1]
        question = "Find the next term in the series: " + ", ".join(terms[:-1]) + ", ?"

        distractors = {
            f"{letters[-2]}{nums[-1]}",
            f"{letters[-1]}{nums[-2]}",
            f"{letters[-1]}{nums[-1] + 1}",
            f"{chr(ord(letters[-1]) + 1)}{nums[-1]}",
        }
        distractors = [d for d in distractors if d != correct]
        self.rng.shuffle(distractors)
        options = [correct] + distractors[:3]
        self.rng.shuffle(options)

        choices = [f"{letter}) {opt}" for letter, opt in zip(["A", "B", "C", "D"], options)]
        answer = self._correct_letter(choices, correct)
        explanation = f"The series follows: letters increase by {letter_steps} and numbers increase by {num_steps}, giving {correct}."
        return GeneratedItem("Mixed Series (Alphanumeric)", question, choices, answer, explanation)

    def _pick_names(self, n: int) -> List[str]:
        pool = ["Aarav", "Bela", "Chirag", "Diya", "Eshan", "Farah", "Gopal", "Heena", "Ishaan", "Jia"]
        self.rng.shuffle(pool)
        return pool[:n]

    def _make_choices(self, correct_name: str, names: List[str]) -> List[str]:
        distractors = [n for n in names if n != correct_name]
        self.rng.shuffle(distractors)
        options = [correct_name] + distractors[:3]
        self.rng.shuffle(options)
        return [f"{letter}) {opt}" for letter, opt in zip(["A", "B", "C", "D"], options)]

    def _correct_letter(self, choices: List[str], correct: str) -> str:
        for choice in choices:
            if choice.endswith(correct):
                return choice.split(")")[0]
        return "A"

    def _permute(self, items: List[str]) -> List[Tuple[str, ...]]:
        if not items:
            return [()]
        res = []
        for i, item in enumerate(items):
            for rest in self._permute(items[:i] + items[i + 1:]):
                res.append((item,) + rest)
        return res


class QAgent:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        lora_path: str = "./q_model_lora",
        use_llm_rewrite: bool = True,
        **_: object,
    ) -> None:
        # Import transformers only when QAgent is initialized
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            self.AutoModelForCausalLM = AutoModelForCausalLM
            self.AutoTokenizer = AutoTokenizer
            self.PeftModel = PeftModel
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency 'transformers' and/or 'peft'. The AAIPL question agent requires them "
                "to load Qwen/Qwen3-4B with LoRA."
            ) from e
        
        self.generator = StructureGenerator()
        self.use_llm_rewrite = use_llm_rewrite

        cache_dir = os.getenv("TRANSFORMERS_CACHE")

        self.tokenizer = self.AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            cache_dir=cache_dir,
        )
        self.model = self.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_dir,
        )
        if lora_path:
            try:
                self.model = self.PeftModel.from_pretrained(self.model, lora_path)
            except Exception:
                pass
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(
        self,
        message: str | List[str],
        system_prompt: Optional[str] = None,
        **kwargs: object,
    ) -> Tuple[str | List[str], int | None, float | None]:
        if system_prompt is None:
            system_prompt = "Generate a tricky reasoning MCQ in strict JSON format. Transform simple statements into adversarial questions."

        prompts = [message] if isinstance(message, str) else list(message)
        results: List[str] = []

        tgps_show = bool(getattr(kwargs, "get", lambda *_: False)("tgps_show", False))
        max_new_tokens = int(getattr(kwargs, "get", lambda *_: 300)("max_new_tokens", 300))
        temperature = float(getattr(kwargs, "get", lambda *_: 0.7)("temperature", 0.7))
        top_p = float(getattr(kwargs, "get", lambda *_: 0.9)("top_p", 0.9))

        t0 = time.time() if tgps_show else 0.0
        total_tokens = 0

        for msg in prompts:
            domain = self._infer_domain(msg)
            item = self.generator.generate(domain)
            ok, _ = self.generator.validate(item)
            if not ok:
                item = self.generator.generate("Syllogisms")

            question_text = item.question
            if self.use_llm_rewrite:
                question_text = self._rewrite_question(question_text, temperature, top_p, max_new_tokens)

            payload = {
                "topic": item.domain,
                "question": question_text,
                "choices": item.choices,
                "answer": item.expected_answer,
                "explanation": item.explanation,
            }
            results.append(json.dumps(payload, ensure_ascii=False, indent=2))

        dt = (time.time() - t0) if tgps_show else None
        out: str | List[str] = results[0] if isinstance(message, str) else results
        return out, (total_tokens if tgps_show else None), dt

    def _infer_domain(self, prompt: str) -> str:
        text = prompt.lower()
        if "seating" in text or "arrangement" in text:
            return "Seating Arrangements"
        if "blood" in text or "relation" in text or "family" in text:
            return "Blood Relations"
        if "syllog" in text or "conclusion" in text:
            return "Syllogisms"
        if "series" in text or "alphanumeric" in text:
            return "Mixed Series (Alphanumeric)"
        return self.generator.rng.choice(DOMAINS)

    def _rewrite_question(self, text: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
        """Rewrite question to be more tricky and adversarial while keeping facts the same."""
        chat = [
            {
                "role": "system",
                "content": (
                    "Transform the given question into a tricky, adversarial version. "
                    "Make it more challenging by:"
                    "1. Using complex sentence structures and elaborate phrasing"
                    "2. Adding conditional clauses and embedded statements"
                    "3. Using indirect references instead of direct ones"
                    "4. Including distractors in the question text"
                    "5. Using double negatives or inverse logic where appropriate\n"
                    "IMPORTANT: Keep all facts, relationships, and the correct answer unchanged. "
                    "Output ONLY the rewritten question text, nothing else."
                ),
            },
            {"role": "user", "content": f"Original question: {text}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_text = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        if new_text:
            return new_text
        return text


def generate_20_questions(output_file: str = "generated_questions.json", max_latency: float = 12.0) -> None:
    """Generate 20 questions (5 from each of 4 domains) with latency tracking."""
    generator = StructureGenerator()
    questions = []
    
    print(f"\nGenerating 20 questions from 4 domains (5 each)...\n")
    print(f"{'Domain':<35} {'Latency (s)':<15} {'Status'}")
    print("-" * 65)
    
    for domain in DOMAINS:
        for i in range(5):
            start_time = time.time()
            try:
                item = generator.generate(domain)
                valid, msg = generator.validate(item)
                latency = time.time() - start_time
                
                if not valid:
                    status = f"Invalid: {msg}"
                elif latency > max_latency:
                    status = f"Slow: {latency:.2f}s > {max_latency}s"
                else:
                    status = f"OK"
                
                print(f"{domain:<35} {latency:<15.3f} {status}")
                
                questions.append({
                    "topic": item.domain,
                    "question": item.question,
                    "choices": item.choices,
                    "answer": item.expected_answer,
                    "explanation": item.explanation,
                    "latency_seconds": round(latency, 3)
                })
            except Exception as e:
                latency = time.time() - start_time
                print(f"{domain:<35} {latency:<15.3f} Error: {str(e)[:30]}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 65)
    print(f"Generated {len(questions)} questions -> {output_file}")
    avg_latency = sum(q["latency_seconds"] for q in questions) / len(questions) if questions else 0
    print(f"Average latency: {avg_latency:.3f}s (target: <{max_latency}s)")
    print("=" * 65 + "\n")


def train_q_model_from_dataset(
    train_path: str,
    val_path: str,
    output_dir: str,
    max_seq_length: int = 1024,
    batch_size: int = 32,
    epochs: int = 2,
) -> None:
    """Train Q-model on instruction-response dataset using Unsloth + LoRA."""
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependencies for training. Install: unsloth, trl, transformers"
        ) from e

    model_name = "Qwen/Qwen3-4B"
    
    print(f"\n{'='*70}")
    print("Training Q-Model (Qwen3-4B) with Unsloth + LoRA")
    print(f"{'='*70}\n")
    print(f"Train data: {train_path}")
    print(f"Val data: {val_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, Max Seq Len: {max_seq_length}")
    print()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    def format_example(example: Dict[str, object]) -> str:
        instruction = example["instruction"]
        input_text = example["input"]
        output_text = json.dumps(example["output"], ensure_ascii=True)
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        )

    train_text = [format_example(e) for e in train_data]
    val_text = [format_example(e) for e in val_data]

    train_dataset = Dataset.from_dict({"text": train_text})
    val_dataset = Dataset.from_dict({"text": val_text})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=2e-4,
            optim="adamw_torch",
            bf16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=2,
            report_to="none",
        ),
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Training complete! Model saved to {output_dir}\n")


def build_training_dataset_from_hf(
    output_train: str,
    output_val: str,
    seed: int = 42,
) -> Tuple[int, int]:
    """Build training dataset from HuggingFace datasets."""
    from agents.question_agent import build_training_dataset
    return build_training_dataset(output_train, output_val, seed)


def cmd_train(args: object) -> None:
    """CLI command: Train model."""
    train_q_model_from_dataset(
        train_path=getattr(args, "train", "train.json"),
        val_path=getattr(args, "val", "val.json"),
        output_dir=getattr(args, "output", "./q_model_lora"),
        batch_size=int(getattr(args, "batch", 32)),
        epochs=int(getattr(args, "epochs", 2)),
    )


def cmd_build_dataset(args: object) -> None:
    """CLI command: Build training dataset from HuggingFace."""
    train_count, val_count = build_training_dataset_from_hf(
        output_train=getattr(args, "train", "train.json"),
        output_val=getattr(args, "val", "val.json"),
    )
    print(f"\n✅ Dataset built: {train_count} train, {val_count} validation")
    print(f"Train: {getattr(args, 'train', 'train.json')}")
    print(f"Val: {getattr(args, 'val', 'val.json')}\n")


def cmd_generate(args: object) -> None:
    """CLI command: Generate 20 questions."""
    lora_path = getattr(args, "lora", "./q_model_lora")
    model = QAgent(lora_path=lora_path, use_llm_rewrite=True)
    generate_20_questions(output_file=getattr(args, "output", "generated_questions.json"))


def main() -> None:
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Q-Model: Question Generation with Training")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # build-dataset command
    build_parser = subparsers.add_parser("build-dataset", help="Build training dataset from HuggingFace")
    build_parser.add_argument("--train", default="train.json", help="Output train file")
    build_parser.add_argument("--val", default="val.json", help="Output val file")
    build_parser.set_defaults(func=cmd_build_dataset)
    
    # train command
    train_parser = subparsers.add_parser("train", help="Train Q-model")
    train_parser.add_argument("--train", default="train.json", help="Train data file")
    train_parser.add_argument("--val", default="val.json", help="Val data file")
    train_parser.add_argument("--output", default="./q_model_lora", help="Output directory")
    train_parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    train_parser.add_argument("--batch", type=int, default=32, help="Batch size")
    train_parser.set_defaults(func=cmd_train)
    
    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate 20 questions")
    gen_parser.add_argument("--lora", default="./q_model_lora", help="LoRA model path")
    gen_parser.add_argument("--output", default="generated_questions.json", help="Output file")
    gen_parser.set_defaults(func=cmd_generate)
    
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

