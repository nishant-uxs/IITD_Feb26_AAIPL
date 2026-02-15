import json
from pathlib import Path


def valid(q):

    required = ["topic","question","choices","answer","explanation"]

    for r in required:
        if r not in q:
            return False

    if len(q["choices"]) != 4:
        return False

    if q["answer"] not in ["A","B","C","D"]:
        return False

    return True


def filter_questions(input_file, output_file):

    with open(input_file) as f:
        data = json.load(f)

    filtered = [q for q in data if valid(q)]

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(filtered, f, indent=4)

    print("✅ Filtered:", len(filtered))


if __name__ == "__main__":
    filter_questions(
        "outputs/questions.json",
        "outputs/filtered_questions.json"
    )
