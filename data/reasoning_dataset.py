# Diverse reasoning dataset for Coconut training
# Includes arithmetic, logic, and multi-step reasoning problems

import random
from typing import List, Dict

# Arithmetic problems (1-2 steps)
ARITHMETIC_EASY = [
    {
        "question": "What is 5 + 3?",
        "steps": ["I need to add 5 and 3.", "5 + 3 = 8."],
        "answer": "8",
        "complexity": 1
    },
    {
        "question": "What is 12 - 7?",
        "steps": ["I need to subtract 7 from 12.", "12 - 7 = 5."],
        "answer": "5",
        "complexity": 1
    },
    {
        "question": "What is 6 × 4?",
        "steps": ["I need to multiply 6 by 4.", "6 × 4 = 24."],
        "answer": "24",
        "complexity": 1
    },
    {
        "question": "What is 20 ÷ 5?",
        "steps": ["I need to divide 20 by 5.", "20 ÷ 5 = 4."],
        "answer": "4",
        "complexity": 1
    },
    {
        "question": "What is 15 + 8?",
        "steps": ["I need to add 15 and 8.", "15 + 8 = 23."],
        "answer": "23",
        "complexity": 1
    },
    {
        "question": "What is 30 - 12?",
        "steps": ["I need to subtract 12 from 30.", "30 - 12 = 18."],
        "answer": "18",
        "complexity": 1
    },
]

# Multi-step arithmetic (2-3 steps)
ARITHMETIC_MEDIUM = [
    {
        "question": "Tom has 8 apples. He gives 3 to Mary. How many does he have left?",
        "steps": [
            "Tom starts with 8 apples.",
            "He gives away 3 apples.",
            "8 - 3 = 5 apples remaining."
        ],
        "answer": "5",
        "complexity": 2
    },
    {
        "question": "A store has 15 books. They sell 7 and receive 4 more. How many books do they have?",
        "steps": [
            "Start with 15 books.",
            "After selling 7: 15 - 7 = 8 books.",
            "After receiving 4: 8 + 4 = 12 books."
        ],
        "answer": "12",
        "complexity": 3
    },
    {
        "question": "Sara has 10 dollars. She buys a book for 4 dollars and a pen for 2 dollars. How much money does she have left?",
        "steps": [
            "Sara starts with 10 dollars.",
            "Book costs 4 dollars, pen costs 2 dollars.",
            "Total spent: 4 + 2 = 6 dollars.",
            "Remaining: 10 - 6 = 4 dollars."
        ],
        "answer": "4",
        "complexity": 3
    },
    {
        "question": "There are 5 red balls and 3 blue balls. If we add 4 more red balls, what is the total number of balls?",
        "steps": [
            "Initial red balls: 5, blue balls: 3.",
            "Add 4 red balls: 5 + 4 = 9 red balls.",
            "Total: 9 + 3 = 12 balls."
        ],
        "answer": "12",
        "complexity": 2
    },
    {
        "question": "A farmer has 20 chickens. He sells 8 and then buys 5 more. How many chickens does he have?",
        "steps": [
            "Start with 20 chickens.",
            "After selling 8: 20 - 8 = 12 chickens.",
            "After buying 5: 12 + 5 = 17 chickens."
        ],
        "answer": "17",
        "complexity": 3
    },
]

# Logic problems (2-4 steps)
LOGIC_PROBLEMS = [
    {
        "question": "If all cats are animals, and Fluffy is a cat, is Fluffy an animal?",
        "steps": [
            "Given: All cats are animals.",
            "Given: Fluffy is a cat.",
            "Since Fluffy is a cat, and all cats are animals, Fluffy must be an animal."
        ],
        "answer": "Yes",
        "complexity": 2
    },
    {
        "question": "If it rains, the ground gets wet. The ground is wet. Did it rain?",
        "steps": [
            "Given: Rain causes wet ground.",
            "Given: The ground is wet.",
            "The ground could be wet for other reasons (sprinkler, etc.).",
            "We cannot conclude it definitely rained."
        ],
        "answer": "Not necessarily",
        "complexity": 3
    },
    {
        "question": "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?",
        "steps": [
            "Alice > Bob in height.",
            "Bob > Carol in height.",
            "Therefore: Alice > Bob > Carol.",
            "Carol is the shortest."
        ],
        "answer": "Carol",
        "complexity": 2
    },
    {
        "question": "A box contains only red and blue marbles. There are 3 red marbles and the total is 7. How many blue marbles are there?",
        "steps": [
            "Total marbles: 7.",
            "Red marbles: 3.",
            "Blue marbles: 7 - 3 = 4."
        ],
        "answer": "4",
        "complexity": 2
    },
    {
        "question": "If today is Monday, what day will it be in 3 days?",
        "steps": [
            "Today is Monday.",
            "In 1 day: Tuesday.",
            "In 2 days: Wednesday.",
            "In 3 days: Thursday."
        ],
        "answer": "Thursday",
        "complexity": 2
    },
]

# Word problems requiring more reasoning (3-5 steps)
WORD_PROBLEMS = [
    {
        "question": "A train travels at 60 km/h for 2 hours. How far does it travel?",
        "steps": [
            "Speed = 60 km/h.",
            "Time = 2 hours.",
            "Distance = Speed × Time.",
            "Distance = 60 × 2 = 120 km."
        ],
        "answer": "120 km",
        "complexity": 3
    },
    {
        "question": "John has twice as many apples as Mary. Mary has 6 apples. How many apples does John have?",
        "steps": [
            "Mary has 6 apples.",
            "John has twice as many as Mary.",
            "John's apples = 2 × 6 = 12."
        ],
        "answer": "12",
        "complexity": 2
    },
    {
        "question": "A rectangle has length 8 and width 5. What is its area?",
        "steps": [
            "Length = 8, Width = 5.",
            "Area = Length × Width.",
            "Area = 8 × 5 = 40."
        ],
        "answer": "40",
        "complexity": 2
    },
    {
        "question": "If 3 notebooks cost 15 dollars, how much does 1 notebook cost?",
        "steps": [
            "3 notebooks = 15 dollars.",
            "1 notebook = 15 ÷ 3 = 5 dollars."
        ],
        "answer": "5 dollars",
        "complexity": 2
    },
    {
        "question": "A class has 24 students. Half are boys. How many girls are there?",
        "steps": [
            "Total students: 24.",
            "Half are boys: 24 ÷ 2 = 12 boys.",
            "Girls = Total - Boys = 24 - 12 = 12."
        ],
        "answer": "12",
        "complexity": 2
    },
    {
        "question": "Tom is 10 years old. His father is 3 times as old. How old is his father?",
        "steps": [
            "Tom's age: 10 years.",
            "Father is 3 times as old.",
            "Father's age = 3 × 10 = 30 years."
        ],
        "answer": "30",
        "complexity": 2
    },
]

# Comparison problems
COMPARISON_PROBLEMS = [
    {
        "question": "Which is greater: 3/4 or 2/3?",
        "steps": [
            "Convert to same denominator.",
            "3/4 = 9/12.",
            "2/3 = 8/12.",
            "9/12 > 8/12, so 3/4 is greater."
        ],
        "answer": "3/4",
        "complexity": 3
    },
    {
        "question": "Is 15 a prime number?",
        "steps": [
            "Check if 15 has factors other than 1 and 15.",
            "15 = 3 × 5.",
            "15 has factors 3 and 5.",
            "Therefore, 15 is not prime."
        ],
        "answer": "No",
        "complexity": 3
    },
    {
        "question": "What is larger: 2^3 or 3^2?",
        "steps": [
            "Calculate 2^3 = 2 × 2 × 2 = 8.",
            "Calculate 3^2 = 3 × 3 = 9.",
            "9 > 8, so 3^2 is larger."
        ],
        "answer": "3^2 (which equals 9)",
        "complexity": 2
    },
]


def get_all_problems() -> List[Dict]:
    """Return all problems from all categories."""
    return (
        ARITHMETIC_EASY +
        ARITHMETIC_MEDIUM +
        LOGIC_PROBLEMS +
        WORD_PROBLEMS +
        COMPARISON_PROBLEMS
    )


def get_problems_by_complexity(min_complexity: int = 1, max_complexity: int = 5) -> List[Dict]:
    """Return problems filtered by complexity range."""
    all_problems = get_all_problems()
    return [
        p for p in all_problems
        if min_complexity <= p.get("complexity", 2) <= max_complexity
    ]


def get_train_test_split(test_ratio: float = 0.2, seed: int = 42) -> tuple:
    """Split problems into train and test sets."""
    random.seed(seed)
    all_problems = get_all_problems()
    random.shuffle(all_problems)

    split_idx = int(len(all_problems) * (1 - test_ratio))
    return all_problems[:split_idx], all_problems[split_idx:]


def generate_similar_problem(base_problem: Dict) -> Dict:
    """Generate a variation of a problem by changing numbers."""
    import re

    question = base_problem["question"]
    answer = base_problem["answer"]
    steps = base_problem["steps"]

    # Find all numbers in the question
    numbers = re.findall(r'\d+', question)

    if len(numbers) >= 2:
        # Generate new random numbers
        new_numbers = [random.randint(2, 20) for _ in numbers]

        # Simple replacement (works for basic arithmetic)
        new_question = question
        for old, new in zip(numbers, new_numbers):
            new_question = new_question.replace(old, str(new), 1)

        return {
            "question": new_question,
            "steps": steps,  # Steps would need recalculation for real use
            "answer": answer,  # Answer would need recalculation for real use
            "complexity": base_problem.get("complexity", 2),
            "is_generated": True
        }

    return base_problem


if __name__ == "__main__":
    # Print dataset statistics
    all_problems = get_all_problems()
    print(f"Total problems: {len(all_problems)}")

    by_complexity = {}
    for p in all_problems:
        c = p.get("complexity", 2)
        by_complexity[c] = by_complexity.get(c, 0) + 1

    print("\nBy complexity:")
    for c in sorted(by_complexity.keys()):
        print(f"  Complexity {c}: {by_complexity[c]} problems")

    print("\nSample problems:")
    for p in random.sample(all_problems, 3):
        print(f"\n  Q: {p['question']}")
        print(f"  A: {p['answer']}")
        print(f"  Complexity: {p['complexity']}")
        print(f"  Steps: {len(p['steps'])}")
