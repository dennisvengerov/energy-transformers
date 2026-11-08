"""The eval prompt must reproduce TinyGSM's docstring layout byte-for-byte and token-for-token."""

import pytest
from common import EOS_ID, format_prompt, get_tokenizer, split_into_sentences, split_prompt_solution

# Verbatim TinyGSM rows: both quote styles, and the splitter's abbreviation / initial handling.
ROWS = [
    {"question": 'Mark has 10 crayons. He gives 2 crayons to his younger sister and loses another 4 while he was playing. How many crayons does Mark have left?',
     "code": 'def simple_math_problem() -> int:\n    """\n    Mark has 10 crayons.\n    He gives 2 crayons to his younger sister and loses another 4 while he was playing.\n    How many crayons does Mark have left?\n    """\n    crayonsTotal = 10\n    crayonsGiven = 2\n    crayonsLost = 4\n    crayonsLeft = crayonsTotal - crayonsGiven - crayonsLost\n    result = crayonsLeft\n\n    return result\n'},
    {"question": 'A toy car goes 6 meters in 2 seconds. How far will it go in 5 seconds?',
     "code": "def simple_math_problem() -> int:\n    '''\n    A toy car goes 6 meters in 2 seconds.\n    How far will it go in 5 seconds?\n    '''\n    distance_per_time = 6 / 2\n    time = 5\n    total_distance = distance_per_time * time\n    result = total_distance\n    return result\n"},
    {"question": 'Lucy, Max, and Ethan had to meet for a group project at 4:00 p.m. Lucy arrived at 4:20 p.m. Max arrived 15 minutes after Lucy, and Ethan was twice as late as Max. How late, in minutes, was Ethan to the project meeting?',
     "code": "def simple_math_problem() -> int:\n    '''\n    Lucy, Max, and Ethan had to meet for a group project at 4:00 p.m. Lucy arrived at 4:20 p.m. Max arrived 15 minutes after Lucy, and Ethan was twice as late as Max.\n    How late, in minutes, was Ethan to the project meeting?\n    '''\n    lucy_arrival = 20\n    max_arrival = lucy_arrival + 15\n    ethan_arrival = max_arrival * 2\n    \n    result = ethan_arrival\n    \n    return result\n"},
    {"question": 'There are two containers A and B. Container A has 5 liters of water and container B has 8 liters of water. Three liters of water were poured from container B into container A. Then, 2 liters of water were poured from container A into container B. What is the ratio of the number of liters of water in container A to that in container B?',
     "code": "def simple_math_problem() -> int:\n    '''\n    There are two containers A and B. Container A has 5 liters of water and container B has 8 liters of water.\n    Three liters of water were poured from container B into container A. Then, 2 liters of water were poured from container A into container B. What is the ratio of the number of liters of water in container A to that in container B?\n    '''\n    a = 5+2-3 #liters of water in container A\n    b = 8-3+2 #liters of water in container B\n    result = a/b\n    return result\n"},
]


@pytest.mark.parametrize("row", ROWS)
def test_prompt_reproduces_tinygsm_docstring(row):
    prompt, solution = split_prompt_solution(row["question"], row["code"])
    quote = row["code"].split("\n")[1].strip()
    assert prompt == format_prompt(row["question"], quote)
    assert prompt + solution == row["code"]
    assert solution.startswith("    ") and not solution.startswith("    '''")


@pytest.mark.parametrize("row", ROWS)
def test_prompt_tokens_are_a_prefix_of_training_tokens(row):
    tok = get_tokenizer()
    prompt, _ = split_prompt_solution(row["question"], row["code"])
    prompt_ids = [EOS_ID] + tok.encode(prompt).ids
    doc_ids = [EOS_ID] + tok.encode(row["code"]).ids  # a training document, as in prepare_data.py
    assert doc_ids[:len(prompt_ids)] == prompt_ids
    assert tok.decode(doc_ids[1:]) == row["code"]


def test_gsm8k_prompt_layout():
    question = "Mr. Bert sells pens for $2.50 each.  He opens at 9 a.m. today. How much do 3 pens cost?"
    assert format_prompt(question) == (
        "def simple_math_problem() -> int:\n"
        "    '''\n"
        "    Mr. Bert sells pens for $2.50 each.\n"
        "    He opens at 9 a.m. today.\n"
        "    How much do 3 pens cost?\n"
        "    '''\n"
    )


def test_sentence_splitter_edge_cases():
    assert split_into_sentences("Question X+1:..... A cat sat.") == ["Question X+1:.....", "A cat sat."]
    assert split_into_sentences("Swim 200 m") == ["Swim 200 m"]  # no final punctuation: text kept
    assert split_into_sentences("Is it 5? Yes! Done.") == ["Is it 5?", "Yes!", "Done."]
    assert split_prompt_solution("Some other question.", ROWS[0]["code"]) is None
