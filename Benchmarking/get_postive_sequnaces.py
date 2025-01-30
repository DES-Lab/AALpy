import json
import random

from aalpy import run_PAPNI
from aalpy.automata import VpaAlphabet


def is_valid_json(s):
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def generate_random_json(valid=True, max_depth=3, max_elements=5):
    """
    Generate a random JSON string using only predefined symbols.
    If valid=True, returns a syntactically correct JSON string.
    If valid=False, introduces errors to create an invalid JSON string.
    """
    call_set = ['{', '[']
    return_set = ['}', ']']
    internal_set = [':', ',', 'key', 'val']
    alphabet = call_set + return_set + internal_set

    def generate_valid_json(depth=0):
        if depth >= max_depth:
            return '"val"'

        if random.choice([True, False]):
            return "{" + ", ".join(f'"key": "val"' for _ in range(random.randint(1, max_elements))) + "}"
        else:
            elements = [generate_valid_json(depth + 1) for _ in range(random.randint(1, max_elements))]
            return "[" + ", ".join(elements) + "]"

    json_string = generate_valid_json()

    if not valid:
        if random.choice([True, False]):
            # Introduce random errors (e.g., missing brackets, missing quotes, misplaced commas)
            if "{" in json_string or "[" in json_string:
                json_string = json_string.replace("{", "", 1) if "{" in json_string else json_string.replace("[", "", 1)
            if "}" in json_string or "]" in json_string:
                json_string = json_string.replace("}", "", 1) if "}" in json_string else json_string.replace("]", "", 1)
        else:
            json_string = json_string.replace(":", " ", 1)  # Remove a colon to break key-value syntax

    json_tuple = tuple(
        json_string.replace('{', ' { ').replace('}', ' } ').replace('[', ' [ ').replace(']', ' ] ').replace(':',
                                                                                                            ' : ').replace(
            ',', ' , ').split())

    return json_string, json_tuple

# Define call, return, and internal symbols for JSON
call_set = ['{', '[']
return_set = ['}', ']']
internal_set = [':', ',', 'key', 'val']

# Define the input alphabet
vpa_alphabet = VpaAlphabet(
    internal_alphabet=internal_set,
    call_alphabet=call_set,
    return_alphabet=return_set
)

dataset = []
for _ in range(20):
    ts, tr = generate_random_json(valid=True, max_depth=5, max_elements=2)
    print('valid', ts)
    assert is_valid_json(ts), ts
    dataset.append((tr, True))

    ts, tr = generate_random_json(valid=False, max_depth=5, max_elements=2)
    print('invalid', ts)
    assert not is_valid_json(ts), ts
    dataset.append((tr, False))

learned_json_model = run_PAPNI(dataset, vpa_alphabet)

learned_json_model.visualize()