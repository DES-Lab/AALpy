import json
import random
from collections import defaultdict

from aalpy import run_PAPNI, load_automaton_from_file
from aalpy.automata import VpaAlphabet
from aalpy.utils import generate_input_output_data_from_vpa




def is_valid_json(s):
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def to_json_string(json_tuple):
    json_str = ''
    for x in json_tuple:
        if x in {'key', 'val'}:
            json_str += f'\"{x}\"'
        else:
            json_str += x
    return json_str


def generate_random_json(max_depth=3, max_elements=5):
    """
    Generate a random valid JSON structure using abstract symbols.
    Args:
        max_depth (int): Maximum nesting depth
        max_elements (int): Maximum number of elements in arrays or objects
    Returns:
        list: List of symbols representing a valid JSON structure
    """

    def generate_value(current_depth):
        if current_depth >= max_depth:
            return ['val']

        # Choose between simple value, array, or object
        choice = random.choice(['simple', 'array', 'object'])

        if choice == 'simple':
            return ['val']
        elif choice == 'array':
            return generate_array(current_depth)
        else:
            return generate_object(current_depth)

    def generate_array(current_depth):
        # Generate array with random number of elements
        num_elements = random.randint(1, max_elements)
        result = ['[']

        for i in range(num_elements):
            result.extend(generate_value(current_depth + 1))
            if i < num_elements - 1:
                result.append(',')

        result.append(']')
        return result

    def generate_object(current_depth):
        # Generate object with random number of key-value pairs
        num_pairs = random.randint(1, max_elements)
        result = ['{']

        for i in range(num_pairs):
            result.append('key')
            result.append(':')
            result.extend(generate_value(current_depth + 1))
            if i < num_pairs - 1:
                result.append(',')

        result.append('}')
        return result

    # Start generation with an object at the root
    json_tuple = tuple(generate_object(0))
    json_str = to_json_string(json_tuple)
    return json_str, json_tuple


def corrupt_json(symbols):
    """
    Take a valid JSON symbol list and make it invalid using various strategies.
    Returns corrupted symbols and description of corruption.
    """
    corrupted = symbols.copy()

    strategies = [
        'bracket_mismatch',  # Replace brackets with mismatched ones
        'add_symbol',  # Add inappropriate symbol
        'drop_symbol',  # Remove necessary symbol
        'naked_value',  # Just a lone value
        'naked_key',  # Just a lone key
        'multiple_commas',  # Add multiple consecutive commas
        'trailing_comma',  # Add comma at the end
        'missing_value',  # Remove value after key
        'missing_key',  # Remove key before value
        'missing_colon',  # Remove colon between key-value
        'standalone_colon',  # Just a colon
        'comma_start_end',  # Start or end with comma
        'empty_structure',  # Empty structure with internal comma
        'multiple_colons'  # Multiple colons in key-value pair
    ]

    strategy = random.choice(strategies)

    if strategy == 'bracket_mismatch':
        bracket_positions = [i for i, s in enumerate(corrupted)
                             if s in ['{', '}', '[', ']']]
        if bracket_positions:
            pos = random.choice(bracket_positions)
            original = corrupted[pos]
            corruptions = {
                '{': random.choice(['}', ']']),
                '}': random.choice(['{', '[']),
                '[': random.choice(['}', ']']),
                ']': random.choice(['{', '['])
            }
            corrupted[pos] = corruptions[original]
            reason = f"Replaced '{original}' with '{corrupted[pos]}'"

    elif strategy == 'naked_value':
        corrupted = ['val']
        reason = "Created lone value without structure"

    elif strategy == 'naked_key':
        corrupted = ['key']
        reason = "Created lone key without value or structure"

    elif strategy == 'multiple_commas':
        num_commas = random.randint(2, 4)
        corrupted = ['{'] + [',' for _ in range(num_commas)] + ['}']
        reason = f"Created structure with {num_commas} consecutive commas"

    elif strategy == 'trailing_comma':
        if len(corrupted) > 2:  # Need at least {} or []
            if corrupted[-1] in ['}', ']']:
                corrupted.insert(-1, ',')
                reason = "Added trailing comma before closing bracket"
            else:
                corrupted.append(',')
                reason = "Added trailing comma at end"
        else:
            corrupted = ['{', ',', '}']
            reason = "Added trailing comma in empty structure"

    elif strategy == 'missing_value':
        key_positions = [i for i, s in enumerate(corrupted) if s == 'key']
        if key_positions:
            pos = random.choice(key_positions)
            if pos + 2 < len(corrupted) and corrupted[pos + 1] == ':':
                corrupted.pop(pos + 2)  # Remove value
                reason = "Removed value after key"
            else:
                corrupted = ['{', 'key', ':', '}']
                reason = "Created key without value"
        else:
            corrupted = ['{', 'key', ':', '}']
            reason = "Created key without value"

    elif strategy == 'missing_key':
        corrupted = ['{', ':', 'val', '}']
        reason = "Created value with missing key"

    elif strategy == 'missing_colon':
        colon_positions = [i for i, s in enumerate(corrupted) if s == ':']
        if colon_positions:
            pos = random.choice(colon_positions)
            corrupted.pop(pos)
            reason = "Removed colon between key and value"
        else:
            corrupted = ['{', 'key', 'val', '}']
            reason = "Created key-value pair without colon"

    elif strategy == 'standalone_colon':
        corrupted = [':']
        reason = "Created standalone colon"

    elif strategy == 'comma_start_end':
        if random.choice([True, False]):
            corrupted = [','] + corrupted
            reason = "Added comma at start"
        else:
            corrupted.append(',')
            reason = "Added comma at end"

    elif strategy == 'empty_structure':
        structure_type = random.choice(['{', '['])
        closing = '}' if structure_type == '{' else ']'
        corrupted = [structure_type, ',', closing]
        reason = f"Created empty {structure_type}{closing} with internal comma"

    elif strategy == 'multiple_colons':
        colon_count = random.randint(2, 3)
        corrupted = ['{', 'key'] + [':' for _ in range(colon_count)] + ['val', '}']
        reason = f"Added {colon_count} colons between key and value"

    elif strategy == 'add_symbol':
        possible_additions = ['key', 'val', ',', ':', '{', '}', '[', ']']
        symbol_to_add = random.choice(possible_additions)
        pos = random.randint(0, len(corrupted))
        corrupted.insert(pos, symbol_to_add)
        reason = f"Added '{symbol_to_add}' at position {pos}"

    else:  # drop_symbol
        if len(corrupted) > 1:
            pos = random.randint(0, len(corrupted) - 1)
            dropped = corrupted.pop(pos)
            reason = f"Dropped '{dropped}' from position {pos}"
        else:
            corrupted = []
            reason = "Dropped all symbols"

    return corrupted


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


def generate_dataset(num_sequances):
    dataset = set()
    while len(dataset) <= num_sequances:
        ts, tt = generate_random_json(max_depth=2, max_elements=3)
        assert is_valid_json(ts), ts
        dataset.add((tuple(tt), True))

        for _ in range(5):
            ft = corrupt_json(list(tt))
            json_str = ''.join(ft)
            if not is_valid_json(json_str):
                dataset.add((tuple(ft), False))

    return dataset


def validate_string_with_json_parser(json_str, json_parser):
    assert json_parser in {'json', 'ujson', 'orjson', 'simplejson', 'demjson', 'pyjson5'}

    import ujson
    import orjson
    import json5
    import simplejson as sj
    import demjson
    import pyjson5

    try:
        if json_parser == "json":
            json.loads(json_str)
        elif json_parser == "ujson":
            ujson.loads(json_str)
        elif json_parser == "orjson":
            orjson.loads(json_str)
        elif json_parser == "json5":
            json5.loads(json_str)
        elif json_parser == "simplejson":
            sj.loads(json_str)
        elif json_parser == "demjson":
            demjson.decode(json_str)
        elif json_parser == "pyjson5":
            pyjson5.loads(json_str)
        else:
            raise ValueError("Unsupported JSON parser")
        return True
    except (json.JSONDecodeError, ujson.JSONDecodeError, orjson.JSONDecodeError,
            demjson.JSONDecodeError, pyjson5.Json5Exception, ValueError) as e:
        return False


use_learned_model = False

model_learning_dataset = []
if not use_learned_model:
    model_learning_dataset = generate_dataset(num_sequances=20000)

    learned_json_model = run_PAPNI(model_learning_dataset, vpa_alphabet)
    # learned_json_model.visualize()

    learned_json_model.save('learned_json.dot')
else:
    learned_json_model = load_automaton_from_file('learned_json.dot', 'vpa')

parsers_under_test = ['json', 'ujson', 'orjson', 'simplejson', 'demjson', 'pyjson5']

num_learning_iterations = 3

disagreements = defaultdict(list)

results = defaultdict(list)

for _ in range(num_learning_iterations):
    disagreements.clear()

    test_dataset = generate_input_output_data_from_vpa(learned_json_model, num_sequances=10000, max_seq_len=16)
    print(f"Num well-matched tests: {len([x for x in test_dataset if learned_json_model.is_balanced(x[0])])}")

    num_new_sequances = 0

    for seq, label in test_dataset:

        json_string = to_json_string(seq)

        for p in parsers_under_test:
            parser_output = validate_string_with_json_parser(json_string, p)

            if parser_output != label:
                disagreements[p].append(json_string)

            if json_string not in results.keys() or json_string in results.keys() and len(results[json_string]) != len(parsers_under_test):
                results[json_string].append(parser_output)

        add_to_test_set = all(json_string in x for x in disagreements.values())
        if add_to_test_set:
            if (seq, label) in model_learning_dataset:
                model_learning_dataset.remove((seq, label))
            model_learning_dataset.add((seq, not label))
            num_new_sequances += 1

    print(f'Added {num_new_sequances} to learning set, total size {len(model_learning_dataset)}')
    learned_json_model = run_PAPNI(model_learning_dataset, vpa_alphabet, print_info=False)
    print(f'Current model size: {learned_json_model.size}')


comparison_results = {}

for key, values in results.items():
    true_indexes = [parsers_under_test[i] for i, v in enumerate(values) if v]
    false_indexes = [parsers_under_test[i] for i, v in enumerate(values) if not v]

    if true_indexes and false_indexes:  # Ensure there are both True and False values
        comparison_results[key] = [true_indexes, false_indexes]

for test_str, res in comparison_results.items():
    print('---------------------')
    print(test_str)
    print('Postive ', res[0])
    print('Negative', res[1])

#
# for key, val in disagreements.items():
#     print('----------------------------------------------------------')
#     print(key)
#     print(f'Total number of discrepancies: {len(val)}')
#
#     # check which disagreements are not present in other parsers
#     values = set(val)
#     other_values = set()
#     for k, v in disagreements.items():
#         if k != key:
#             other_values.update(v)
#
#     unique = list(values - other_values)
#
#     print(f'Unique discrepancies: {len(unique)}')
#     if unique:
#         print('Printing unique discrepancies')
#         for i in unique:
#             print("".join(i))
