import sys
from itertools import product
from typing import List

from aalpy import load_automaton_from_file, run_PAPNI, run_RPNI
from aalpy.automata import Vpa, VpaState
from aalpy.utils import generate_input_output_data_from_vpa


def get_total_size(obj, seen=None):
    """Recursively find the size of an object and all its referenced objects."""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:  # Avoid processing the same object multiple times
        return 0

    seen.add(obj_id)
    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(get_total_size(k, seen) + get_total_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_total_size(i, seen) for i in obj)
    elif hasattr(obj, '__dict__'):  # For objects with __dict__ attribute
        size += get_total_size(vars(obj), seen)
    elif hasattr(obj, '__slots__'):  # For objects with __slots__
        size += sum(get_total_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))

    return size


def size_in_mb(obj):
    size_bytes = get_total_size(obj)
    return size_bytes / (1024 ** 2)


# TODO, when runni ng these experiments modify RPNI to not delete tree!

#gt = load_automaton_from_file('../DotModels/arithmetics.dot', 'vpa')
from aalpy.utils.BenchmarkVpaModels import get_all_VPAs

for gt in get_all_VPAs():
    e = gt.compute_characterization_set()
    print(e)

exit()

data = generate_input_output_data_from_vpa(gt,
                                           num_sequances=50000,
                                           max_seq_len=50)
vpa_alph = gt.get_input_alphabet()

gt.compute_characterization_set()

exit()

y = run_RPNI(data, automaton_type='dfa', print_info=True)

x = run_PAPNI(data, vpa_alph, print_info=True)

print(get_total_size(y))

print(get_total_size(x))
