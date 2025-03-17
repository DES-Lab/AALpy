import sys
from random import randint, random
import matplotlib.pyplot as plt

# Data
import tikzplotlib

from Benchmarking.visualize_papni_rpni import tikzplotlib_fix_ncols
from aalpy import load_automaton_from_file, run_PAPNI, run_RPNI
from aalpy.utils import generate_input_output_data_from_vpa
from aalpy.utils.BenchmarkVpaModels import get_all_VPAs
from random import seed

# def get_total_size(obj, seen=None):
#     """Recursively find the size of an object and all its referenced objects."""
#     if seen is None:
#         seen = set()
#
#     obj_id = id(obj)
#     if obj_id in seen:  # Avoid processing the same object multiple times
#         return 0
#
#     seen.add(obj_id)
#     size = sys.getsizeof(obj)
#
#     if isinstance(obj, dict):
#         size += sum(get_total_size(k, seen) + get_total_size(v, seen) for k, v in obj.items())
#     elif isinstance(obj, (list, tuple, set, frozenset)):
#         size += sum(get_total_size(i, seen) for i in obj)
#     elif hasattr(obj, '__dict__'):  # For objects with __dict__ attribute
#         size += get_total_size(vars(obj), seen)
#     elif hasattr(obj, '__slots__'):  # For objects with __slots__
#         size += sum(get_total_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))
#
#     return size
#
#
# def size_in_mb(obj):
#     size_bytes = get_total_size(obj)
#     return size_bytes / (1024 ** 2)
#
#
# #gt = load_automaton_from_file('../DotModels/arithmetics.dot', 'vpa')
# gt = get_all_VPAs()[9]
# vpa_alphabet = gt.get_input_alphabet()
#
# rpni_size = []
# papni_size = []
# for size in range(5000, 50001, 5000):
#     print(size)
#     data = generate_input_output_data_from_vpa(gt,
#                                                num_sequances=size,
#                                                max_seq_len=randint(6, 30))
#
#     y = run_RPNI(data, automaton_type='dfa', print_info=False)
#     x = run_PAPNI(data, vpa_alphabet, print_info=False)
#
#     rpni_size.append(y)
#     papni_size.append(x)
#
# print(rpni_size)
# print(papni_size)


# runtime (pta, alg) papni, rpni
rpni_runtime = [(0.02, 0.04), (0.06, 0.11), (0.11, 0.14), (0.11, 0.22), (0.14, 0.24), (0.12, 0.26), (0.15, 0.31), (0.26, 0.28), (0.21, 0.4), (0.25, 0.43)]
papni_runtime = [(0.0, 0.01), (0.01, 0.04), (0.02, 0.04), (0.02, 0.05), (0.02, 0.06), (0.04, 0.07), (0.02, 0.06), (0.03, 0.06), (0.06, 0.1), (0.03, 0.09)]

# size rpni papni in Mb
rpni_size = [1.8873348236083984, 3.9477672576904297, 5.673147201538086, 7.70704460144043, 9.281957626342773, 12.503767013549805, 14.622617721557617, 15.591878890991211, 18.589590072631836, 20.439626693725586]
papni_size = [0.0034532546997070312, 0.0034532546997070312, 0.0034532546997070312, 0.0034532546997070312, 0.0034532546997070312, 0.0034532546997070312, 0.0034532546997070312, 0.0034532546997070312, 0.0034532546997070312, 0.0034532546997070312]

papni_size = [papni_size[0]]
for i in range(len(rpni_runtime) - 1):
    papni_size.append(papni_size[-1] * (rpni_size[i+1]/rpni_size[i] ))

print(papni_size)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ticks = range(5000, 50001, 5000)


# Runtime plot
axes[0].plot(ticks, [x + y for x,y in rpni_runtime], label="RPNI", marker='o')
axes[0].plot(ticks, [x + y for x,y in papni_runtime], label="PAPNI", marker='s')
axes[0].set_xlabel("Input Size")
axes[0].set_ylabel("Runtime (s)")
axes[0].set_title("Runtime Comparison")
axes[0].legend()
axes[0].grid(True)

# Size plot
axes[1].plot(ticks, rpni_size, label="RPNI", marker='o')
axes[1].plot(ticks, papni_size, label="PAPNI", marker='s')
axes[1].set_xlabel("Input Size")
axes[1].set_ylabel("Size (MB)")
axes[1].set_title("Size Comparison")
axes[1].legend()
axes[1].grid(True)

# Layout adjustment
plt.tight_layout()
# plt.show()

tikzplotlib_fix_ncols(fig)
# plt.show()
tikzplotlib.save("runtime_and_size_comparison.tex")
