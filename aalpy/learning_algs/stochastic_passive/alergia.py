from math import sqrt, log


class FptaNode:

    def __init__(self, output):
        self.output = output
        self.frequency = 0
        self.children = dict()
        self.prefix = []


def create_fpta(data):
    root_node = FptaNode(data[0][0])
    for seq in data:
        if seq[0] != root_node.output:
            print('All strings should have the same initial output')
            assert False
        curr_node = root_node

        for el in seq[1:]:
            if el not in curr_node.children.keys():
                node = FptaNode(el)
                node.prefix = list(curr_node.prefix)
                node.prefix.append(el)
                curr_node.children[el] = node

            curr_node = curr_node.children[el]
            curr_node.frequency += 1

    return root_node


def HoeffdingCompatibility(a: FptaNode, b: FptaNode, eps):
    if a.children.keys() != b.children.keys():
        return False

    n1 = sum([child.frequency for child in a.children.values()])
    n2 = sum([child.frequency for child in b.children.values()])

    if n1 > 0 and n2 > 0:
        for o in a.children.keys():
            if abs(a.children[o].frequency / n1 - b.children[o].frequency / n2) > \
                    ((sqrt(1 / n1) + sqrt(1 / n2)) * sqrt(0.5 * log(2 / eps))):
                return False
    return True


def compatibility_test(a: FptaNode, b: FptaNode, eps):
    if a.output != b.output:
        return False

    if not a.children.values() or not b.children.values():
        return True

    if not HoeffdingCompatibility(a, b, eps):
        return False

    for el in a.children.keys():
        if not compatibility_test(a.children[el], b.children[el], eps):
            return False

    return True


def get_paths(t, paths=None, current_path=None):
    if paths is None:
        paths = []
    if current_path is None:
        current_path = []

    if not t.children:
        paths.append(current_path)
    else:
        for inp, child in t.children.items():
            current_path.append(inp)
            get_paths(child, paths, list(current_path))
    return paths


def merge(a, t, q_r, q_b):
    q_r_a, q_r_b = a, t
    for i in q_r.prefix:
        q_r_a = q_r_a.children[i]
    prefix_to_b = t
    for i in q_b.prefix[:-1]:
        prefix_to_b = q_r_b.children[i]

    q_r_b = prefix_to_b.children[q_b.prefix[-1]]
    prefix_to_b.children[q_b.prefix[-1]] = q_r_a

    paths_to_update = get_paths(q_r_b)

    for path in paths_to_update:
        curr_node_r = q_r_a
        curr_node_b = q_r_b
        for i in path:
            curr_node_r = q_r_a.children[i]
            curr_node_b = q_r_b.children[i]
            curr_node_r.frequency += curr_node_b.frequency


def run_alergia(data, eps):
    assert 0 < eps <= 2

    # FPTA T is kept as a data representation from which relevant statistics are retrieved during the execution of
    # the algorithm
    t = create_fpta(data)
    # The FPTA A is iteratively transformed by merging nodes that have passed a statistical compatibility test
    a = create_fpta(data)

    t, a = get_fptas()

    red = {t}  # representative nodes and will be included in the final output model # TODO FIX
    blue = set()  # scheduled for testing
    for c in t.children.values():
        blue.add(c)

    while blue:
        lex_min_blue = min(list(blue), key=lambda x: len(x.prefix))

        red_sorted = sorted(list(red), key=lambda x: len(x.prefix))

        merged = False

        for q_r in red_sorted:
            if compatibility_test(q_r, lex_min_blue, eps):
                print('Compatible', q_r.prefix, lex_min_blue.prefix)
                merge(a, t, q_r, lex_min_blue)
                merged = True
                # break?

        if not merged:
            red.add(lex_min_blue)

        blue.clear()
        for r in red:
            for child in r.children.values():
                if child.prefix not in [c.prefix for c in red]:
                    blue.add(child)

    print(len(red))
    for r in red:
        print(r.prefix)
    return a


def get_fptas():
    root = FptaNode('A')
    root.children['B'] = FptaNode('B')
    root.children['B'].frequency = 15
    root.children['B'].prefix = ['B']
    root.children['A'] = FptaNode('A')
    root.children['A'].frequency = 7
    root.children['A'].prefix = ['A']

    root.children['B'].children['A'] = FptaNode('A')
    root.children['B'].children['A'].frequency = 14
    root.children['B'].children['A'].prefix = ['B', 'A']

    root.children['B'].children['A'].children['B'] = FptaNode('B')
    root.children['B'].children['A'].children['B'].frequency = 10
    root.children['B'].children['A'].children['B'].prefix = ['B', 'A', 'B']
    root.children['B'].children['A'].children['A'] = FptaNode('A')
    root.children['B'].children['A'].children['A'].frequency = 4
    root.children['B'].children['A'].children['A'].prefix = ['B', 'A', 'A']
    root.children['B'].children['A'].children['B'].children['A'] = FptaNode('A')
    root.children['B'].children['A'].children['B'].children['A'].frequency = 8
    root.children['B'].children['A'].children['B'].children['A'].prefix = ['B', 'A', 'B', 'A']

    root.children['B'].children['A'].children['A'].children['A'] = FptaNode('A')
    root.children['B'].children['A'].children['A'].children['A'].frequency = 3
    root.children['B'].children['A'].children['A'].children['A'].prefix = ['B', 'A', 'A', 'A']

    root.children['A'].children['A'] = FptaNode('A')
    root.children['A'].children['A'].frequency = 5
    root.children['A'].children['A'].prefix = ['A', 'A']
    root.children['A'].children['A'].children['A'] = FptaNode('A')
    root.children['A'].children['A'].children['A'].frequency = 5
    root.children['A'].children['A'].children['A'].prefix = ['A', 'A', 'A']

    root1 = FptaNode('A')
    root1.children['B'] = FptaNode('B')
    root1.children['B'].frequency = 15
    root1.children['B'].prefix = ['B']
    root1.children['A'] = FptaNode('A')
    root1.children['A'].frequency = 7
    root1.children['A'].prefix = ['A']

    root1.children['B'].children['A'] = FptaNode('A')
    root1.children['B'].children['A'].frequency = 14
    root1.children['B'].children['A'].prefix = ['B', 'A']

    root1.children['B'].children['A'].children['B'] = FptaNode('B')
    root1.children['B'].children['A'].children['B'].frequency = 10
    root1.children['B'].children['A'].children['B'].prefix = ['B', 'A', 'B']
    root1.children['B'].children['A'].children['A'] = FptaNode('A')
    root1.children['B'].children['A'].children['A'].frequency = 4
    root1.children['B'].children['A'].children['A'].prefix = ['B', 'A', 'A']
    root1.children['B'].children['A'].children['B'].children['A'] = FptaNode('A')
    root1.children['B'].children['A'].children['B'].children['A'].frequency = 8
    root1.children['B'].children['A'].children['B'].children['A'].prefix = ['B', 'A', 'B', 'A']

    root1.children['B'].children['A'].children['A'].children['A'] = FptaNode('A')
    root1.children['B'].children['A'].children['A'].children['A'].frequency = 3
    root1.children['B'].children['A'].children['A'].children['A'].prefix = ['B', 'A', 'A', 'A']

    root1.children['A'].children['A'] = FptaNode('A')
    root1.children['A'].children['A'].frequency = 5
    root1.children['A'].children['A'].prefix = ['A', 'A']
    root1.children['A'].children['A'].children['A'] = FptaNode('A')
    root1.children['A'].children['A'].children['A'].frequency = 5
    root1.children['A'].children['A'].children['A'].prefix = ['A', 'A', 'A']

    return root, root1


if __name__ == '__main__':
    data = [[1, 2, 3, 4], [1, 1, 3, 4], [1, 2, 3, 2]]
    model = run_alergia(data, 0.5)


    exit(1)