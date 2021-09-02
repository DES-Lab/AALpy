from math import sqrt, log


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


class FptaNode:

    def __init__(self, output):
        self.output = output
        self.frequency = 0
        self.children = dict()
        self.prefix = []

    def succs(self):
        queue  = list(self.children.values())
        succs = []
        while queue:
            node = queue.pop()
            succs.append(node)
            for child in node.children.values():
                queue.append(child)
        return succs

    def get_child(self, prefix):
        child = None
        for p in prefix:
            child = self.children[p]
        assert child
        return child


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


class Alergia:
    def __init__(self, data, eps=0.05):
        assert 0 < eps <= 2

        # self.t, self.a = self.create_fpta(data)
        self.t, self.a = get_fptas()
        self.eps = eps

    def create_fpta(self, data):
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

    def compatibility_test(self, a: FptaNode, b: FptaNode):
        if a.output != b.output:
            return False

        if not a.children.values() or not b.children.values():
            return True

        if not HoeffdingCompatibility(a, b, self.eps):
            return False

        for el in a.children.keys():
            if not self.compatibility_test(a.children[el], b.children[el]):
                return False

        return True

    def merge(self, q_r, q_b):
        t_q_b = self.get_blue_node(q_b)
        prefix_leading_to_state = q_b.prefix[:-1]
        to_update = q_r
        for p in prefix_leading_to_state:
            to_update = to_update.children[p]

        to_update.children[q_b.prefix[-1]] = q_r

        paths = [p.prefix for p in self.get_leaf_nodes(t_q_b)]
        paths = [p[len(t_q_b.prefix):] for p in paths]

        for path in paths:
            state_in_a = q_r
            state_in_t = t_q_b
            for p in path:
                state_in_t = state_in_t.children[p]
                state_in_a = state_in_a.children[p]
                state_in_a.frequency += state_in_t.frequency
        print('MERGE ENDED')

    def get_leaf_nodes(self, origin):
        leaves = []

        def _get_leaf_nodes(origin):
            if origin is not None:
                if not origin.children:
                    leaves.append(origin)
                for n in origin.children.values():
                    _get_leaf_nodes(n)

        _get_leaf_nodes(origin)
        return leaves

    def run(self):

        red = {self.a}  # representative nodes and will be included in the final output model
        blue = self.a.succs()  # scheduled for testing

        while blue:
            lex_min_blue = min(list(blue), key=lambda x: len(x.prefix))

            red_sorted = sorted(list(red), key=lambda x: len(x.prefix))

            merged = False

            for q_r in red_sorted:
                if self.compatibility_test(self.get_blue_node(q_r), self.get_blue_node(lex_min_blue)):
                    print('Compatible', q_r.prefix, lex_min_blue.prefix)
                    self.merge(q_r, lex_min_blue)
                    merged = True
                    # break?

            if not merged:
                red.add(lex_min_blue)

            blue.clear()
            all_succs = set()
            for r in red:
                all_succs.update(r.succs())
            print('REEEEEEEEEEEEEEEEEEEEEEEEEEE')
            blue = list({el for el in all_succs if el.prefix not in [r.prefix for r in red]})
            print('BLUE CREATION ENDED')


        print(len(red))
        for r in red:
            print(r.prefix)
        return self.a

    def get_blue_node(self, red_node):
        blue = self.t
        for p in red_node.prefix:
            blue = blue.children[p]
        return blue

def run_Alergia(data, eps):
    alergia = Alergia(data, eps)
    return alergia.run()


if __name__ == '__main__':
    data = [[1, 2, 3, 4], [1, 1, 3, 4], [1, 2, 3, 2]]
    model = run_Alergia(data, 0.5)

    exit(1)
