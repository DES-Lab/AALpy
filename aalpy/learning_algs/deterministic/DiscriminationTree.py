from aalpy.base import SUL


class DTNode:
    def __init__(self, parent, path_to_node):
        self.parent = parent
        self.path_to_node = path_to_node

    @property
    def root(self):
        p = self.parent
        while p:
            parent = p.parent
        return p


class DTDiscriminatorNode(DTNode):
    def __init__(self, discriminator: tuple, parent: DTNode, path_to_node):
        super().__init__(parent, path_to_node)
        self.true_child = None
        self.false_child = None
        self.discriminator = discriminator

    def __repr__(self):
        return f'{self.__class__.__name__} \'{self.discriminator}\''


class DTStateNode(DTNode):
    def __init__(self, parent: DTNode, path_to_node, prefix: tuple):
        super().__init__(parent, path_to_node)
        self.hyp_state = None
        self.prefix = prefix

    def __repr__(self):
        if self.hyp_state:
            return f'{self.__class__.__name__} \'{self.hyp_state.state_id}\' [prefix: \'{self.prefix}\']'
        else:
            return f'{self.__class__.__name__} (unlinked)'

class DiscriminationTree:
    def __init__(self, alphabet: list, sul: SUL):
        self.root = DTDiscriminatorNode(discriminator=(None,), parent=None, path_to_node="root")
        self.alphabet = alphabet
        self.sul = sul

    def sift(self, word):
        '''
        Sifting a word into the discrimination tree.
        Starting at the root, at every inner node (a DTDiscriminationNode),
        we branch into the "true" or "false" child, depending on the result of the
        membership query (word * node.discriminator). Repeated until a leaf
        (a DTStateNode) is reached, which is the result of the sifting.

        Args:

            word: the word to sift into the discrimination tree (a tuple of all letters)

        Returns:

            the DTStateNode that is reached by the sifting operation.
            if no node exists at this leaf yet, it will be created.
        '''
        for letter in word:
            assert letter is None or letter in self.alphabet

        node = self.root

        while isinstance(node, DTDiscriminatorNode):
            mq_result = self.sul.query((*word, *node.discriminator))[-1]
            if mq_result is True:
                if node.true_child is None:
                    node.true_child = DTStateNode(node, path_to_node="true", prefix=word)
                node = node.true_child
            elif mq_result is False:
                if node.false_child is None:
                    node.false_child = DTStateNode(node, path_to_node="false", prefix=word)
                node = node.false_child
            else:
                assert False

        assert isinstance(node, DTStateNode)
        return node


