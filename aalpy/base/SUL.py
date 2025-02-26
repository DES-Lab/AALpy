from abc import ABC, abstractmethod

from aalpy.base.CacheTree import CacheTree, CacheDict


class SUL(ABC):
    """
    System Under Learning (SUL) abstract class. Defines the interaction between the learning algorithm and the system
    under learning. All systems under learning have to implement this class, as it is
    passed to the learning algorithm and the equivalence oracle.
    """

    def __init__(self):
        self.num_queries = 0
        self.num_steps = 0
        self.num_cached_queries = 0

    def query(self, word: tuple) -> list:
        """
        Performs a membership query on the SUL. Before the query, pre() method is called and after the query post()
        method is called. Each letter in the word (input in the input sequence) is executed using the step method.

        Args:

            word: membership query (word consisting of letters/inputs)

        Returns:

            list of outputs, where the i-th output corresponds to the output of the system after the i-th input

        """
        self.pre()
        # Empty string for DFA
        if len(word) == 0:
            out = [self.step(None)]
        else:
            out = [self.step(letter) for letter in word]
        self.post()
        self.num_queries += 1
        self.num_steps += len(word)
        return out

    def io_query(self, word : tuple):
        return list(zip(word, self.query(word)))

    def adaptive_query(self, word, ads):
        """

        Performs an adaptive output query on the SUL. Before the query, pre() method is called and after the query post()
        method is called. The ADS is a tree like object, the next input depends on the previous input-output pairs. Each input is executed using the step method. Currently only implemented for Mealy machines

        Args:

            word: membership query (word consisting of letters/inputs)

            ads: adaptive distinguishing suffix

        Returns:

            list of outputs, where the i-th output corresponds to the output of the system after the i-th input
        """
        self.pre()

        outputs_received = []
        last_output = None

        for inp in word:
            output = self.step(inp)
            outputs_received.append(output)

        self.num_steps += len(word)

        while True:
            next_input = ads.next_input(last_output)
            if next_input is None:
                break
            if next_input is tuple(): # Relevant for DFA/Moore
                last_output = self.step(None)
            else:
                word.append(next_input)
                output = self.step(next_input) 
                outputs_received.append(output)
                last_output = output
                self.num_steps += 1

        self.num_queries += 1
        self.post()

        return word, outputs_received

    @abstractmethod
    def pre(self):
        """
        Resets the system. Called after post method in the equivalence query.
        """
        pass

    @abstractmethod
    def post(self):
        """
        Performs additional cleanup on the system in necessary. Called before pre method in the equivalence query.
        """
        pass

    @abstractmethod
    def step(self, letter):
        """
        Executes an action on the system under learning and returns its result.

        Args:

            letter: Single input that is executed on the SUL.

        Returns:

            Output received after executing the input.

        """
        pass


class CacheSUL(SUL):
    """
    System under learning that keeps a multiset of all queries in memory.
    This multiset/cache is encoded as a tree.
    """

    def __init__(self, sul: SUL, cache_type='tree'):
        super().__init__()
        self.sul = sul
        self.cache = CacheTree() if cache_type == 'tree' else CacheDict()

    def query(self, word):
        """
        Performs a membership query on the SUL if and only if `word` is not a prefix of any trace in the cache.
        Before the query, pre() method is called and after the query post()
        method is called. Each letter in the word (input in the input sequence) is executed using the step method.

        Args:

            word: membership query (word consisting of letters/inputs)

        Returns:

            list of outputs, where the i-th output corresponds to the output of the system after the i-th input

        """
        cached_query = self.cache.in_cache(word)
        if cached_query:
            self.num_cached_queries += 1
            return cached_query

        # get outputs using default query method
        out = self.sul.query(word)

        # add input/outputs to tree
        self.cache.reset()
        for i, o in zip(word, out):
            self.cache.step_in_cache(i, o)

        self.num_queries += 1
        self.num_steps += len(word)
        return out

    def pre(self):
        """
        Reset the system under learning and current node in the cache tree.
        """
        self.cache.reset()
        self.sul.pre()

    def post(self):
        self.sul.post()

    def step(self, letter):
        """
        Executes an action on the system under learning, adds it to the cache and returns its result.

        Args:

           letter: Single input that is executed on the SUL.

        Returns:

           Output received after executing the input.

        """
        out = self.sul.step(letter)
        self.cache.step_in_cache(letter, out)
        return out
