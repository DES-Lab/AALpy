# Passive automata learning for pushdown automata (PAPNI), by state merging over well-matched words.
import time

from aalpy.automata.Sevpa import Sevpa, SevpaAlphabet, SevpaState, SevpaTransition
from aalpy.automata.Vpa import Vpa, VpaAlphabet, vpa_from_sevpa
from aalpy.utils import is_balanced


def run_PAPNI(data: list, vpa_alphabet: SevpaAlphabet | VpaAlphabet, automaton_type: str = 'vpa',
              print_info: bool = True) -> Vpa | Sevpa | None:
    """
    Run PAPNI, a deterministic passive model learning algorithm of deterministic pushdown automata.
    Resulting model conforms to the provided data.

    Learning is state merging over the congruence of the canonical single entry VPA, which is documented on the
    PAPNI class. The stack symbols of that model carry the state a call was read in, so unlike a learner over an
    alphabet that identifies a stack symbol with the call symbol that pushed it, PAPNI is not restricted to VPAs
    whose pushed stack symbol is determined by the call symbol (see vpa_call_symbol_conflicts).

    :param list data: sequence of input sequences and corresponding label. Eg. [[(i1,i2,i3, ...), label], ...]
    :param SevpaAlphabet | VpaAlphabet vpa_alphabet: grouping of alphabet elements to call symbols, return symbols,
        and internal symbols. Call symbols push to stack, return symbols pop from stack, and internal symbols do not
        affect the stack.
    :param str automaton_type: either 'vpa' for a VPA, or 'sevpa' for the learned 1-SEVPA itself
    :param bool print_info: print learning progress and runtime information
    :return Vpa | Sevpa | None: Model conforming to the data, or None if data is non-deterministic.
    """
    assert automaton_type in {'vpa', 'sevpa'}

    papni = PAPNI(data, vpa_alphabet, print_info)

    if papni.conflicting_data:
        if print_info:
            print('Data provided to PAPNI is not deterministic. Ensure that the data is deterministic, '
                  'or consider using Alergia.')
        return None

    learned_model = papni.run_papni()

    # cast the learned 1-SEVPA to the more general VPA model, in which the call transitions are explicit
    if automaton_type == 'vpa':
        learned_model = vpa_from_sevpa(learned_model, vpa_alphabet)

    return learned_model


class PAPNI:
    """
    State merging over the congruence of the canonical single-entry VPA (Alur, Kumar, Madhusudan and Viswanathan,
    "Congruences for Visibly Pushdown Languages", ICALP 2005), which is the congruence that
    get_characterizing_sequences generates a characteristic sample for.

    A single-entry VPA reads every nesting level starting from the initial state: on a call it pushes the pair
    (current state, call symbol) and resets, and on the matching return it pops that pair and resumes. The state
    reached after a matched block c v r from a state p is therefore determined by p, by c, by the class of the
    well-matched word v, and by r. What is learned is thus the right congruence on well-matched words whose
    extensions are internal symbols and whole matched blocks, rather than a DFA over a stack-annotated alphabet.

    Two consequences distinguish this from learning a DFA over a stack-annotated alphabet. The stack symbol carries
    the state at the call, so VPAs with call symbol conflicts are expressible. And the nesting depth is held in the
    stack rather than in the state, so there are no states at different stack heights that the merging would have to
    keep apart, which a sample of well-matched sequences could never provide evidence for.
    """

    def __init__(self, data: list, alphabet: SevpaAlphabet | VpaAlphabet, print_info: bool = True) -> None:
        """
        Creates a PAPNI instance and constructs the block tree, that is, every well-matched prefix occurring
        in the data at any nesting level together with the extensions between those prefixes.

        :param list data: Sequence of (input sequence, label) pairs.
        :param SevpaAlphabet | VpaAlphabet alphabet: Grouping of alphabet elements to call, return, and internal
            symbols.
        :param bool print_info: Whether to print learning progress and runtime information.
        """
        self.alphabet = alphabet
        self.print_info = print_info

        # union-find over well-matched words, with a trail of parent assignments so that a merge attempt that turns
        # out to be inconsistent with the data can be undone
        self.parent = dict()
        self.trail = []

        self.extensions = dict()
        self.labels = dict()
        self.conflicting_data = False

        # the label of every class that holds a labeled word, maintained as the classes are merged, with its own
        # trail so that it is rolled back with them. Carrying it along keeps a merge from having to rescan the
        # data to find out whether it put an accepted and a rejected word in the same class.
        self.class_label = dict()
        self.label_trail = []
        self.label_conflict = False

        block_tree_construction_start = time.time()
        # the empty word is the initial state of the learned model, so it is a class even if the data is empty
        self.parent[()] = ()
        for input_seq, label in data:
            input_seq = tuple(input_seq)
            # if input sequance is not balanced we do not consider it (it would lead to error state anyway)
            if not is_balanced(input_seq, alphabet):
                continue
            self._collect(input_seq)
            if self.labels.setdefault(input_seq, label) != label:
                self.conflicting_data = True

        # every word is still a class of its own, so it carries its own label
        self.class_label.update(self.labels)

        # length-lexicographic order over the words, with the symbols compared by repr so that an alphabet mixing
        # types stays totally ordered. The rank is what the merging compares, so it is looked up rather than rebuilt
        self.order = sorted(self.parent, key=lambda word: (len(word), tuple(map(repr, word))))
        self.rank = {word: index for index, word in enumerate(self.order)}

        if self.print_info:
            print(f'Block Tree Construction Time: {round(time.time() - block_tree_construction_start, 2)}')

    def run_papni(self) -> Sevpa:
        """
        Runs the state-merging procedure over the block tree and constructs the resulting 1-SEVPA.

        :return Sevpa: The learned 1-SEVPA.
        """
        start_time = time.time()

        # every class is still a singleton, so the block tree is already congruence closed
        extensions = self.extensions
        # the empty word is the lowest ranked, so it is the first class and the initial state
        red = [self.order[0]]
        for word in self.order[1:]:
            # already merged into an earlier class by the congruence closure
            if self._find(word) != word:
                continue

            for red_word in red:
                mark = (len(self.trail), len(self.label_trail))
                self.label_conflict = False
                self._union(word, red_word)
                merge_candidate = self._close(extensions)
                if self._compatible():
                    extensions = merge_candidate
                    # nothing below this point can be rolled back any more, the merge is part of the model
                    self.trail.clear()
                    self.label_trail.clear()
                    break
                self._undo(mark)
            else:
                red.append(word)
                if self.print_info:
                    print(f'\rCurrent automaton size: {len(red)}', end="")

        learned_model = self._to_sevpa(extensions)

        if self.print_info:
            print(f'\nPAPNI Learning Time: {round(time.time() - start_time, 2)}')
            print(f'PAPNI Learned {len(learned_model.states)} state 1-SEVPA.')

        return learned_model

    def _split_items(self, word: tuple) -> list:
        """
        Decomposes a well-matched word into its top level items, which are either an internal symbol, encoded as
        ('int', symbol), or a matched block, encoded as ('blk', call symbol, enclosed word, return symbol).

        :param tuple word: Well-matched word to decompose.
        :return list: The top level items of the word, in order.
        """
        items, index = [], 0
        while index < len(word):
            symbol = word[index]
            if symbol not in self.alphabet.call_alphabet:
                items.append(('int', symbol))
                index += 1
                continue

            # scan forward to the return symbol matching the call symbol at index
            depth, end = 1, index + 1
            while depth:
                if word[end] in self.alphabet.call_alphabet:
                    depth += 1
                elif word[end] in self.alphabet.return_alphabet:
                    depth -= 1
                end += 1

            items.append(('blk', symbol, word[index + 1:end - 1], word[end - 1]))
            index = end
        return items

    def _collect(self, word: tuple) -> None:
        """
        Registers every well-matched prefix of a word, and of every block enclosed in it, together with the
        extension leading from one prefix to the next. A registered word is always fully decomposed, so a word that
        is already known needs no further work.

        The enclosed blocks are worked off through an explicit worklist rather than by recursing, so that deeply
        nested data does not exhaust the interpreter stack.

        :param tuple word: Well-matched word to register.
        """
        pending = [word]
        while pending:
            word = pending.pop()
            if word in self.parent:
                continue
            self.parent[word] = word

            prefix = ()
            for item in self._split_items(word):
                if item[0] == 'int':
                    following = prefix + (item[1],)
                else:
                    following = prefix + (item[1],) + item[2] + (item[3],)
                    pending.append(item[2])
                self.parent.setdefault(following, following)
                self.extensions[(prefix, item)] = following
                prefix = following

    def _find(self, word: tuple) -> tuple:
        """
        Looks up the representative of the class a word currently belongs to, compressing the path to it.

        :param tuple word: Word whose class is looked up.
        :return tuple: Representative of the class, which is its lowest ranked member.
        """
        root = word
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[word] != root:
            following = self.parent[word]
            self._reassign(word, root)
            word = following
        return root

    def _union(self, first: tuple, second: tuple) -> bool:
        """
        Merges the classes of two words, keeping the lower ranked representative.

        :param tuple first: A word of the first class.
        :param tuple second: A word of the second class.
        :return bool: True if the two words were in different classes, False if nothing changed.
        """
        first, second = self._find(first), self._find(second)
        if first == second:
            return False
        if self.rank[second] < self.rank[first]:
            first, second = second, first
        self._reassign(second, first)

        # the absorbed class hands its label to the kept one, which is where a conflict with the data shows up
        if second in self.class_label:
            label = self.class_label[second]
            if first not in self.class_label:
                self.label_trail.append((first, False, None))
                self.class_label[first] = label
            elif self.class_label[first] != label:
                self.label_conflict = True
        return True

    def _reassign(self, word: tuple, root: tuple) -> None:
        """
        Points a word at a new representative, recording the previous one so that the change can be undone. The
        path compression in _find goes through here too, so rolling back also undoes compression, which is harmless.

        :param tuple word: Word to reassign.
        :param tuple root: New representative of the word.
        """
        self.trail.append((word, self.parent[word]))
        self.parent[word] = root

    def _undo(self, mark: tuple) -> None:
        """
        Rolls the union-find and the class labels back to the state they were in when the trails had the given
        lengths.

        :param tuple mark: Lengths of the parent trail and of the label trail to roll back to.
        """
        parent_mark, label_mark = mark
        for word, previous in reversed(self.trail[parent_mark:]):
            self.parent[word] = previous
        del self.trail[parent_mark:]

        for root, had_label, previous in reversed(self.label_trail[label_mark:]):
            if had_label:
                self.class_label[root] = previous
            else:
                del self.class_label[root]
        del self.label_trail[label_mark:]

    def _item_key(self, item: tuple) -> tuple:
        """
        Computes the extension a block tree item currently stands for. Two blocks are the same extension once the
        words they enclose are in the same class, so the key of a block item follows the merging.

        :param tuple item: Item to compute the key of.
        :return tuple: Key of the item under the current classes.
        """
        return item if item[0] == 'int' else ('blk', item[1], self._find(item[2]), item[3])

    def _close(self, extensions: dict) -> dict:
        """
        Computes the congruence closure of the current classes, that is, merges the targets of every two extensions
        that leave the same class with the same key, until nothing changes. Every pass works on the quotient
        produced by the previous one, which shrinks as classes are merged.

        :param dict extensions: Map from (source word, item) to the word the extension leads to.
        :return dict: The same map, quotiented by the closed congruence.
        """
        while True:
            closed, changed = dict(), False
            for (source, item), target in extensions.items():
                key = (self._find(source), self._item_key(item))
                if key in closed:
                    changed |= self._union(closed[key], target)
                else:
                    closed[key] = target
            extensions = closed
            # the merge is already known to contradict the data, so closing it up further is wasted work
            if not changed or self.label_conflict:
                return closed

    def _compatible(self) -> bool:
        """
        Check if current classes are compatible with the data, that is, no class holds both an accepted and a
        rejected sequence.

        Classes only ever grow, so a class holding two different labels is exactly a class that took in a label
        differing from the one it already had. _union notices that as it happens, which is why the answer is read
        off a flag here rather than by going over the data again.

        :return bool: True if the classes are compatible with all labeled sequences, False otherwise.
        """
        return not self.label_conflict

    def _to_sevpa(self, extensions: dict) -> Sevpa:
        """
        Constructs a 1-SEVPA from the learned classes. Call transitions are implicit in a 1-SEVPA, which always
        pushes the pair (state the call is read in, call symbol) and continues from the initial state. An extension
        by a block, leading from a class p to a class t through an enclosed word of class s, becomes the transition
        popping (p, call symbol) in s and leading to t.

        :param dict extensions: Congruence closed map from (source word, item) to the word the extension leads to.
        :return Sevpa: The constructed 1-SEVPA.
        """
        accepting = {self._find(word) for word, label in self.labels.items() if label}
        roots = sorted({self._find(word) for word in self.parent}, key=lambda word: self.rank[word])
        states = {root: SevpaState(state_id=f'q{index}', is_accepting=root in accepting)
                  for index, root in enumerate(roots)}

        for (source, item), target in extensions.items():
            origin_state, reached_state = states[self._find(source)], states[self._find(target)]
            if item[0] == 'int':
                origin_state.transitions[item[1]].append(
                    SevpaTransition(reached_state, item[1], None))
            else:
                _, call_symbol, enclosed, return_symbol = item
                enclosed_state = states[self._find(enclosed)]
                enclosed_state.transitions[return_symbol].append(
                    SevpaTransition(reached_state, return_symbol, 'pop', (origin_state.state_id, call_symbol)))

        # the alphabet is passed in rather than recovered from the transitions, which would lose the symbols that
        # no transition of the learned model happens to use
        return Sevpa(states[self._find(())], list(states.values()),
                     SevpaAlphabet(list(self.alphabet.internal_alphabet), list(self.alphabet.call_alphabet),
                                   list(self.alphabet.return_alphabet)))
