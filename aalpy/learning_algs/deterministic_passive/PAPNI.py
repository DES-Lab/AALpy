# Passive automata learning for pushdown automata (PAPNI), by state merging over well-matched words.
import time
from collections import defaultdict

from aalpy.automata.Sevpa import Sevpa, SevpaAlphabet, SevpaState, SevpaTransition
from aalpy.automata.Vpa import Vpa, VpaAlphabet, vpa_from_sevpa
from aalpy.utils import is_balanced

# marks a transition that was not there before a merge, so that the rollback removes it instead of restoring one
_MISSING = object()


def run_PAPNI(data: list, vpa_alphabet: SevpaAlphabet | VpaAlphabet, automaton_type: str = 'vpa',
              print_info: bool = True, algorithm: str = 'edsm') -> Vpa | Sevpa | None:
    """
    Run PAPNI, a deterministic passive model learning algorithm of deterministic pushdown automata.
    By construction, learned model conforms to the provided data. The learned model is a 1-SEVPA, see the PAPNI
    class for details.

    :param list data: sequence of input sequences and corresponding label. Eg. [[(i1,i2,i3, ...), label], ...]
    :param SevpaAlphabet | VpaAlphabet vpa_alphabet: grouping of alphabet elements to call symbols, return symbols,
        and internal symbols. Call symbols push to stack, return symbols pop from stack, and internal symbols do not
        affect the stack.
    :param str automaton_type: either 'vpa' for a VPA, or 'sevpa' for the learned 1-SEVPA itself
    :param bool print_info: print learning progress and runtime information
    :param str algorithm: either 'gsm' (or its alias 'classic') for classic RPNI-style greedy merging, which merges a
        class with the first compatible state, or 'edsm' for evidence driven state merging, which performs the merge
        supported by the most evidence in the data
    :return Vpa | Sevpa | None: Model conforming to the data, or None if data is non-deterministic.
    """
    assert automaton_type in {'vpa', 'sevpa'}
    assert algorithm in {'gsm', 'classic', 'edsm'}

    papni = PAPNI(data, vpa_alphabet, print_info)

    if papni.conflicting_data:
        if print_info:
            print('Data provided to PAPNI is not deterministic. Ensure that the data is deterministic, '
                  'or consider using Alergia.')
        return None

    learned_model = papni.run_papni() if algorithm != 'edsm' else papni.run_edsm()

    # cast the learned 1-SEVPA to the more general VPA model, in which the call transitions are explicit
    if automaton_type == 'vpa':
        learned_model = vpa_from_sevpa(learned_model)

    return learned_model


class PAPNI:
    """
    State merging over the congruence of the canonical single-entry VPA (Alur, Kumar, Madhusudan and Viswanathan,
    "Congruences for Visibly Pushdown Languages", ICALP 2005).

    A single-entry VPA reads every nesting level starting from the initial state: on a call it pushes the pair
    (current state, call symbol) and resets, and on the matching return it pops that pair and resumes. The state
    reached after a matched block c v r from a state p is therefore determined by p, by c, by the class of the
    well-matched word v, and by r. What is learned is thus the right congruence on well-matched words whose
    extensions are internal symbols and whole matched blocks, rather than a DFA over a stack-annotated alphabet.

    As the stack symbol carries the state at the call, PAPNI is not restricted to VPAs whose pushed stack symbol is
    determined by the call symbol. And as the nesting depth is held in the stack rather than in the state, there are
    no states at different stack heights to keep apart, which well-matched data could never provide evidence for.
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
        self.call_symbols = set(alphabet.call_alphabet)
        self.return_symbols = set(alphabet.return_alphabet)

        # union-find over well-matched words, with a trail of parent assignments so that a merge attempt that turns
        # out to be inconsistent with the data can be undone
        self.parent = dict()
        self.trail = []

        # the quotient automaton: transitions[c][key] is the word reached by extending class c, keyed by
        # (internal symbol,) or by (call symbol, class of the enclosed word, return symbol). Only the entry of a
        # class representative is up to date, the entry of an absorbed class is left behind for the rollback.
        self.transitions = defaultdict(dict)
        self.transition_trail = []

        # enclosing[c] holds the (word, key) pairs whose key is a block enclosing class c, so that a merge finds the
        # keys it invalidates. It is never rolled back, so it may hold stale pairs, which _merge skips.
        self.enclosing = defaultdict(set)

        self.conflicting_data = False

        # the label of every class that holds a labeled word, maintained as the classes are merged, with its own
        # trail so that it is rolled back with them. Carrying it along keeps a merge from having to rescan the
        # data to find out whether it put an accepted and a rejected word in the same class.
        self.class_label = dict()
        self.label_trail = []
        self.label_conflict = False
        # the evidence of the current merge, that is, the number of times two labeled classes were merged into one
        self.merge_score = 0
        # representatives of the red classes of run_edsm, which a merge keeps as representatives
        self.red = set()

        block_tree_construction_start = time.time()
        # the empty word is the initial state of the learned model, so it is a class even if the data is empty
        self.parent[()] = ()
        for input_seq, label in data:
            input_seq = tuple(input_seq)
            # if input sequance is not balanced we do not consider it (it would lead to error state anyway)
            if not is_balanced(input_seq, alphabet):
                continue
            self._collect(input_seq)
            # every word is still a class of its own, so it carries its own label
            if self.class_label.setdefault(input_seq, label) != label:
                self.conflicting_data = True

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

        # the empty word is the lowest ranked, so it is the first class and the initial state
        red = [self.order[0]]
        for word in self.order[1:]:
            # already merged into an earlier class while the congruence was restored
            if self._find(word) != word:
                continue

            for red_word in red:
                if self._merge(word, red_word):
                    self._commit()
                    break
                self._undo()
            else:
                red.append(word)
                if self.print_info:
                    print(f'\rCurrent automaton size: {len(red)}', end="")

        return self._finish(start_time)

    def run_edsm(self) -> Sevpa:
        """
        Runs evidence driven state merging over the block tree and constructs the resulting 1-SEVPA.

        The blue classes are the ones reached by a single transition from a red class, whose key encloses a red class
        if it is a block. Every blue class is tried against every red one, and a merge is scored by the number of
        times it puts two labeled classes together, as run_EDSM does. A blue class that merges with no red one is
        promoted, otherwise the best scoring merge over all blue and red classes is performed.

        :return Sevpa: The learned 1-SEVPA.
        """
        start_time = time.time()

        self.red = {self.order[0]}
        while True:
            # a merge may have put two red classes together, which leaves one representative of them
            self.red = {self._find(word) for word in self.red}
            red = sorted(self.red, key=self.rank.get)
            blue = self._blue()
            if not blue:
                break

            best_merge, promoted = None, False
            for blue_word in blue:
                best_for_blue = None
                for red_word in red:
                    if self._merge(blue_word, red_word) and \
                            (best_for_blue is None or self.merge_score > best_for_blue[0]):
                        best_for_blue = (self.merge_score, blue_word, red_word)
                    self._undo()

                if best_for_blue is None:
                    self.red.add(blue_word)
                    promoted = True
                    if self.print_info:
                        print(f'\rCurrent automaton size: {len(self.red)}', end="")
                    break
                if best_merge is None or best_for_blue[0] > best_merge[0]:
                    best_merge = best_for_blue

            if not promoted:
                self._merge(best_merge[1], best_merge[2])
                self._commit()

        return self._finish(start_time)

    def _blue(self) -> list:
        """
        Computes the blue classes, that is, the classes that are not red and are reached by a single transition from a
        red class, where a block key must enclose a red class too.

        :return list: Representatives of the blue classes, in length-lexicographic order.
        """
        blue = set()
        for root in self.red:
            for key, target in self.transitions[root].items():
                if len(key) == 3 and key[1] not in self.red:
                    continue
                target = self._find(target)
                if target not in self.red:
                    blue.add(target)
        return sorted(blue, key=self.rank.get)

    def _finish(self, start_time: float) -> Sevpa:
        """
        Constructs the learned 1-SEVPA and reports the runtime.

        :param float start_time: Time at which the merging started.
        :return Sevpa: The learned 1-SEVPA.
        """
        learned_model = self._to_sevpa()

        if self.print_info:
            print(f'\nPAPNI Learning Time: {round(time.time() - start_time, 2)}')
            print(f'PAPNI Learned {len(learned_model.states)} state 1-SEVPA.')

        return learned_model

    def _items(self, word: tuple) -> list:
        """
        Decomposes a well-matched word into its top level items, which are either an internal symbol, encoded as
        (symbol,), or a matched block, encoded as (call symbol, enclosed word, return symbol).

        :param tuple word: Well-matched word to decompose.
        :return list: The top level items of the word, in order.
        """
        items, index = [], 0
        while index < len(word):
            symbol = word[index]
            if symbol not in self.call_symbols:
                items.append((symbol,))
                index += 1
                continue

            # scan forward to the return symbol matching the call symbol at index
            depth, end = 1, index + 1
            while depth:
                if word[end] in self.call_symbols:
                    depth += 1
                elif word[end] in self.return_symbols:
                    depth -= 1
                end += 1

            items.append((symbol, word[index + 1:end - 1], word[end - 1]))
            index = end
        return items

    def _collect(self, word: tuple) -> None:
        """
        Registers every well-matched prefix of a word, and of every block enclosed in it, together with the
        transition leading from one prefix to the next. A worklist is used instead of recursion, so that deeply
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
            for item in self._items(word):
                if len(item) == 1:
                    following = prefix + item
                else:
                    following = prefix + (item[0],) + item[1] + (item[2],)
                    pending.append(item[1])
                    self.enclosing[item[1]].add((prefix, item))
                self.parent.setdefault(following, following)
                self.transitions[prefix][item] = following
                prefix = following

    def _find(self, word: tuple) -> tuple:
        """
        Looks up the representative of the class a word currently belongs to, compressing the path to it.

        :param tuple word: Word whose class is looked up.
        :return tuple: Representative of the class, which is its lowest ranked member, unless run_edsm keeps the
            representative of a red class instead.
        """
        parent = self.parent
        root = word
        while parent[root] != root:
            root = parent[root]
        while parent[word] != root:
            following = parent[word]
            self._reassign(word, root)
            word = following
        return root

    def _union(self, first: tuple, second: tuple) -> tuple | None:
        """
        Merges the classes of two words, keeping the representative of a red class, or else the lower ranked one.

        :param tuple first: A word of the first class.
        :param tuple second: A word of the second class.
        :return tuple | None: The kept and the absorbed representative, or None if both words were already in the
            same class.
        """
        first, second = self._find(first), self._find(second)
        if first == second:
            return None
        if (second in self.red, self.rank[first]) > (first in self.red, self.rank[second]):
            first, second = second, first
        self._reassign(second, first)

        # the absorbed class hands its label to the kept one, which is where a conflict with the data shows up
        if second in self.class_label:
            label = self.class_label[second]
            if first not in self.class_label:
                self.label_trail.append(first)
                self.class_label[first] = label
            elif self.class_label[first] != label:
                self.label_conflict = True
            else:
                self.merge_score += 1
        return first, second

    def _reassign(self, word: tuple, root: tuple) -> None:
        """
        Points a word at a new representative, recording the previous one so that the change can be undone. The
        path compression in _find goes through here too, so rolling back also undoes compression, which is harmless.

        :param tuple word: Word to reassign.
        :param tuple root: New representative of the word.
        """
        self.trail.append((word, self.parent[word]))
        self.parent[word] = root

    def _merge(self, first: tuple, second: tuple) -> bool:
        """
        Merges the classes of two words and restores the congruence, that is, merges the targets of any two
        transitions that end up leaving the same class under the same key, until nothing changes. The transitions
        of the absorbed class move to the kept one, and the keys enclosing the absorbed class are rewritten to
        enclose the kept one; every collision with an existing transition is another merge to perform.

        :param tuple first: A word of the first class.
        :param tuple second: A word of the second class.
        :return bool: True if the resulting classes are compatible with the data, that is, no class holds both an
            accepted and a rejected word. False otherwise, in which case the caller rolls the merge back.
        """
        self.label_conflict = False
        self.merge_score = 0
        pending = [(first, second)]
        while pending:
            merged = self._union(*pending.pop())
            if merged is None:
                continue
            # the merge is already known to contradict the data, so restoring the congruence is wasted work
            if self.label_conflict:
                return False
            kept, absorbed = merged

            # what left the absorbed class now leaves the kept one
            for key, target in self.transitions[absorbed].items():
                self._extend(kept, key, target, pending)

            # and a key enclosing the absorbed class now encloses the kept one
            for word, key in self.enclosing.get(absorbed, ()):
                source = self._find(word)
                transitions = self.transitions[source]
                # the pair is stale if that class no longer has the key, be it rewritten or merged away
                if key in transitions:
                    self.transition_trail.append((source, key, transitions[key]))
                    self._extend(source, (key[0], kept, key[2]), transitions.pop(key), pending)

        return True

    def _extend(self, source: tuple, key: tuple, target: tuple, pending: list) -> None:
        """
        Records that the transition under a key leads from a class to a word. If the class already has a
        transition under that key, the two targets belong to the same class, so merging them is scheduled instead.

        :param tuple source: Representative of the class the transition leaves.
        :param tuple key: Key of the transition.
        :param tuple target: Word the transition leads to.
        :param list pending: Worklist of class pairs still to be merged, appended to on a collision.
        """
        transitions = self.transitions[source]
        if key in transitions:
            pending.append((transitions[key], target))
            return

        self.transition_trail.append((source, key, _MISSING))
        transitions[key] = target
        if len(key) == 3:
            self.enclosing[key[1]].add((source, key))

    def _commit(self) -> None:
        """
        Keeps the last merge, which makes it part of the model: nothing recorded up to here can be undone any more.
        """
        self.trail.clear()
        self.transition_trail.clear()
        self.label_trail.clear()

    def _undo(self) -> None:
        """
        Rolls the union-find, the quotient automaton and the class labels back to the state they were in after the
        last merge that was kept.
        """
        for word, previous in reversed(self.trail):
            self.parent[word] = previous
        self.trail.clear()

        for source, key, previous in reversed(self.transition_trail):
            if previous is _MISSING:
                del self.transitions[source][key]
            else:
                self.transitions[source][key] = previous
        self.transition_trail.clear()

        for root in self.label_trail:
            del self.class_label[root]
        self.label_trail.clear()

    def _to_sevpa(self) -> Sevpa:
        """
        Constructs a 1-SEVPA from the learned classes. Call transitions are implicit in a 1-SEVPA, which always
        pushes the pair (state the call is read in, call symbol) and continues from the initial state. A
        transition keyed by a block, leading from a class p to a class t through an enclosed class s, becomes the
        transition popping (p, call symbol) in s and leading to t.

        :return Sevpa: The constructed 1-SEVPA.
        """
        accepting = {self._find(word) for word, label in self.class_label.items() if label}
        roots = sorted({self._find(word) for word in self.parent}, key=self.rank.get)
        states = {root: SevpaState(state_id=f'q{index}', is_accepting=root in accepting)
                  for index, root in enumerate(roots)}

        for root in roots:
            origin_state = states[root]
            for key, target in self.transitions[root].items():
                reached_state = states[self._find(target)]
                if len(key) == 1:
                    origin_state.transitions[key[0]].append(SevpaTransition(reached_state, key[0], None))
                else:
                    call_symbol, enclosed, return_symbol = key
                    states[self._find(enclosed)].transitions[return_symbol].append(
                        SevpaTransition(reached_state, return_symbol, 'pop', (origin_state.state_id, call_symbol)))

        return Sevpa(states[self._find(())], list(states.values()),
                     SevpaAlphabet(list(self.alphabet.internal_alphabet), list(self.alphabet.call_alphabet),
                                   list(self.alphabet.return_alphabet)))
