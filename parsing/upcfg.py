from nltk.tree import Tree

from collections import defaultdict

from parsing.util import lexicalize, unlexicalize

from nltk.grammar import ProbabilisticProduction, PCFG, Nonterminal

from parsing.cky_parser import CKYParser


class UPCFG:
    """Unlexicalized PCFG.
    """
 
    def __init__(self, parsed_sents, start='sentence', horzMarkov=None):
        """
        parsed_sents -- list of training trees.
        start -- start symbol.
        horzMarkov -- None for default. A number n >= 0 for horizontal markov.
        """
        self.start = start

        count_Y_Z = defaultdict(lambda: defaultdict(int))
        count_X = defaultdict(int)
        for t in parsed_sents:
            # it's a copy of tree. We don't want to modify the original tree.
            # mutable structures
            unle_trees = unlexicalize(t.copy(deep=True))
            # chomsky normal form with horizontal markov.
            unle_trees.chomsky_normal_form(horzMarkov=horzMarkov)
            # collapse subtrees with a single child.
            unle_trees.collapse_unary(collapsePOS=True)
            for prod in unle_trees.productions():
                count_Y_Z[prod.lhs()][prod.rhs()] += 1
                count_X[prod.lhs()] += 1

        # create a list of productions.
        productions = []
        for X, c_X in count_X.items():
            for (Y_Z, c_Y_Z) in count_Y_Z[X].items():
                q = c_Y_Z / float(c_X)
                productions.append(ProbabilisticProduction(X, Y_Z, prob=q))

        self.production = productions

        grammar = PCFG(Nonterminal(start), productions)
        self.parser = CKYParser(grammar)

    def productions(self):
        """Returns the list of UPCFG probabilistic productions.
        """
        return self.production
 
    def parse(self, tagged_sent):
        """Parse a tagged sentence.
 
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        sent, tags = zip(*tagged_sent)
        prob_sent, tree = self.parser.parse(tags)

        if prob_sent == float('-inf'):
            # flat tree
            return Tree(self.start, [Tree(tag, [word]) for word, tag in tagged_sent])

        # because we want the unchomsky normal form
        # cky's tree is in chomsky normal form.
        tree.un_chomsky_normal_form()

        # now the leaft are words. words in terminal_symbols.
        return lexicalize(tree, sent)
