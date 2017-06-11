from nltk.tree import Tree

from collections import defaultdict

from parsing.util import lexicalize, unlexicalize

from nltk.grammar import ProbabilisticProduction


class UPCFG:
    """Unlexicalized PCFG.
    """
 
    def __init__(self, parsed_sents, start='sentence'):
        """
        parsed_sents -- list of training trees.
        """

        count_Y_Z = defaultdict(lambda: defaultdict(int))
        count_X = defaultdict(int)
        for t in parsed_sents:
            unle_tree = unlexicalize(t.copy(deep=True))
            for unle_t in unle_tree:
                unle_t.chomsky_normal_form()
                for prod in unle_t.productions():
                    count_Y_Z[prod.lhs()][prod.rhs()] += 1
                    count_X[prod.lhs()] += 1


        productions = []
        for X, c_X in count_X.items():
            for (Y_Z, c_Y_Z) in count_Y_Z[X].items():
                productions.append(ProbabilisticProduction(X, Y_Z, probs=float(c_Y_Z) / float(c_X)))

        self.productions = productions

    def productions(self):
        """Returns the list of UPCFG probabilistic productions.
        """
        return self.productions
 
    def parse(self, tagged_sent):
        """Parse a tagged sentence.
 
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """

# t = Tree.fromstring(
#     """
#         (S
#             (NP (Det el) (Noun gato))
#             (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
#         )
#     """)

# model = UPCFG([t])