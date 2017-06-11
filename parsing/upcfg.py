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

        count_X = defaultdict(lambda: defaultdict(int))
        count_Y_Z = defaultdict(int)
        for t in parsed_sents:
            unle_tree = unlexicalize(t.copy(deep=True))
            for unle_t in unle_tree:
                unle_t.chomsky_normal_form()
                for prod in unle_t.productions():
                    count_gramat[prod.lhs()][prod.rhs()] += 1
                    count_gramat[prod.lhs()] += 1



    def productions(self):
        """Returns the list of UPCFG probabilistic productions.
        """
 
    def parse(self, tagged_sent):
        """Parse a tagged sentence.
 
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """

t = Tree.fromstring(
    """
        (S
            (NP (Det el) (Noun gato))
            (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
        )
    """)

model = UPCFG([t])