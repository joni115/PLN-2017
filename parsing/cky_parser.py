from collections import defaultdict

from nltk.grammar import PCFG

from math import log2

def log2m(x):
    return (lambda x: log2(x) if x > 0 else float('-inf'))(x)

class CKYParser:
 
    def __init__(self, grammar):
        """
        grammar -- a binarised NLTK PCFG. In chomsky normal form.
        """
        q = defaultdict(dict)
        q_2 = defaultdict(dict)

        for prod in grammar.productions():
            if len(prod.rhs()) == 1:
                q_2[prod.rhs()][prod.lhs()] = prod.prob()
            q[prod.lhs()][prod.rhs()] = prod.prob()

        self.q_nonTerminals = dict(q)
        self.q_terminals = dict(q_2)

    def parse(self, sent):
        """Parse a sequence of terminals.
 
        sent -- the sequence of terminals.
        """
        n = len(sent)
        # set of Nonterminals
        N = self.q_nonTerminals.keys()

        # inicialization
        pi = defaultdict(dict)
        for i in range(1, n + 1):
            xi = sent[i-1]
            for nonT, prob in self.q_terminals[(xi,)].items():
                pi[(i,i)][nonT] = log2m(prob)

        for l in range(1, n):
            for i in range(1, n-l+1):
                j = i + l
                for s in range(i, j):
                    for X in N:
                        R = ((term, prob) for (term, prob) in self.q_nonTerminals[X].items() if len(term) == 2)
                        for ((Y, Z), prob) in R:
                             prob = log2m(prob)
                             prob += pi[(i, s)].get(Y, float('-inf'))
                             prob +=  pi[(s+1, j)].get(Z, float('-inf'))
                             actual_prob = pi[(i, j)].get(X, float('-inf'))
                             if prob > actual_prob:
                                 pi[(i, j)][X] = prob

        return pi


grammar = PCFG.fromstring(
    """
        S -> NP VP              [1.0]
        NP -> Det Noun          [0.6]
        NP -> Noun Adj          [0.4]
        VP -> Verb NP           [1.0]
        Det -> 'el'             [1.0]
        Noun -> 'gato'          [0.9]
        Noun -> 'pescado'       [0.1]
        Verb -> 'come'          [1.0]
        Adj -> 'crudo'          [1.0]
    """)

parser = CKYParser(grammar)
print(parser.parse('el gato come pescado crudo'.split()))
