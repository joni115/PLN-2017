from collections import defaultdict

from nltk.grammar import PCFG

from math import log2

from nltk.tree import Tree

def log2m(x):
    return (lambda x: log2(x) if x > 0 else float('-inf'))(x)

class CKYParser:
 
    def __init__(self, grammar):
        """
        grammar -- a binarised NLTK PCFG. In chomsky normal form.
        """
        self.S = grammar.start()
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
        bp = defaultdict(dict)
        for i in range(1, n + 1):
            xi = sent[i-1]
            for nonT, prob in self.q_terminals[(xi,)].items():
                pi[(i,i)][nonT] = log2m(prob)
                bp[(i,i)][nonT] = Tree(nonT, [xi])

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
                                 last_tree = [bp[(i, s)][Y], bp[(s+1, j)][Z]]
                                 bp[(i, j)][X] = Tree(X, last_tree)

        best_prob = pi[(1, n)].get(self.S, float('-inf'))
        best_parser = bp[(1, n)].get(self.S, None)

        return best_prob, best_parser

