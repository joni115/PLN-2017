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
        self.__S = grammar.start().symbol()
        q = defaultdict(dict)

        # we want to distinguish nonTerminal from terminal for inicialization.
        # would be util in parse.
        for prod in grammar.productions():
            if len(prod.rhs()) == 1:
                # terminals
                q[prod.rhs()][prod.lhs().symbol()] = prod.prob()
            else:
                # nonTerminals
                # for passing test. Symbol is a string representation
                # for nonTerminals.
                # q will be a dictonary where keys are
                # (nonTerminal, nonTerminal) and (terminals,).
                rhs = (prod.rhs()[0].symbol(), prod.rhs()[1].symbol())
                q[rhs][prod.lhs().symbol()] = prod.prob()

        self.__q_prob = dict(q)

    def parse(self, sent):
        """Parse a sequence of terminals.
 
        sent -- the sequence of terminals.
        """
        n = len(sent)

        # inicialization
        pi = defaultdict(dict)
        bp = defaultdict(dict)
        for i in range(1, n + 1):
            xi = sent[i-1]
            for nonT, prob in self.__q_prob[(xi,)].items():
                pi[(i,i)][nonT] = log2m(prob)
                bp[(i,i)][nonT] = Tree(nonT, [xi])

        for l in range(1, n):
            # begin 1
            for i in range(1, n-l+1):
                j = i + l
                for s in range(i, j):
                    # optimization.
                    # we only want the X-> Y Z in R for s={i..j} where X -> Y Z
                    # have probabilty > 0 
                    for Y, y_prob in pi[(i, s)].items():
                        for Z, z_prob in pi[(s + 1, j)].items():
                            print(self.__q_prob)
                            for X, x_prob in self.__q_prob.get((Y, Z), {}).items():
                                prob = y_prob + z_prob + x_prob
                                best_prob = pi[(i, j)].get(X, float('-inf'))
                                if best_prob < prob:
                                    pi[(i, j)][X] = prob
                                    last_tree = [bp[(i, s)][Y], bp[(s+1, j)][Z]]
                                    bp[(i, j)][X] = Tree(X, last_tree)

        best_prob = pi[(1, n)].get(self.__S, float('-inf'))
        best_parser = bp[(1, n)].get(self.__S, None)

        self._pi = dict(pi)
        self._bp = dict(bp)
        return best_prob, best_parser