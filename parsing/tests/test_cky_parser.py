# https://docs.python.org/3/library/unittest.html
from unittest import TestCase
from math import log2

from nltk.tree import Tree
from nltk.grammar import PCFG

from parsing.cky_parser import CKYParser


class TestCKYParser(TestCase):

    def test_parse(self):
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

        lp, t = parser.parse('el gato come pescado crudo'.split())

        # check chart
        pi = {
            (1, 1): {'Det': log2(1.0)},
            (2, 2): {'Noun': log2(0.9)},
            (3, 3): {'Verb': log2(1.0)},
            (4, 4): {'Noun': log2(0.1)},
            (5, 5): {'Adj': log2(1.0)},

            (1, 2): {'NP': log2(0.6 * 1.0 * 0.9)},
            (2, 3): {},
            (3, 4): {},
            (4, 5): {'NP': log2(0.4 * 0.1 * 1.0)},

            (1, 3): {},
            (2, 4): {},
            (3, 5): {'VP': log2(1.0) + log2(1.0) + log2(0.4 * 0.1 * 1.0)},

            (1, 4): {},
            (2, 5): {},

            (1, 5): {'S':
                     log2(1.0) +  # rule S -> NP VP
                     log2(0.6 * 1.0 * 0.9) +  # left part
                     log2(1.0) + log2(1.0) + log2(0.4 * 0.1 * 1.0)},  # right part
        }
        self.assertEqualPi(parser._pi, pi)

        # check partial results
        bp = {
            (1, 1): {'Det': Tree.fromstring("(Det el)")},
            (2, 2): {'Noun': Tree.fromstring("(Noun gato)")},
            (3, 3): {'Verb': Tree.fromstring("(Verb come)")},
            (4, 4): {'Noun': Tree.fromstring("(Noun pescado)")},
            (5, 5): {'Adj': Tree.fromstring("(Adj crudo)")},

            (1, 2): {'NP': Tree.fromstring("(NP (Det el) (Noun gato))")},
            (4, 5): {'NP': Tree.fromstring("(NP (Noun pescado) (Adj crudo))")},

            (3, 5): {'VP': Tree.fromstring(
                "(VP (Verb come) (NP (Noun pescado) (Adj crudo)))")},


            (1, 5): {'S': Tree.fromstring(
                """(S
                    (NP (Det el) (Noun gato))
                    (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
                   )
                """)},
        }
        self.assertEqual(parser._bp, bp)

        # check tree
        t2 = Tree.fromstring(
            """
                (S
                    (NP (Det el) (Noun gato))
                    (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
                )
            """)
        self.assertEqual(t, t2)

        # check log probability
        lp2 = log2(1.0 * 0.6 * 1.0 * 0.9 * 1.0 * 1.0 * 0.4 * 0.1 * 1.0)
        self.assertAlmostEqual(lp, lp2)

    def test_ambiguo(self):
        grammar = PCFG.fromstring(
            """
                S -> NP VP              [1.0]
                VP -> Vt NP             [0.3]
                VP -> VP PP             [0.7]
                NP -> NP PP             [0.6]
                NP -> DT NN             [0.4]
                PP -> IN NP             [1.0]
                Vt -> 'saw'             [1.0]
                NN -> 'man'             [0.33]
                NN -> 'telescope'       [0.33]
                NN -> 'dog'             [0.34]
                DT -> 'the'             [1.0]
                IN -> 'with'            [1.0]
            """)

        parser = CKYParser(grammar)

        lp, t = parser.parse('the man saw the dog with the telescope'.split())

        pi =  {
                (1, 1): {'DT': 0.0},
                (2, 2): {'NN': -1.5994620704162712},
                (3, 3): {'Vt': 0.0},
                (4, 4): {'DT': 0.0},
                (5, 5): {'NN': -1.5563933485243853},
                (6, 6): {'IN': 0.0},
                (7, 7): {'DT': 0.0},
                (8, 8): {'NN': -1.5994620704162712},

                (1, 2): {'NP': -2.9213901653036336},
                (2, 3): {},
                (3, 4): {},
                (4, 5): {'NP': -2.8783214434117474},
                (5, 6): {},
                (6, 7): {},
                (7, 8): {'NP': -2.9213901653036336},

                (1, 3): {},
                (2, 4): {},
                (3, 5): {'VP': -4.6152870375779536},
                (4, 6): {},
                (5, 7): {},
                (6, 8): {'PP': -2.9213901653036336},

                (1, 4): {},
                (2, 5): {},
                (3, 6): {},
                (4, 7): {},
                (5, 8): {},

                (1, 5): {'S': -7.536677202881587},
                (2, 6): {},
                (3, 7): {},
                (4, 8): {'NP': -6.536677202881587},

                (1, 6): {},
                (2, 7): {},
                (3, 8): {'VP': -8.051250375711346},

                (1, 7): {},
                (2, 8): {},

                (1, 8): {'S': -10.972640541014979}
            }

        self.assertEqualPi(parser._pi, pi)

        t2 = Tree.fromstring("""
            (S
              (NP (DT the) (NN man))
              (VP
                (VP (Vt saw) (NP (DT the) (NN dog)))
                (PP (IN with) (NP (DT the) (NN telescope)))))
            """)

        self.assertEqual(t, t2)

    def assertEqualPi(self, pi1, pi2):
        self.assertEqual(set(pi1.keys()), set(pi2.keys()))

        for k in pi1.keys():
            d1, d2 = pi1[k], pi2[k]
            self.assertEqual(d1.keys(), d2.keys(), k)
            for k2 in d1.keys():
                prob1 = d1[k2]
                prob2 = d2[k2]
                self.assertAlmostEqual(prob1, prob2)
