# https://docs.python.org/3/library/collections.html
from collections import defaultdict

from math import log

from random import random

class NGram(object):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)
        # to save only the (n-1)-gram tuple.
        self.tokens = []

        for sent in sents:
            sent += ['</s>']
            sent = ['<s>'] * (n-1) + sent
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1
                if len(ngram) == n:
                    self.tokens += [ngram]

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self.counts.get(tokens)

    def cond_prob(self, token, prev_tokens=None):

        n = self.n
        if not prev_tokens:
            prev_tokens = []
        assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]

        count_prev_tokens = self.counts[tuple(prev_tokens)]
        if count_prev_tokens == 0:
            prob = 0
        else:
            prob = float(self.counts[tuple(tokens)] / count_prev_tokens)
        return prob

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.
        sent -- the sentence as a list of tokens.
        """
        n = self.n

        sent += ['</s>']
        sent = ['<s>'] * (n - 1) + sent

        prob = 1
        for index_word in range(n - 1, len(sent)):
            # the probability of P(word.index_word|word.index_word-1,..., word.(index_word - (n-1)))
            # in a n-gram model.
            prob *= self.cond_prob(sent[index_word], sent[index_word-n+1: index_word])

        return prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.
        sent -- the sentence as a list of tokens.
        """
        n = self.n
        # log2 domain extended. Domain = N U 0.
        log2 = lambda x: log(x, 2) if x > 0 else float('-inf')

        sent += ['</s>']
        sent = ['<s>'] * (n-1) + sent

        prob = 0
        for index_word in range(n - 1, len(sent)):
            prob += log2(self.cond_prob(sent[index_word], sent[index_word-n+1: index_word]))

        return prob

class NGramGenerator:

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self.model = model
        self.probs = probs = defaultdict(dict)
        self.sorted_probs = sorted_probs = defaultdict()

        list_tokens = self.model.tokens
        n = self.model.n
        cond_prob = self.model.cond_prob

        for token in list_tokens:
            word = token[n-1]
            tokens = token[0:n-1]
            # conditional probability of word given the n-1 tokens
            prob_conditional = cond_prob(word, list(tokens))
            probs[tokens][word] = prob_conditional
            # be carefull. ask about this!
            # sorted_probs is for inverse transform sampling.
            sorted_probs[tokens] = sorted(probs[tokens].items())


    def generate_sent(self):
        """Randomly generate a sentence."""



    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self.model.n

        if not prev_tokens:
            prev_tokens = []
        assert len(prev_tokens) == n - 1

        # a copy of self.sorted_probs[prev_tokens]
        sorted_probs_tokens = list(self.sorted_probs[prev_tokens])
        # uniform variable
        U = random()
        # p will represent the p0 + ... + pi probabilities
        p = 0
        next_token = ''
        # inverse transform sampling
        while U > p:
            next_token, pi = sorted_probs_tokens.pop(0)
            p += pi

        return next_token





sents = [
    'el gato come pescado .'.split(),
    'la gata come salmón .'.split(),
]


ngram = NGram(2, sents)
generator = NGramGenerator(ngram)
print (generator.sorted_probs)
