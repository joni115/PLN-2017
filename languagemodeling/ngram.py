# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from math import log

class NGram(object):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)

        for sent in sents:
            sent += ['</s>']
            sent = ['<s>'] * (n-1) + sent
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1

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
            prob *= self.cond_prob(sent[index_word], sent[index_word-n+1: index_word])

        return prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.
        sent -- the sentence as a list of tokens.
        """
        n = self.n
        log2 = lambda x: log(x, 2) if x > 0 else float('-inf')

        sent += ['</s>']
        sent = ['<s>'] * (n-1) + sent

        prob = 0
        for index_word in range(n - 1, len(sent)):
            prob += log2(self.cond_prob(sent[index_word], sent[index_word-n+1: index_word]))

        return prob
