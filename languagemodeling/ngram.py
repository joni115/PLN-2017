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
        self.tokens = tokens = []

        # to save type_tokens
        type_token = set()
        for sent in sents:
            sent += ['</s>']
            sent = ['<s>'] * (n-1) + sent
            # we want not repetitive tokens
            type_token.update(set(sent))
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1
                # for generate_sent. Save the n-uples tokens.
                if len(ngram) == n:
                    tokens.append(ngram)

        # dont save repetitive tokens
        self.tokens = list(set(tokens))
        # n_vocalbulary = |type_token - {<s>}|
        self.n_vocalbulary = len(type_token - {'<s>'})

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self.counts[tokens]

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

    # define the log_probability. It's usefull for a test_corpus.
    # sents will be a list of test corpus' sentences.
    def log_probability(self, sents):
        log_prob = 0
        for sent in sents:
            log_prob += self.sent_log_prob(sent)

        return log_prob

    def cross_entropy(self, sents):
        # M = |tokens|
        M = len(sents)

        log_prob = self.log_probability(sents)

        return float(log_prob / M)

    def perplexity(self, sents):
        return pow(2, -self.cross_entropy(sents))

class NGramGenerator:

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self.model = model
        self.probs = probs = defaultdict(dict)
        self.sorted_probs = sorted_probs = defaultdict()

        tokens = self.model.tokens
        n = self.model.n
        cond_prob = self.model.cond_prob

        for token in tokens:
            word = token[n-1]
            tokens = token[0:n-1]
            # conditional probability of word given the n-1 tokens
            prob_conditional = cond_prob(word, list(tokens))
            probs[tokens][word] = prob_conditional

        for token in probs.keys():
            # sorted_probs is for inverse transform sampling.
            sorted_probs[token] = sorted(probs[token].items())

    def generate_sent(self):
        """Randomly generate a sentence."""
        n = self.model.n

        # save the sentce in sent_list i.e. if sentece = 'hola como andas'
        # sent_list = ['hola', 'como', 'andas']
        sent_list = ['<s>'] * (n-1)
        prev_tokens = tuple(sent_list)
        next_token = ''
        while '</s>' not in next_token:
            # generate a sent with generate_token.
            next_token = self.generate_token(prev_tokens)
            # space between the words
            sent_list += [next_token]
            # we want the n-1 word for generate the next token
            prev_tokens = tuple(sent_list[len(sent_list)-n + 1:])
        # remove <s>s and </s> for test. return a list.
        return sent_list[n-1:len(sent_list) - 1]

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self.model.n

        if not prev_tokens:
            prev_tokens = ()
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

class AddOneNGram(NGram):
    """
       Todos los métodos de NGram.
       Inheritance class from NGRam.
    """
    def __init__(self, n, sents):
        super().__init__(n, sents)

    def cond_prob(self, token, prev_tokens=None):
        n = self.n
        if not prev_tokens:
            prev_tokens = []
        assert len(prev_tokens) == n - 1

        # |typeWords|
        V = self.n_vocalbulary
        tokens = prev_tokens + [token]

        # add one. i.e. Adding 1 in numerator and V in denominator
        count_prev_tokens = self.counts[tuple(prev_tokens)] + V
        count_tokens = self.counts[tuple(tokens)] + 1
        prob = float(count_tokens / count_prev_tokens)
        return prob

    def V(self):
        """Size of the vocabulary.
        """
        return self.n_vocalbulary
