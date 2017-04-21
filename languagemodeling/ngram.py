# https://docs.python.org/3/library/collections.html
from collections import defaultdict

from math import log

import numpy as np

from random import random


def log2(x):
    # log2 domain extended. Domain = N U 0.
    return (lambda x: log(x, 2) if x > 0 else float('-inf'))(x)


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
            # the probability of
            # P(word.index_word,...,word.(index_word - (n-1)))
            # in a n-gram model.
            prob *= self.cond_prob(sent[index_word],
                                   sent[index_word-n+1: index_word])

        return prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.
        sent -- the sentence as a list of tokens.
        """
        n = self.n

        sent += ['</s>']
        sent = ['<s>'] * (n-1) + sent

        prob = 0
        for index_word in range(n - 1, len(sent)):
            prob += log2(self.cond_prob(sent[index_word],
                         sent[index_word-n+1: index_word]))

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
        M = 0
        for sent in sents:
            M += len(sent)

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
            # token is a n-tuple
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

class InterpolatedNGram(NGram):

    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: False).
        """
        assert n > 0
        self.n = n
        self.gamma = gamma
        # for addone
        self.addone = addone

        self.counts = counts = defaultdict(int)

        if not gamma:
            # the porcent of held-out. It's 90% of train data.
            # last 10% for held_out because of test.
            porcent = int(0.9 * len(sents))
            held_out = sents[porcent:]
            sents = sents[:porcent]

        # for type_tokens to count V
        type_token = set()
        for sent in sents:
            # to evit underflow
            sent += ['</s>']
            sent = ['<s>'] * (n-1) + sent
            type_token.update(set(sent))
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                # all the n-uples where n = {1, 2, 3, ..., n}
                # i.e. if sents=['<s>' , 'hola', 'che', '</s>']
                # with n = 2
                # counts will be {(): 4, (hola,): 1, (che,):1, (</s>,):1,
                # ('<s>', 'hola'): 1, ('hola', 'che'): 1,
                # ('che', '</s>'): 1}
                for j in range(1, n+1):
                    counts[ngram[j:]] += 1

        # n_vocalbulary = |type_token - {<s>}|
        self.n_vocalbulary = len(type_token - {'<s>'})

        if not gamma:
            # use "barrido" for get the gamma
            self.get_gamma(held_out)

    def cond_prob(self, token, prev_tokens=None):
        n = self.n
        if not prev_tokens:
            prev_tokens = []
        assert len(prev_tokens) == n - 1
        # a list of gamma i.e. gamma = [gamma, gamma, gamma, gamma, 0].
        # look for another solution
        gamma = [self.gamma for i in range(n - 1)]
        gamma.append(0)

        # |type_vocabulary|
        V = self.n_vocalbulary
        # P(tokens| prev_tokens)
        tokens = prev_tokens + [token]
        # lambas will be a list of lambda1, lambda2, ..., lambdan
        lambdas = []
        prob = 0
        # calculate q_ML from n-gram down to unigram
        for i in range(n):
            # q_ML(token | prev_tokens) = count_token / count_prev_tokens.
            # we want q_ML for n-grams n={1,2,3,..,n} i.e.
            # q_ML(xn| xi...xn-1) that i={1, 2, 3,...,n-1}.
            # example tokens = ['hola', 'que', 'haces'] which
            # prev_tokens = ['que', 'haces'] so we want
            # q_ML(hola | que haces), q_ML(hola | que) and q_ML(hola)
            count_token = self.count(tuple(tokens[i:]))
            count_prev_tokens = self.count(tuple(prev_tokens[i:]))
            # if addone in unigrams.
            if self.addone and i == (n - 1):
                count_token += 1
                count_prev_tokens += self.n_vocalbulary
            # if denominator is 0, the probability of i-gram will be 0.
            if count_prev_tokens:
                # lambd  =
                # (c(prev_tokens) / (c(prev_tokens) + gamma))*(1 - sum(lambd))
                lambd = count_prev_tokens / float(count_prev_tokens + gamma[i])
                lambd *= (1 - sum(lambdas))
                # lambdas must be between 0 and 1.
                assert (lambd >= 0 and lambd <= 1)
                lambdas.append(lambd)
                q_ML = count_token / float(count_prev_tokens)
                prob += lambd * q_ML

        assert (sum(lambdas) == 1 or not self.addone)
        return prob

    def get_gamma(self, sents, minim=0, maximun=1000, jump = 100):
        # to save best gamma
        best_gamma = minim
        best_log_prob = float('-inf')
        # 'barrer' from minim to maximun with jump
        for gamma in range(minim, maximun, jump):
            # to get the log_probability with this gamma.
            self.gamma = gamma
            # to get the 'best gamma' is better use log_probability than
            # cross_entropy or perplexity because of operation's number.
            log_prob_gamma = self.log_probability(sents)
            if best_log_prob < log_prob_gamma:
                best_gamma = gamma
                best_log_prob = log_prob_gamma
            print(gamma, ' |-> ', log_probability)
        self.gamma = best_gamma
        print('best gamma was:', self.gamma)

class BackOffNGram(NGram):

    def __init__(self, n, sents, beta=None, addone=True):
        """
        Back-off NGram model with discounting as described by Michael Collins.

        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        beta -- discounting hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        assert not beta or (0 <= beta <= 1)
        self.n = n
        # for beta
        self.beta = beta
        # for the counts
        self.counts = counts = defaultdict(int)
        # for set A
        self.Aset = Aset = defaultdict(set)

        # for addone
        self.addone = addone

        copy_sents = list(sents)
        # a copy of sents because of test.
        if not beta:
            # the porcent of held-out. It's 90% of train data.
            # last 10% for held_out because of test.
            porcent = int(0.9 * len(sents))
            held_out = copy_sents[porcent:]
            copy_sents = copy_sents[:porcent]

        # for type_tokens to count V
        type_token = set()
        for sent in copy_sents:
            # to evit underflow
            # have to do a copy of sent because the tests.
            copy_sent = ['<s>'] * (n-1) + list(sent) + ['</s>']
            # for save the count of <s>.
            for i in range(1, n):
                counts[tuple(['<s>'] * i)] += n - i
            type_token.update(set(copy_sent))
            for i in range(len(copy_sent) - n + 1):
                ngram = tuple(copy_sent[i: i + n])
                counts[ngram] += 1
                # bigram: Aset(v) = {w | c(ngram) > 0}.
                # trigram: Aset(ngram[:-1]) = {w | c(ngram[:-1]) > 0}.
                Aset[ngram[:-1]].add(ngram[-1])
                # all the n-uples where n = {1, 2, 3, ..., n}
                # i.e. if sents=['<s>' , 'hola', 'che', '</s>']
                # with n = 2
                # counts will be {(): 4, (hola,): 1, (che,):1, (</s>,):1,
                # ('<s>', 'hola'): 1, ('hola', 'che'): 1,
                # ('che', '</s>'): 1}
                for j in range(1, n - 1):
                    # Aset for all the models.
                    Aset[ngram[j:][:-1]].add(ngram[j:][-1])
                for j in range(1, n+1):
                    counts[ngram[j:]] += 1

        # n_vocalbulary = |type_token - {<s>}|
        self.n_vocalbulary = len(type_token - {'<s>'})

        if not beta:
            # this method get denominator and alpha
            # use "barrido" for get beta
            self._get_beta(held_out)
        else:
            # to get alphas
            self._get_alphas()
            # to get denominator
            self._get_denominator()

    def _get_beta(self, sents):
        # to choose best log probability and beta
        best_log_prob = float('-inf')
        best_beta = 0
        # for use barrido in betta = [0.0, 0.05, 0.1, 0.15, ....., 1.0]
        for beta in np.arange(0.0, 1.05, 0.05):
            self.beta = beta
            self._get_alphas()
            self._get_denominator()

            log_probability = self.log_probability(sents)

            if best_log_prob < log_probability:
                best_log_prob = log_probability
                best_beta = self.beta

            # print(beta, ' |-> ', log_probability)

        self.beta = best_beta

    def _get_alphas(self):
        self.alphas = alphas = defaultdict(float)
        beta = self.beta
        for tokens in self.Aset.keys():
            alpha = 1
            # |Aset(tokens)|
            len_Aset = len(self.A(tokens))
            # if A != empty
            if len_Aset:
                alpha = beta * len_Aset / self.count(tokens)
            alphas[tokens] = alpha

    def _get_denominator(self):
        self.denominator = defaultdict(float)
        for tokens in self.Aset.keys():
            # to get denominator of tokens.
            sumatory = sum(self.cond_prob(x, list(tokens[1:])) for x in self.A(tokens))
            self.denominator[tokens] = 1 - sumatory
            assert 0 <= sumatory <= 1

    def A(self, tokens):
        """Set of words with counts > 0 for a k-gram with 0 < k < n.
        tokens -- the k-gram tuple.
        """
        return self.Aset[tokens]

    def alpha(self, tokens):
        """Missing probability mass for a k-gram with 0 < k < n.
        tokens -- the k-gram tuple.
        """
        return self.alphas[tokens]

    def denom(self, tokens):
        """Normalization factor for a k-gram with 0 < k < n.
        tokens -- the k-gram tuple.
        """
        return self.denominator.get(tokens)

    def cond_prob(self, token, prev_tokens=None):
        n = self.n
        beta = self.beta
        if not prev_tokens:
            prev_tokens = []
        # P(tokens| prev_tokens)
        tokens = prev_tokens + [token]
        # token in A(previous_token)
        probability = 0
        # get the probability of q_D(token_tuple) with or without addone.
        if len(prev_tokens) == 0:
            numerator = self.count(tuple([token]))
            denominator = self.count(tuple())
            if self.addone:
                numerator += 1
                denominator += self.n_vocalbulary

            probability = numerator / float(denominator)
            return probability
        set_of_token = self.A(tuple(prev_tokens))
        # if token in set A(prev_tokens).
        if token in set_of_token:
            numerator = self.count(tuple(tokens)) - beta
            denominator = self.count(tuple(prev_tokens))
            if denominator:
                probability = numerator / float(denominator)
            return probability
        # if token in set B(prev_tokens).
        else:
            numerator = self.alpha(tuple(prev_tokens)) * self.cond_prob(token, prev_tokens[1:])
            denominator = self.denom(tuple(prev_tokens))
            if denominator:
                probability = numerator / float(denominator)
            return probability
