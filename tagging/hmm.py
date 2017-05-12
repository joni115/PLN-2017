from collections import defaultdict, Counter

from math import log2

def log2m(x):
    return (lambda x: log2(x) if x > 0 else float('-inf'))(x)

class HMM:

    def __init__(self, n, tagset, trans, out):
        """
        n -- n-gram size.
        tagset -- set of tags.
        trans -- transition probabilities dictionary.
        out -- output probabilities dictionary.
        """
        self.n = n
        self.tag_set = tagset
        self.trans = trans
        self.out = out

    def tagset(self):
        """Returns the set of tags.
        """
        return self.tag_set

    def trans_prob(self, tag, prev_tags):
        """Probability of a tag.

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        return self.trans.get(prev_tags).get(tag, 0)

    def out_prob(self, word, tag):
        """Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        return self.out.get(tag).get(word, 0)

    def tag_prob(self, y):
        """
        Probability of a tagging.
        Warning: subject to underflow problems.

        y -- tagging.
        """
        n = self.n

        list_y = list(y)
        list_y = ['<s>'] * (n - 1) + list_y + ['</s>']

        prob = 1
        for index in range(n - 1, len(list_y)):
            prob *= self.trans_prob(list_y[index], tuple(list_y[index-n+1: index]))

        return prob

    def prob(self, x, y):
        """
        Joint probability of a sentence and its tagging.
        Warning: subject to underflow problems.

        x -- sentence.
        y -- tagging.
        """
        assert len(x) == len(y)

        prob = 1
        for i in range(len(x)):
            prob *= self.out_prob(x[i], y[i])

        return prob * self.tag_prob(y)

    def tag_log_prob(self, y):
        """
        Log-probability of a tagging.

        y -- tagging.
        """
        n = self.n

        list_y = list(y)
        list_y = ['<s>'] * (n - 1) + list_y + ['</s>']

        prob = 0
        for index in range(n - 1, len(list_y)):
            prob += log2m(self.trans_prob(list_y[index], tuple(list_y[index-n+1: index])))

        return prob

    def log_prob(self, x, y):
        """
        Joint log-probability of a sentence and its tagging.

        x -- sentence.
        y -- tagging.
        """
        assert len(x) == len(y)

        prob = 0
        for i in range(len(x)):
            prob += log2m(self.out_prob(x[i], y[i]))

        return prob + self.tag_log_prob(y)

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        V = ViterbiTagger(self)
        return V.tag(sent)


class ViterbiTagger:

    def __init__(self, hmm):
        """
        hmm -- the HMM.
        """
        self.hmm = hmm
        self._pi = None

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        N = self.hmm.n
        trans = self.hmm.trans
        tag_prob = self.hmm.trans_prob
        out_prob = self.hmm.out_prob
        words = list(sent)


        self._pi = pi = defaultdict(dict)

        pi[0][tuple(['<s>'] * (N - 1))] = (0, [])

        for k in range(1, len(sent) + 1):
            for w in pi[k-1].keys():
                for next_tagg in trans[w].keys():
                    prob = 0
                    previous_tagg =  pi[k-1][w][1]
                    list_of_tag = previous_tagg + [next_tagg]
                    # prob pi(k-1, w, v)
                    prob += pi[k-1][w][0]
                    prob += log2m(tag_prob(next_tagg, w))
                    prob += log2m(out_prob(words[k-1], next_tagg))
                    key_tag = tuple(list(w[1:]) + list(next_tagg))
                    if key_tag not in pi[k] or pi[k][key_tag][0] < prob:
                        pi[k][key_tag] = (prob, list_of_tag)




tagset = {'D', 'N', 'V'}
trans = {
    ('<s>', '<s>'): {'D': 1.0},
    ('<s>', 'D'): {'N': 1.0},
    ('D', 'N'): {'V': 0.8, 'N': 0.2},
    ('N', 'N'): {'V': 1.0},
    ('N', 'V'): {'</s>': 1.0},
}
out = {
    'D': {'the': 1.0},
    'N': {'dog': 0.4, 'barks': 0.6},
    'V': {'dog': 0.1, 'barks': 0.9},
}
hmm = HMM(3, tagset, trans, out)
tagger = ViterbiTagger(hmm)

x = 'the dog barks'.split()
y = tagger.tag(x)
