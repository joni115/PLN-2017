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
        m = len(sent)


        self._pi = pi = defaultdict(dict)

        pi[0][tuple(['<s>'] * (N - 1))] = (0, [])

        for k in range(1, m + 1):
            # the previous keys.
            for w in pi[k-1].keys():
                # only the next possible tags.
                # for others tags the probability will be -inf.
                for next_tagg in trans[w].keys():
                    prob = 0
                    previous_tagg =  pi[k-1][w][1]
                    list_of_tag = previous_tagg + [next_tagg]
                    # prob pi(k-1, w, v)
                    prob += pi[k-1][w][0]
                    prob += log2m(tag_prob(next_tagg, w))
                    prob += log2m(out_prob(words[k-1], next_tagg))
                    # key_tag will be taggs (the actual taggs).
                    key_tag = tuple(list(w[1:]) + list(next_tagg))
                    if key_tag not in pi[k] or pi[k][key_tag][0] < prob:
                        pi[k][key_tag] = (prob, list_of_tag)

        max_prob = float('-inf')
        best_tagging = []
        for taggs in pi[m].keys():
            prob = pi[m][taggs][0]
            prob += log2m(tag_prob('</s>', taggs))
            if max_prob < prob:
                max_prob = prob
                best_tagging = pi[m][taggs][1]

        return best_tagging


class MLHMM(HMM):

    def __init__(self, n, tagged_sents, addone=True):
        """
        n -- order of the model.
        tagged_sents -- training sentences, each one being a list of pairs.
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self.n = n
        self.addone = addone


        words_tagg = [word_tagg for sents in tagged_sents for word_tagg in sents + [('', '</s>')]]
        words, taggs = zip(*words_tagg)
        # for estimate e.
        self.__count_word_tag = count_word_tag = Counter(words_tagg)
        self.__count_tagg = count_tag = Counter(taggs)
        # for uknown words.
        self.__V = set(words)
        # for tcount.
        lcount = []
        for i in range(n):
            lcount.append(taggs[i:])

        self.tcount1 = Counter(zip(*(lcount)))

        if n == 1:
            self.tcount2 = {'()': len(taggs)}
        else:
            self.tcount2 = Counter(zip(*(lcount[:-1])))

        self.init_hmm()

    def tcount(self, tokens):
        """Count for an n-gram or (n-1)-gram of tags.
        tokens -- the n-gram or (n-1)-gram tuple of tags.
        """
        n = self.n
        result = 0

        if len(tokens) == n:
            result = self.tcount1.get(tokens, 0)
        else:
            result = self.tcount2.get(tokens, 0)

        return result

    def unknown(self, w):
        """Check if a word is unknown for the model.
        w -- the word.
        """
        return w in self.__V

    def init_hmm(self):
        n = self.n
        V = len(self.__V)
        count_word_tag = self.__count_word_tag
        count_tagg = self.__count_tagg
        addone = self.addone

        tagset = count_tagg.keys()
        out = defaultdict(dict)
        for (word, tagg), count in count_word_tag.items():
            denom = float(count_tagg[tagg])
            out[tagg][word] = count / denom

        trans = defaultdict(dict)
        for tagg in self.tcount1.keys():
            num = self.tcount(tagg[:n-1])
            denom = self.tcount(tagg[n-1])
            if addone:
                num += 1
                denom += V
                trans[tagg[:n-1]][tagg[n-1]] = num / float(denom)
            else:
                if not denom:
                    trans[tagg[:n-1]][tagg[n-1]] = 0
                else:
                    trans[tagg[:n-1]][tagg[n-1]] = num / float(denom)

        HMM.__init__(self, N, tagset, trans, out)
