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
        return self.trans.get(prev_tags, {}).get(tag, 0)

    def out_prob(self, word, tag):
        """Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        return self.out.get(tag, {}).get(word, 0)

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

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        N = self.hmm.n
        tag_prob = self.hmm.trans_prob
        out_prob = self.hmm.out_prob
        set_tags = self.hmm.tag_set
        words = list(sent)
        m = len(sent)


        self._pi = pi = defaultdict(dict)

        # inicializate pi.
        pi[0][tuple(['<s>'] * (N - 1))] = (0, [])

        # here begins the magic..
        for k in range(1, m + 1):
            for tag in set_tags:
                e = out_prob(sent[k-1], tag)
                # if e < 0 it's not necesary
                # e = e(word | tag)
                if not e:
                    continue

                for prev_tags, (prob_tagg, tagging) in pi[k-1].items():
                    # q = q(tag | prev_tags)
                    q = tag_prob(tag, prev_tags)
                    # if q == 0. Don't need to add to pi.
                    if q:
                        prob_tagg += log2m(q) + log2m(e)
                        key_tag = (prev_tags + (tag,))[1:]
                        if key_tag not in pi[k] or  pi[k][key_tag][0] < prob_tagg:
                            pi[k][key_tag] = (prob_tagg, tagging + [tag])

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
        self.addone = addone
        self.n = n

        # tcount of n-grams
        self.tcount1 = tcount1 = defaultdict(int)
        # tcount of (n-1)-grams
        # to calculate parameter q is better.
        self.tcount2 = tcount2 = defaultdict(int)
        # for save the vocabulary words
        self.__W = W = set()
        # for out probability. e probability.
        self.__count_out = count_out = Counter()
        # for count tags for e probability.
        self.__count_tags = count_tags = Counter()

        for sent in tagged_sents:
            # why?
            if sent == []:
                continue
            count_out += Counter(sent)
            words, tags = zip(*sent)
            W.update(words)
            count_tags += Counter(tags)
            tags = ('<s>',)*(n-1) + tags + ('</s>',)
            for index in range(len(tags) - n+1):
                tcount1[tags[index:index+n]] += 1
                tcount2[tags[index:index+n-1]] += 1


        self.tag_set = set(count_tags.keys())
        # estimate trans and out probabilities
        self.__get_out_prob()
        self.__get_trans_prob()


    def __get_out_prob(self):
        """
        Calculate out probs i.e. parameter e.
        """
        count_out = self.__count_out
        count_tags = self.__count_tags

        self.out = out = defaultdict(dict)
        for (word, tag), count in count_out.items():
            out[tag][word] = count / count_tags[tag]

    def __get_trans_prob(self):
        """
        Calculate trans probs i.e. parameter q.
        """
        addone = self.addone
        tcount1 = self.tcount1

        # for addone. +1 because of </s>
        T = len(self.tag_set) + 1

        self.trans = trans = defaultdict(dict)
        for tags, count in tcount1.items():
            num = count
            denom = self.tcount(tags[:-1])
            if addone:
                num += 1
                denom += T
            trans[tags[:-1]][tags[-1]] = num / denom


    def tcount(self, tokens):
        """Count for an n-gram or (n-1)-gram of tags.
 
        tokens -- the n-gram or (n-1)-gram tuple of tags.
        """
        n = self.n

        # for n-gram
        if n == len(tokens):
            result = self.tcount1.get(tokens, 0)
        else:
            result = self.tcount2.get(tokens, 0)
        return result

    def unknown(self, w):
        """Check if a word is unknown for the model.
        w -- the word.
        """
        return w not in self.__W

    def trans_prob(self, tag, prev_tags):
        """Overwrite of tans_probs. Probability of a tag with smooth addone.

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        # +1 for </s>
        T = len(self.tag_set) + 1

        trans = self.trans.get(prev_tags, {})
        if self.addone and tag not in trans:
            denom = self.tcount(prev_tags) + T
            result = 1 / denom
        else:
            result = trans.get(tag, 0.0)
        return result

    def out_prob(self, word, tag):
        """OverWrite of out_prob. 
        Probability of a word given a tag whith smooth addone.

        word -- the word.
        tag -- the tag.
        """
        if self.addone and self.unknown(word):
            result = 1 / len(self.__W)
        else:
            result = self.out.get(tag, {}).get(word, 0)

        return result
