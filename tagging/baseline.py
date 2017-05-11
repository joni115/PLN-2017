from collections import Counter, defaultdict

class BaselineTagger:

    def __init__(self, tagged_sents):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        """

        words_tagged = [word_tagg for sent in tagged_sents for word_tagg in sent]
        words_tagged_count = Counter(words_tagged)

        # best_tag will save the best tag of the word.
        # example {(word1, tagg1): 10, (word2, tagg20): 30,...}
        best_tag = defaultdict(int)
        # to save the best tagg for a word
        taggs_for_words = defaultdict(str)
        for word, tagg in words_tagged_count.keys():
            if best_tag[(word, tagg)] < words_tagged_count[(word, tagg)]:
                taggs_for_words[word] = tagg
                best_tag[(word, tagg)] = words_tagged_count[(word, tagg)]

        self.taggs_for_words = taggs_for_words
        self.unk_word = 'nc0s000'

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """Tag a word.
        w -- the word.
        """
        return self.taggs_for_words.get(w, self.unk_word)

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self.taggs_for_words
