from featureforge.vectorizer import Vectorizer

from sklearn.linear_model import LogisticRegression

from tagging.features import (History, word_lower, word_istitle, word_isupper,
                              word_isdigit, prev_tags, NPrevTags, PrevWord)


class MEMM:
 
    def __init__(self, n, tagged_sents):
        """
        n -- order of the model.
        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        assert n > 0
        self.n = n

        self.__V = set(word for sent in tagged_sents for word, tag in sent)

    def sents_histories(self, tagged_sents):
        """
        Iterator over the histories of a corpus.
 
        tagged_sents -- the corpus (a list of sentences)
        """
        return [self.sent_histories(sent) for sent in tagged_sents]
 
    def sent_histories(self, tagged_sent):
        """
        Iterator over the histories of a tagged sentence.
 
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        n = self.n

        words, tags = zip(*tagged_sent)
        tags = ('<s>',) * (n - 1) + tags
        sent = list(words)

        return [History(sent, tags[index:index+n-1], index) for index in range(len(words))]


    def sents_tags(self, tagged_sents):
        """
        Iterator over the tags of a corpus.
 
        tagged_sents -- the corpus (a list of sentences)
        """
        return [self.sent_tags(sent) for sent in tagged_sents]
 
    def sent_tags(self, tagged_sent):
        """
        Iterator over the tags of a tagged sentence.
 
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        _, tags = zip(*tagged_sents)
        return tags

    def tag(self, sent):
        """Tag a sentence.
 
        sent -- the sentence.
        """
 
    def tag_history(self, h):
        """Tag a history.
 
        h -- the history.
        """
 
    def unknown(self, w):
        """Check if a word is unknown for the model.
 
        w -- the word.
        """
        return w not in self.__V
