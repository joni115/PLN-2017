from featureforge.vectorizer import Vectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline

from tagging.features import (History, word_lower, word_istitle, word_isupper,
                              word_isdigit, NPrevTags, PrevWord)

# for classifier
CLF = {
    'lg': LogisticRegression,
    'mnb': MultinomialNB,
    'lsvc': LinearSVC
}

class MEMM:

    def __init__(self, n, tagged_sents, clf='lg'):
        """
        n -- order of the model.
        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        assert n > 0
        self.n = n

        self.__V = set(word for sent in tagged_sents for word, tag in sent)

        # features.
        features = [word_lower, word_istitle, word_isupper, word_isdigit]
        features += [PrevWord(f) for f in features]
        features += [NPrevTags(i) for i in range(1, n)]

        # training data
        training_histories = self.sents_histories(tagged_sents)
        tags = self.sents_tags(tagged_sents)
        # pipeline:
        # make the vectorizer => transformer => classifier easier to work
        classifier = Pipeline([('vect', Vectorizer(features)),
                              ('clf',CLF[clf]())])
        # train the classifier.
        self.classifier = classifier.fit(training_histories, tags)


    def sents_histories(self, tagged_sents):
        """
        Iterator over the histories of a corpus.
 
        tagged_sents -- the corpus (a list of sentences)
        """
        return [histories for sent in tagged_sents for histories in self.sent_histories(sent)]
 
    def sent_histories(self, tagged_sent):
        """
        Iterator over the histories of a tagged sentence.
 
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        n = self.n

        if not tagged_sent:
            return []
        words, tags = zip(*tagged_sent)
        tags = ('<s>',) * (n - 1) + tags
        sent = list(words)

        return [History(sent, tags[index:index+n-1], index) for index in range(len(words))]


    def sents_tags(self, tagged_sents):
        """
        Iterator over the tags of a corpus.
 
        tagged_sents -- the corpus (a list of sentences)
        """
        return [tag for sent in tagged_sents for tag in self.sent_tags(sent)]
 
    def sent_tags(self, tagged_sent):
        """
        Iterator over the tags of a tagged sentence.
 
        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        if not tagged_sent:
            return []
        _, tags = zip(*tagged_sent)
        return tags

    def tag(self, sent):
        """Tag a sentence.
 
        sent -- the sentence.
        """
        n = self.n

        tags = []
        prev_tags = ['<s>'] * (n - 1)
        tags.append(self.tag_history(History(sent, prev_tags, 0)))
        for index in range(1, len(sent)):
            prev_tags = (prev_tags + [tags[index-1]])[1:]
            h = History(sent, prev_tags, index)
            tags.append(self.tag_history(h))

        return tags

    def tag_history(self, h):
        """Tag a history.
 
        h -- the history.
        """
        return self.classifier.predict([h])[0]

    def unknown(self, w):
        """Check if a word is unknown for the model.
 
        w -- the word.
        """
        return w not in self.__V
