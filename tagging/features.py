from collections import namedtuple

from featureforge.feature import Feature


# sent -- the whole sentence.
# prev_tags -- a tuple with the n previous tags.
# i -- the position to be tagged.
History = namedtuple('History', 'sent prev_tags i')


def word_lower(h):
    """Feature: current lowercased word.

    h -- a history.
    """
    sent, i = h.sent, h.i
    return sent[i].lower()

def prev_tags(h):
    """ Return the prevs tags
    """
    return h.prev_tags

def word_istitle(h):
    """Feature: if current first letter uppercased word.

    h -- a history
    """
    sent, i = h.sent, h.i
    return sent[i].istitle()

def word_isupper(h):
    """Feature: if current uppercased word.

    h -- a history
    """
    sent, i = h.sent, h.i
    return sent[i].isupper()


def word_isdigit(h):
    """Feature: current number word.

    h -- a history
    """
    sent, i = h.sent, h.i
    return sent[i].isdigit()

def sufix_feature(h):
    sent, i = h.sent, h.i
    return sent[i].endswith('ar') or sent[i].endswith('er') or sent[i].endswith('ir')

class NPrevTags(Feature):
 
    def __init__(self, n):
        """Feature: n previous tags tuple.
 
        n -- number of previous tags to consider.
        """
        self.n = n
 
    def _evaluate(self, h):
        """n previous tags tuple.
 
        h -- a history.
        """
        return prev_tags(h)[-self.n:]


class PrevWord(Feature):
 
    def __init__(self, f):
        """Feature: the feature f applied to the previous word.
 
        f -- the feature.
        """
        self.feature = f
 
    def _evaluate(self, h):
        """Apply the feature to the previous word in the history.
 
        h -- the history.
        """
        i = h.i
        if i == 0:
            result = 'BOS'
        else:
            history = History(h.sent, prev_tags(h), i - 1)
            result = str(self.feature(history))
        return result