"""Evaulate a tagger.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Tagging model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import sys

from corpus.ancora import SimpleAncoraCorpusReader

from itertools import starmap

def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    # \b backspace
    print('\b' * width + msg, end='')
    sys.stdout.flush()


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # load the data
    files = '3LB-CAST/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora-3.0.1es/', files)
    sents = list(corpus.tagged_sents())

    # tag
    hits, total = 0, 0
    hits_unknown, total_unknown = 0, 0
    hits_known, total_known = 0, 0
    n = len(sents)
    for i, sent in enumerate(sents):
        word_sent, gold_tag_sent = zip(*sent)

        model_tag_sent = model.tag(word_sent)
        assert len(model_tag_sent) == len(gold_tag_sent), i

        # global score
        hits_sent = [m == g for m, g in zip(model_tag_sent, gold_tag_sent)]
        hits += sum(hits_sent)
        total += len(sent)
        acc = float(hits) / total

        progress('{:3.1f}% ({:2.2f}%)'.format(float(i) * 100 / n, acc * 100))

        # unknown words
        hits_sent_unknown = [hits_sent[j] and model.unknown(word_sent[j]) for j in range(len(hits_sent))]
        hits_unknown += sum(hits_sent_unknown)
        total_unknown += len(hits_sent_unknown)

        # known word
        hits_sent_known = [hits_sent[j] and not model.unknown(word_sent[j]) for j in range(len(hits_sent))]
        hits_known += sum(hits_sent_known)
        total_known += len(hits_sent_known)

    acc = float(hits) / total
    acc_known = float(hits_known) / total_known
    acc_unkown = float(hits_unknown) / total_unknown

    print('')
    print('Accuracy: {:2.2f}%'.format(acc * 100))
    print('Accuracy known words:{:2.2f}%'.format(acc_known * 100))
    print('Acurracy unknown words:{:2.2f}%'.format(acc_unkown * 100))
