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

from sklearn.metrics import confusion_matrix

from collections import Counter


def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    # \b backspace
    print('\b' * width + msg, end='')
    sys.stdout.flush()

def print_confusion_matrix(conf_matrix, labels):
    print('|', end='')
    print('{:^9}'.format('label/label'), end = '')
    for label in labels:
        print ('|{:^9}'.format(label), end='')
    print('|')
    print('|:{:-<9}:'.format(''), end='')
    for _ in range(len(labels)):
        print('|:{:-<7}:'.format(''), end='')
    print('|')
    for i, label in enumerate(labels):
        print('|{:^10} '.format(label), end='')
        for value in conf_matrix[i]:
            print('|{:^9.2f}'.format(value), end='')
        print('|')


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
    # for confusion_matrix
    y_label, x_label = [], []
    n = len(sents)
    for i, sent in enumerate(sents):
        word_sent, gold_tag_sent = zip(*sent)

        model_tag_sent = model.tag(word_sent)
        assert len(model_tag_sent) == len(gold_tag_sent), i

        y_label += gold_tag_sent
        x_label += model_tag_sent

        # global score
        hits_sent = [m == g for m, g in zip(model_tag_sent, gold_tag_sent)]
        hits += sum(hits_sent)
        total += len(sent)

        # unknown words
        hits_sent_unknown = [hits_sent[j] for j in range(len(hits_sent)) if model.unknown(word_sent[j])]
        hits_unknown += sum(hits_sent_unknown)
        total_unknown += len(hits_sent_unknown)

        # known word
        hits_sent_known = [hits_sent[j] for j in range(len(hits_sent)) if not model.unknown(word_sent[j])]
        hits_known += sum(hits_sent_known)
        total_known += len(hits_sent_known)

        acc = float(hits) / total
        acc_known = float(hits_known) / total_known
        acc_unknown = float(hits_unknown) / total_unknown

        progress('{:3.1f}% ({:2.2f}% {:2.2f}% {:2.2f}%)'.format(float(i) * 100 / n, acc * 100, acc_known * 100, acc_unknown * 100))


    assert total == total_known + total_unknown
    assert hits == hits_known + hits_unknown
    acc = float(hits) / total
    acc_known = float(hits_known) / total_known
    acc_unknown = float(hits_unknown) / total_unknown

    # configuration for confusion_matrix
    most_common_tagg, _ = zip(*Counter(y_label).most_common(10))
    conf_matrix = confusion_matrix(y_label, x_label, most_common_tagg)
    conf_matrix = conf_matrix / float(total) * 100

    print('')
    print('Accuracy: {:2.2f}%'.format(acc * 100))
    print('Accuracy known words:{:2.2f}%'.format(acc_known * 100))
    print('Acurracy unknown words:{:2.2f}%'.format(acc_unknown * 100))

    print('\033[91m' + 'Confusion matrix' + '\033[0m')
    print_confusion_matrix(conf_matrix, most_common_tagg)
