"""Train a sequence tagger.

Usage:
  train.py [-m <model>] [-c <classifier>] [-n <int>] -o <file>
  train.py -h | --help

Options:
  -m <model>    Model to use [default: base].
                  base: Baseline
                  hmm: MLHMM
                  memm: MEMM
  -c <classifier> Classifier to use for memm [default: 'lr'].
                  lr: LogisticRegression
                  mnb: MultinomialNB
                  lsvc: LinearSVC
  -n <int>      (n-1)-gram to evaluate (for hmm) [default: None].
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from corpus.ancora import SimpleAncoraCorpusReader
from tagging.baseline import BaselineTagger
from tagging.hmm import MLHMM
from tagging.memm import MEMM


def MLHMM_trainer(tagged_sents):
    return MLHMM(n, tagged_sents)


def MEM_trainer(tagged_sents):
    return MEMM(n, tagged_sents, c)


models = {
    'base': BaselineTagger,
    'hmm': MLHMM_trainer,
    'memm': MEM_trainer,
}


if __name__ == '__main__':
    opts = docopt(__doc__)
    n = int(opts['-n'])
    c = opts['-c']
    # load the data
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora-3.0.1es/', files)
    sents = list(corpus.tagged_sents())

    # train the model
    model = models[opts['-m']](sents)
    print('{}-{} will be ready'.format(n, opts['-m']))

    # save it
    filename = opts['-o']
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    f.close()
