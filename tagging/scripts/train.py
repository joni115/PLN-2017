"""Train a sequence tagger.

Usage:
  train.py [-m <model>] [-n <int>] -o <file>
  train.py -h | --help

Options:
  -m <model>    Model to use [default: base]:
                  base: Baseline
                  MLHMM: MLHMM
  -n <int>      (n-1)-gram to evaluate (for MLHMM) [default: None].
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from corpus.ancora import SimpleAncoraCorpusReader
from tagging.baseline import BaselineTagger
from tagging.hmm import MLHMM

def MLHMM_trainer(tagged_sents):
    return MLHMM(n, tagged_sents)

models = {
    'base': BaselineTagger,
    'MLHMM': MLHMM_trainer,
}

if __name__ == '__main__':
    opts = docopt(__doc__)
    n = int(opts['-n'])
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
