"""Train a parser.

Usage:
  train.py [-m <model>] [-n <int>] -o <file>
  train.py -h | --help

Options:
  -m <model>    Model to use [default: flat]:
                  flat: Flat trees
                  rbranch: Right branching trees
                  lbranch: Left branching trees
                  upcfg: upcfg
  -n <int>      Orden n for horzMarkov.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from corpus.ancora import SimpleAncoraCorpusReader

from parsing.baselines import Flat, RBranch, LBranch

from parsing.upcfg import UPCFG


def train_upcfg(parsed_sents):
    return UPCFG(parsed_sents,horzMarkov=n)

models = {
    'flat': Flat,
    'rbranch': RBranch,
    'lbranch': LBranch,
    'upcfg':train_upcfg,
}

if __name__ == '__main__':
    opts = docopt(__doc__)

    print('Loading corpus...')
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora-3.0.1es/', files)

    n = opts['-n']
    if n:
      n = int(opts['-n'])

    print('Training model...')
    model = models[opts['-m']](corpus.parsed_sents())

    print('Saving...')
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
