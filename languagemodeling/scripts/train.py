"""Train an n-gram model.

Usage:
  train.py [-n <n>] [-o <file>]
  train.py -h | --help

Options:
  -n <n>        Order of the model [default: None].
  -o <file>     Output model file [default: ejemplo.txt].
  -h --help     Show this screen.
"""

DEFAULT_DIR = 'test'

from docopt import docopt
import pickle

from nltk.corpus import gutenberg
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import RegexpTokenizer

from languagemodeling.ngram import NGram

def gutenberg(arg):
    n = int(arg['-n'])
    filename = arg['-o']

    sents = gutenberg.sents('austen-emma.txt')
    # train the model
    model = NGram(n, sents)

    # save it
    f = open(filename, 'wb')
    pickle.dump(model, f)
    print ("Se guardara en", filename)
    f.close()

def my_token(filename, directory=DEFAULT_DIR):
    pattern = r'''(?ix)    # set flag to allow verbose regexps
            (?:sr\.|sra\.)
            | (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
            | \w+(?:-\w+)*        # words with optional internal hyphens
            | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
            | \.\.\.            # ellipsis
            | [][.,;"'?():-_`]
            '''
    tokenizer = RegexpTokenizer(pattern)
    corpus = PlaintextCorpusReader(DEFAULT_DIR, 'ejemplo.txt', word_tokenizer=tokenizer)
    return corpus.sents()

if __name__ == '__main__':
    opts = docopt(__doc__)
    print (not opts['-n'])
    if not opts['-n']:
        gutenberg(opts)
    else:
        print (my_token(opts['-o']))
