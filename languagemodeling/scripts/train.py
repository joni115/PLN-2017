"""
Train an n-gram model.

Usage:
  train.py -n <n> [-m <model>] -i <file> -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
  -i <file>     Input corpus.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from nltk.corpus import gutenberg
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import RegexpTokenizer

import os
from languagemodeling.ngram import NGram, AddOneNGram

DEFAULT_DIR = 'corpus'

if __name__ == '__main__':
    opts = docopt(__doc__)

    corpus = opts['-i']

    pattern = r'''(?ix)    # set flag to allow verbose regexps
            (?:sr\.|sra\.|mr\.|mrs\.)
            | (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
            | \w+(?:-\w+)*        # words with optional internal hyphens
            | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
            | \.\.\.              # ellipsis
            | \w*[ñáéíóúÑÁÉÍÓÚ]\w* # special character for spanish
            | [][.,;"'?():-_`]
            '''
    tokenizer = RegexpTokenizer(pattern)
    corpus = PlaintextCorpusReader(DEFAULT_DIR, corpus, word_tokenizer=tokenizer)
    sents = corpus.sents()

    # train the model
    type_model = opts['-m']
    n = int(opts['-n'])
    if type_model == 'ngram':
        model = NGram(n, sents)
    elif type_model == 'addone':
        model = AddOneNGram(n, sents)
    else:
        print ('modelo erroneo')
        exit(0)

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    # to load a object pickle.load(file)
    pickle.dump(model, f)
    f.close()
