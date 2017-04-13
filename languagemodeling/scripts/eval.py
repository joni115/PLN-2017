"""
Evaulate a language model using the test set.

Usage:
  eval.py -i <file> [-c <file>]
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -c <file>    Input test corpus [default:].
  -h --help     Show this screen.
"""
from docopt import docopt

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import RegexpTokenizer
from nltk.data import load

import pickle

DEFAULT_DIR = 'corpus'

if __name__ == '__main__':
    opts = docopt(__doc__)

    file_model = opts['-i']
    corpus_test = opts['-c']

    if not corpus_test:
        corpus_test = 'corpus_GOT4_test.txt'

    pattern = r'''(?ix)    # set flag to allow verbose regexps
            (?:sr\.|sra\.|mr\.|mrs\.)
            | (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
            | \w+(?:-\w+)*        # words with optional internal hyphens
            | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
            | \.\.\.              # ellipsis
            | [][.,;"'?():-_`]
            '''

    # for sent in spanish
    sent_tokenizer = load('tokenizers/punkt/spanish.pickle')
    tokenizer = RegexpTokenizer(pattern)
    corpus = PlaintextCorpusReader(DEFAULT_DIR, corpus_test,
                                   word_tokenizer=tokenizer,
                                   sent_tokenizer=sent_tokenizer)
    # sents will be a tokens' list of the corpus
    sents = corpus.sents()

    # open a model in rb
    file_model = open(file_model, 'rb')
    # load the model
    ngram = pickle.load(file_model)
    # wont use so it close.
    file_model.close()

    print('the perplexity is:', ngram.perplexity(sents))
