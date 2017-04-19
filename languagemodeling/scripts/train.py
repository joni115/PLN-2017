"""
Train an n-gram model.

Usage:
  train.py -n <n> [-m <model>] [-i <file>] -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
                  interpolated: interpolated smoothing.
  -i <file>     Input corpus [default:].
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from nltk.data import load
# from nltk.corpus import gutenberg
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import RegexpTokenizer

from languagemodeling.ngram import NGram, AddOneNGram, InterpolatedNGram

DEFAULT_DIR = 'corpus'

if __name__ == '__main__':
    opts = docopt(__doc__)

    corpus = opts['-i']
    if not corpus:
        corpus = 'corpus_GOT123_train.txt'

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
    corpus = PlaintextCorpusReader(DEFAULT_DIR, corpus,
                                   word_tokenizer=tokenizer,
                                   sent_tokenizer=sent_tokenizer)
    # sents will be a tokens' list of the corpus
    sents = corpus.sents()

    # train the model
    type_model = opts['-m']
    n = int(opts['-n'])
    if type_model == 'ngram':
        model = NGram(n, sents)
        print(str(n) + '-gram will be ready')
    elif type_model == 'addone':
        model = AddOneNGram(n, sents)
        print(str(n) + '-addone will be ready')
    elif type_model == 'interpolated':
        model = InterpolatedNGram(n, sents)
        print(str(n) + '-interpolated will be ready')
    else:
        print('modelo erroneo')
        exit(0)

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    # to load a object pickle.load(file)
    # dump save the object in bytes
    pickle.dump(model, f)
    f.close()
