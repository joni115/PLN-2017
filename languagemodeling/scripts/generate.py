"""Generate natural language sentences using a language model.

Usage:
  generate.py -i <file> -n <n>
  generate.py -h | --help

Options:
  -i <file>     Language model file.
  -n <n>        Number of sentences to generate.
  -h --help     Show this screen.
"""

from docopt import docopt

from ngram import NGram, NGramGenerator

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import RegexpTokenizer

DEFAULT_DIR = 'corpus'

if __name__ == '__main__':
    opts = docopt(__doc__)

    n = int(opts['-n'])
    filename = opts['-i']
    # we want to tokenize the corpus filename
    pattern = r'''(?ix)    # set flag to allow generate_sentverbose regexps
            (?:sr\.|sra\.)
            | (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
            | \w+(?:-\w+)*        # words with optional internal hyphens
            | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
            | \.\.\.            # ellipsis
            | [][.,;"'?():-_`]
            '''
    tokenizer = RegexpTokenizer(pattern)
    corpus = PlaintextCorpusReader(DEFAULT_DIR, filename, word_tokenizer=tokenizer)
    tokens = corpus.sents()

    # the output will be written in test/output.txt
    file_output = open('outputs/output.txt', 'w')
    for i in range(1, 5):
        # instance an n-gram object whith n={1,2,3,4}
        # add a method instead init. how do I inicializate one time only?.
        ngram = NGram(i, tokens)
        generator = NGramGenerator(ngram)
        file_output.write(str(i) + 'NGram')
        file_output.write('\r\n')
        for _ in range(0, n):
            list_sentence = generator.generate_sent()
            file_output.write(' '.join(list_sentence) + '\n')
        file_output.write('\r\n')

    file_output.close()
