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
import pickle

from languagemodeling.ngram import NGram, NGramGenerator

# need to create a output directory in scripts
DEFAULT_OUTPUT_DIR = 'output'

if __name__ == '__main__':
    opts = docopt(__doc__)

    n = int(opts['-n'])
    filename = opts['-i']

    # the output will be written in test/output.txt
    file_output = open(DEFAULT_OUTPUT_DIR + '/output.txt', 'w')
    for i in range(1, 5):
        # instance an n-gram object whith n={1,2,3,4}
        # open the model to read
        file_model = open(str(i) + '-gram.txt', 'rb')
        # ngram is a model trained.
        ngram = pickle.load(file_model)
        # close the file
        file_model.close()
        # an instance of NGramGenerator with ngram
        generator = NGramGenerator(ngram)
        # tittle i-Gram
        file_output.write(str(i) + '-Gram')
        file_output.write('\r\n')
        # generate N sentences
        for _ in range(0, n):
            list_sentence = generator.generate_sent()
            # join list with spaces between word
            file_output.write(' '.join(list_sentence) + '\n')
        # put an EOL
        file_output.write('\r\n')

    file_output.close()
