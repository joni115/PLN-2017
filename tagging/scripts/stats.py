"""Print corpus statistics.

Usage:
  stats.py
  stats.py -h | --help

Options:
  -h --help     Show this screen.
"""
from docopt import docopt

from corpus.ancora import SimpleAncoraCorpusReader

from collections import defaultdict


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    corpus = SimpleAncoraCorpusReader('ancora-3.0.1es/')
    sents = list(corpus.tagged_sents())
    sents_length = len(sents)

    # to save all words and taggs.
    words = []
    tags = []
    # dict_frecuent_words will be dictonary of dictonary where keys where the tagg.
    # example: {'aq000':{la: 10, el: 50, .....}, 'sp0000':{de: 80, ....}}
    dict_frecuent_words = defaultdict(dict)
    # keys were word and values are taggs.
    ambiguity = defaultdict(set)
    for sent in sents:
        for word, tagg in sent:
            try:
                dict_frecuent_words[tagg][word] += 1
            except KeyError:
                dict_frecuent_words[tagg][word] = 1
            ambiguity[word].update((tagg,))
            words.append(word)
            tags.append(tagg)

    # to save ambiguity levels. for example
    # {1: {'el', 'la'}, 2: {'flaco', 'enano'}, ...}
    level_ambiguity = defaultdict(set)
    for key in ambiguity.keys():
        level_ambiguity[len(ambiguity[key])].update((key,))

    total_words = len(words)
    vocabulary_words = len(set(words))
    vocabulary_tags = len(set(tags))

    # sorted_tags = [(tag1, quantity_tag1, porcent1),...,
    # [(tagn, quantity_tagn, porcentn)]]
    sorted_tags = sorted(map(lambda x: (x, tags.count(x), tags.count(x) / len(tags)), dict_frecuent_words.keys()), key=lambda p: p[1], reverse=True)
    ten_sorted_tags = sorted_tags[:10]

    # compute the statistics
    print('{:_^80}'.format(''))
    print('\033[91m' + '{:>30}'.format('Basic Statics') + '\033[0m')
    print('\033[93m' + 'Sents: {}'.format(sents_length))
    print('Total words: {}'.format(total_words))
    print('Vocabulary words: {}'.format(vocabulary_words))
    print('Vocabulary taggs: {}'.format(vocabulary_tags) + '\033[0m')

    print('{:_^80}'.format(''))
    print('\033[91m' + '{:>35}'.format('Frequencies Taggs') + '\033[0m')
    print('\033[93m' + '{:^10}  {:^2}  {:^6}  {:>10}'.format('Tagg', 'Frequencie', 'Percent', 'Words') + '\033[0m')
    for tagg, freq, porcent in ten_sorted_tags:
        word_tag = sorted(dict_frecuent_words[tagg].items(), key=lambda p: p[1], reverse=True)[:5]
        print('{:^11} {:^12} {:4.3f} {:^19}'.format(tagg, freq, porcent, word_tag[0][0]))
        for word, _ in word_tag[1:]:
            print('{:^80}'.format(word))

    print('{:_^80}'.format(''))
    print('\033[91m' + '{:>35}'.format('Ambiguity') + '\033[0m')
    print('\033[93m' + '{:^10}  {:^2}  {:^10}'.format('Level', 'Amount', 'Percent') + '\033[0m')
    for i in range(1, 10):
        amount = len(level_ambiguity[i])
        percent = len(level_ambiguity[i]) / float(vocabulary_words)
        print('{:>5} {:>12} {:>10.6f}'.format(i, amount, percent))
