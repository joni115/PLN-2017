"""Print corpus statistics.

Usage:
  stats.py
  stats.py -h | --help

Options:
  -h --help     Show this screen.
"""
from docopt import docopt

from corpus.ancora import SimpleAncoraCorpusReader

from collections import defaultdict, Counter


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    corpus = SimpleAncoraCorpusReader('ancora-3.0.1es/')
    sents = list(corpus.tagged_sents())

    words_tags = [(word, tag) for sent in sents for word, tag in sent]
    words, tags = zip(*words_tags)

    # basic statics
    sents_length = len(sents)
    total_words = len(words)
    vocabulary_words = len(set(words))
    vocabulary_tags = len(set(tags))

    # more frequency tags and ambiguity
    counter_words = Counter(words)
    dict_frecuent_words = defaultdict(dict)
    ambiguity = defaultdict(int)

    counter_words_tags = Counter(words_tags)
    for (word, tag), frequency in counter_words_tags.items():
        dict_frecuent_words[tag][word] = frequency
        ambiguity[word] += 1

    # sorted_tags = [(tag1, quantity_tag1, porcent1),...,
    # [(tagn, quantity_tagn, porcentn)]]
    sorted_tags = sorted(map(
                            lambda x: (x, tags.count(x),
                                       tags.count(x) / len(tags)),
                            dict_frecuent_words.keys()),
                         key=lambda p: p[1],
                         reverse=True)
    ten_sorted_tags = sorted_tags[:10]

    level_ambiguity = defaultdict(dict)
    # level_ambiguity
    for word, freq in ambiguity.items():
        level_ambiguity[freq][word] = counter_words[word]

    # compute the statistics
    print('{:_^80}'.format(''))
    print('\033[91m' + '{:>30}'.format('Basic Statics') + '\033[0m')
    print('\033[93m' + 'Sents: {}'.format(sents_length))
    print('Total words: {}'.format(total_words))
    print('Vocabulary words: {}'.format(vocabulary_words))
    print('Vocabulary taggs: {}'.format(vocabulary_tags) + '\033[0m')

    print('{:_^80}'.format(''))
    print('\033[91m' + '{:>35}'.format('Frequencies Taggs') + '\033[0m')
    print('\033[93m' +
          '{:^10}  {:^2}  {:^6}  {:>10}'.format('Tagg',
                                                'Frequencie',
                                                'Percent',
                                                'Words') + '\033[0m')
    for tagg, freq, porcent in ten_sorted_tags:
        word_tag = sorted(dict_frecuent_words[tagg].items(),
                          key=lambda p: p[1],
                          reverse=True)[:5]
        print('{:^11} {:^12} {:4.2f}%'.format(tagg, freq, porcent*100),
              end=' '*2)
        for word, _ in word_tag[0:]:
            print('{}'.format(word), end=';')
        print('')

    print('{:_^80}'.format(''))
    print('\033[91m' + '{:>35}'.format('Ambiguity') + '\033[0m')
    print('\033[93m' + '{:^10}  {:^2}  {:^14}  {}'.format('Level',
                                                          'Amount',
                                                          'Percent',
                                                          'Words') + '\033[0m')
    for i in range(1, 10):
        frequency = len(level_ambiguity[i])
        percent = len(level_ambiguity[i]) / float(vocabulary_words)
        print('{:>5} {:>12} {:>10.2f}%  '.format(i, frequency, percent * 100),
              end='')
        sorted_ambiguity = sorted(level_ambiguity[i].items(),
                                  key=lambda p: p[1],
                                  reverse=True)[:5]
        for word, _ in sorted_ambiguity:
            print('{}'.format(word), end=';')
        print('')
