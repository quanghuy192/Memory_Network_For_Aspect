import gensim
import numpy as np
from collections import Counter
import os
from distutils.version import LooseVersion

dir_path = os.path.dirname(os.path.realpath(__file__))


# download from https://drive.google.com/open?id=0B1GKSX6YCHXlakkzQ2plZVdUUE0

def load_embedding_file(embed_file_name, word_set):
    model = dir_path + embed_file_name

    embeddings = {}
    if os.path.isfile(model):
        print('Loading word2vec model ...')
        if LooseVersion(gensim.__version__) >= LooseVersion("1.0.1"):
            from gensim.models import KeyedVectors
            word2vec_model = KeyedVectors.load_word2vec_format(model, binary=True)
        else:
            from gensim.models import Word2Vec
            word2vec_model = Word2Vec.load_word2vec_format(model, binary=True)

        for word in word2vec_model.wv.vocab:
            if word not in word_set:
                vec = word2vec_model.wv[word]
                embeddings[word] = vec

    else:
        print(
            "Download word2vec model and put into ./data/. File: "
            "https://drive.google.com/open?id=0B1GKSX6YCHXlakkzQ2plZVdUUE0")
    return embeddings


def get_dataset_resources(data_file_name, sent_word2idx, target_word2idx, word_set, max_sent_len):

    if len(sent_word2idx) == 0:
        sent_word2idx["<pad>"] = 0

    word_count = []
    sent_word_count = []
    target_count = []

    words = []
    sentence_words = []
    target_words = []

    with open(data_file_name, 'r') as data_file:
        lines = data_file.read().split('\n')
        for line_no in range(0, len(lines) - 1, 3):
            sentence = lines[line_no]
            target = lines[line_no + 1]

            sentence = sentence.replace("$T$", "")
            sentence = sentence.lower()
            target = target.lower()
            max_sent_len = max(max_sent_len, len(sentence.split()))
            sentence_words.extend(sentence.split())
            target_words.extend([target])
            words.extend(sentence.split() + target.split())

        sent_word_count.extend(Counter(sentence_words).most_common())
        target_count.extend(Counter(target_words).most_common())
        word_count.extend(Counter(words).most_common())

        for word, _ in sent_word_count:
            if word not in sent_word2idx:
                sent_word2idx[word] = len(sent_word2idx)

        for target, _ in target_count:
            if target not in target_word2idx:
                target_word2idx[target] = len(target_word2idx)

        for word, _ in word_count:
            if word not in word_set:
                word_set[word] = 1

    return max_sent_len
