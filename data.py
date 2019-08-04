# -*- coding: utf-8 -*-

import gensim
import numpy as np
from collections import Counter
import os
from distutils.version import LooseVersion
from underthesea import word_tokenize

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

            sentence.replace("$T$", "")
            sentence = sentence.lower()
            target = target.lower()
            max_sent_len = max(max_sent_len, len(sentence.split()))

            vn_sentences = word_tokenize(sentence, format='text')
            vn_sentences = vn_sentences.replace("$ t $", "$t$")
            print(vn_sentences)

            sentence_words.extend(vn_sentences.split())
            target_words.extend([target])
            words.extend(vn_sentences.split() + target.split())

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


def get_dataset(data_file_name, sent_word2idx, target_word2idx, embeddings):
    sentence_list = []
    location_list = []
    target_list = []
    polarity_list = []

    with open(data_file_name, 'r') as data_file:
        lines = data_file.read().split('\n')
        for line_no in range(0, len(lines) - 1, 3):
            sentence = lines[line_no].lower()
            target = lines[line_no + 1].lower()
            polarity = int(lines[line_no + 2])

            # sent_words = sentence.split()
            sent_words = word_tokenize(sentence, format='text')
            sent_words = sent_words.replace("$ t $", "$t$").split()
            print(sent_words)
            # target_words = target.split()
            target_words = word_tokenize(target, format='text')
            try:
                target_location = sent_words.index("$t$")
            except:
                print(sentence)
                print("sentence does not contain target element tag")
                exit()

            is_included_flag = 1
            id_tokenised_sentence = []
            location_tokenised_sentence = []

            for index, word in enumerate(sent_words):
                if word == "$t$":
                    continue
                try:
                    word_index = sent_word2idx[word]
                except:
                    print(word)
                    print(sentence)
                    print("id not found for word in the sentence")
                    exit()

                location_info = abs(index - target_location)

                if word in embeddings:
                    id_tokenised_sentence.append(word_index)
                    location_tokenised_sentence.append(location_info)

                # if word not in embeddings:
                #   is_included_flag = 0
                #   break

            is_included_flag = 0
            for word in target_words:
                if word in embeddings:
                    is_included_flag = 1
                    break

            try:
                target_index = target_word2idx[target]
            except:
                print("id not found for target")
                exit()

            if not is_included_flag:
                continue

            sentence_list.append(id_tokenised_sentence)
            location_list.append(location_tokenised_sentence)
            target_list.append(target_index)
            polarity_list.append(polarity)

    return sentence_list, location_list, target_list, polarity_list
