# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter
from underthesea import word_tokenize
from gensim.models import KeyedVectors
from pyvi import ViTokenizer, ViPosTagger


def load_embedding_file(embed_file_name, word_set):

    embeddings = {}
    with open(embed_file_name, 'r') as embed_file:
        for line in embed_file:
            content = line.strip().split()
            word = content[1]
            if word in word_set:
                embedding = np.array(content[2:], dtype=float)
                embeddings[word] = embedding

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

    stop_words = []
    with open('vietnamese-stopwords.txt', 'r') as data_file:
        lines = data_file.read().split('\n')
        stop_words.extend(lines)

    with open(data_file_name, 'r') as data_file:
        lines = data_file.read().split('\n')
        for line_no in range(0, len(lines) - 1, 3):
            sentence = lines[line_no]
            target = lines[line_no + 1]

            sentence.replace("$T$", "")
            sentence = sentence.lower()
            target = target.lower()
            max_sent_len = max(max_sent_len, len(sentence.split()))

            # vn_sentences = word_tokenize(sentence, format='text')
            vn_sentences = ViTokenizer.tokenize(sentence)
            vn_sentences = vn_sentences.replace("$ t $", "$t$")

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


def get_embedding_matrix(sent_word2idx, target_word2idx, edim):

    word_embed_matrix = np.zeros([len(sent_word2idx), edim], dtype=float)
    target_embed_matrix = np.zeros([len(target_word2idx), edim], dtype=float)

    for word in sent_word2idx:
        if check_vector(word):
            word_embed_matrix[sent_word2idx[word]] = word2vec_model.wv[word]

    for target in target_word2idx:
        for word in target:
            if check_vector(word):
                target_embed_matrix[target_word2idx[target]] += word2vec_model.wv[word]
        target_embed_matrix[target_word2idx[target]] /= max(1, len(target.split()))

    return word_embed_matrix, target_embed_matrix


embed_file_name = 'baomoi.model.bin'
word2vec_model = KeyedVectors.load_word2vec_format(embed_file_name, binary=True)


def get_dataset(data_file_name, sent_word2idx, target_word2idx):

    sentence_list = []
    location_list = []
    target_list = []
    polarity_list = []

    stop_words = []
    with open('vietnamese-stopwords.txt', 'r') as data_file:
        lines = data_file.read().split('\n')
        stop_words.extend(lines)

    dict = {}
    with open(data_file_name, 'r') as data_file:
        lines = data_file.read().split('\n')
        for line_no in range(0, len(lines) - 1, 3):
            sentence = lines[line_no].lower()
            target = lines[line_no + 1].lower()
            polarity = int(lines[line_no + 2])

            # sent_words = sentence.split()
            # sent_words = word_tokenize(sentence, format='text')
            sent_words = ViTokenizer.tokenize(sentence)
            sent_words = sent_words.replace("$ t $", "$t$").split()
            # target_words = target.split()
            # target_words = word_tokenize(target, format='text')
            target_words = ViTokenizer.tokenize(target)
            print(sentence)
            try:
                target_location = sent_words.index("$t$")
            except:
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
                    print("id not found for word in the sentence")
                    exit()

                location_info = abs(index - target_location)

                if check_vector(word):
                    id_tokenised_sentence.append(word_index)
                    location_tokenised_sentence.append(location_info)
                else:
                    dict[word] = word

                # if word not in embeddings:
                #   is_included_flag = 0
                #   break

            is_included_flag = 0
            for word in target_words:
                if check_vector(word):
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

    print(len(dict))
    print(dict)
    return sentence_list, location_list, target_list, polarity_list


def check_vector(word) -> bool:
    try:
        if len(word2vec_model.wv[word]) > 0:
            return True
        else:
            return False
    except:
        return False
