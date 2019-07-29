import numpy as np
import requests
# from flask import Flask, render_template, jsonify
from flask import Flask, render_template, abort, request, jsonify
from flask import request, redirect, url_for
import codecs
import gensim
from distutils.version import LooseVersion, StrictVersion

import os

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
            "Download word2vec model and put into ./data/. File: https://drive.google.com/open?id=0B1GKSX6YCHXlakkzQ2plZVdUUE0")
    return embeddings
