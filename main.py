import os
import pprint
import tensorflow as tf

from nltk import word_tokenize
from data import *

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 300, "internal state dimension [300]")
flags.DEFINE_integer("lindim", 300, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 3, "number of hops [7]")
flags.DEFINE_integer("batch_size", 1, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 300, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.01, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 100, "clip gradients to this norm [50]")
flags.DEFINE_string("pretrain_file", "/data/wiki.vi.model.bin",
                    "pre-trained glove vectors file path [../data/wiki.vi.model.bin]")
flags.DEFINE_string("train_data", "data/Laptops_Train.xml.seg",
                    "train gold data set path [./data/Laptops_Train.xml.seg]")
flags.DEFINE_string("test_data", "data/Laptops_Test_Gold.xml.seg",
                    "test gold data set path [./data/Laptops_Test_Gold.xml.seg]")
flags.DEFINE_boolean("show", False, "print progress [False]")
flags.DEFINE_integer("pad_idx", 0, "pad_idx")
flags.DEFINE_integer("nwords", 3425, "nwords")
flags.DEFINE_integer("mem_size", 83, "mem_size")

FLAGS = flags.FLAGS

if __name__ == '__main__':
    embeddings = load_embedding_file(FLAGS.pretrain_file, [])
    print(embeddings)