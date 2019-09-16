import pprint
import tensorflow as tf
from model import MemN2N

from data import *

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 400, "internal state dimension [300]")
flags.DEFINE_integer("lindim", 400, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 4, "number of hops [3]")
flags.DEFINE_integer("batch_size", 1, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 400, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.01, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 100, "clip gradients to this norm [50]")
flags.DEFINE_string("pretrain_file", "baomoi.model.bin",
                    "pre-trained glove vectors file path [baomoi.model.bin]")
flags.DEFINE_string("train_data", "iphone_train.txt", "train gold data set path [.iphone_train_zz.txt]")
flags.DEFINE_string("test_data", "iphone_test.txt", "test gold data set path [.iphone_train_zz.txt]")
flags.DEFINE_boolean("show", False, "print progress [False]")
# flags.DEFINE_integer("pad_idx", 0, "pad_idx")
# flags.DEFINE_integer("nwords", 1642, "nwords")
# flags.DEFINE_integer("mem_size", 134, "mem_size")

FLAGS = flags.FLAGS


def main(_):
    source_word2idx, target_word2idx, word_set = {}, {}, {}
    max_sent_len = -1

    max_sent_len = get_dataset_resources(FLAGS.train_data, source_word2idx, target_word2idx, word_set, max_sent_len)
    max_sent_len = get_dataset_resources(FLAGS.test_data, source_word2idx, target_word2idx, word_set, max_sent_len)

    train_data = get_dataset(FLAGS.train_data, source_word2idx, target_word2idx)
    test_data = get_dataset(FLAGS.test_data, source_word2idx, target_word2idx)

    # FLAGS.pad_idx = source_word2idx['<pad>']
    # FLAGS.nwords = len(source_word2idx)
    # FLAGS.mem_size = max_sent_len

    pp.pprint(flags.FLAGS.__flags)

    print('loading pre-trained word vectors...')
    print('loading pre-trained word vectors for train and test data')

    pre_trained_context_wt, pre_trained_target_wt = get_embedding_matrix(source_word2idx, target_word2idx, FLAGS.edim)

    with tf.Session() as sess:
        model = MemN2N(FLAGS, sess, pre_trained_context_wt, pre_trained_target_wt, source_word2idx['<pad>'],
                       len(source_word2idx), max_sent_len)
        model.build_model()
        model.run(train_data, test_data)


if __name__ == '__main__':
    tf.app.run()
