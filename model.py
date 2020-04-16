import os
import math
import numpy as np
import tensorflow as tf
from past.builtins import xrange
import matplotlib.pyplot as plt
from underthesea import word_tokenize


class MemN2N(object):
    def __init__(self, config, sess, pre_trained_context_wt, pre_trained_target_wt, pad_idx, nwords, mem_size, target_word2idx):
        # self.nwords = config.nwords
        self.init_hid = config.init_hid
        self.init_std = config.init_std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.nhop = config.nhop
        self.edim = config.edim
        # self.mem_size = config.mem_size
        self.lindim = config.lindim
        self.max_grad_norm = config.max_grad_norm
        # self.pad_idx = config.pad_idx
        self.pre_trained_context_wt = pre_trained_context_wt
        self.pre_trained_target_wt = pre_trained_target_wt
        self.checkpoint_dir = config.checkpoint_dir

        self.pad_idx = pad_idx
        self.nwords = nwords
        self.mem_size = mem_size

        self.input = tf.compat.v1.placeholder(tf.compat.v1.int32, [self.batch_size, 1], name="input")
        self.time = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, self.mem_size], name="time")
        self.target = tf.compat.v1.placeholder(tf.compat.v1.int64, [self.batch_size], name="target")
        self.context = tf.compat.v1.placeholder(tf.compat.v1.int32, [self.batch_size, self.mem_size], name="context")
        self.mask = tf.compat.v1.placeholder(tf.compat.v1.float32, [self.batch_size, self.mem_size], name="mask")
        self.neg_inf = tf.compat.v1.fill([self.batch_size, self.mem_size], -1 * np.inf, name="neg_inf")

        self.target_word2idx = target_word2idx

        self.show = config.show

        self.hid = []

        self.lr = None
        self.current_lr = config.init_lr
        self.loss = None
        self.step = None
        self.optim = None

        self.sess = sess
        self.log_loss = []
        self.log_perp = []

    def build_memory(self):
        self.global_step = tf.compat.v1.Variable(0, name="global_step")

        self.A = tf.compat.v1.Variable(tf.compat.v1.random_uniform([self.nwords, self.edim], minval=-0.01, maxval=0.01))
        self.ASP = tf.compat.v1.Variable(
            tf.compat.v1.random_uniform([self.pre_trained_target_wt.shape[0], self.edim], minval=-0.01, maxval=0.01))
        self.C = tf.compat.v1.Variable(tf.compat.v1.random_uniform([self.edim, self.edim], minval=-0.01, maxval=0.01))

        self.C_B = tf.compat.v1.Variable(tf.compat.v1.random_uniform([1, self.edim], minval=-0.01, maxval=0.01))
        self.BL_W = tf.compat.v1.Variable(tf.compat.v1.random_uniform([2 * self.edim, 1], minval=-0.01, maxval=0.01))
        self.BL_B = tf.compat.v1.Variable(tf.compat.v1.random_uniform([1, 1], minval=-0.01, maxval=0.01))

        # # Location
        # location_encoding = 1 - tf.compat.v1.truediv(self.time, self.mem_size)
        # location_encoding = tf.compat.v1.cast(location_encoding, tf.compat.v1.float32)
        # location_encoding3dim = tf.compat.v1.tile(tf.compat.v1.expand_dims(location_encoding, 2), [1, 1, self.edim])

        self.Ain_c = tf.compat.v1.nn.embedding_lookup(self.A, self.context)
        # self.Ain = self.Ain_c * location_encoding3dim
        self.Ain = self.Ain_c

        self.ASPin = tf.compat.v1.nn.embedding_lookup(self.ASP, self.input)
        self.ASPout2dim = tf.compat.v1.reshape(self.ASPin, [-1, self.edim])
        self.hid.append(self.ASPout2dim)

        for h in xrange(self.nhop):
            '''
            Bi-linear scoring function for a context word and aspect term
            '''
            self.til_hid = tf.compat.v1.tile(self.hid[-1], [1, self.mem_size])
            self.til_hid3dim = tf.compat.v1.reshape(self.til_hid, [-1, self.mem_size, self.edim])
            self.a_til_concat = tf.compat.v1.concat(axis=2, values=[self.til_hid3dim, self.Ain])
            self.til_bl_wt = tf.compat.v1.tile(self.BL_W, [self.batch_size, 1])
            self.til_bl_3dim = tf.compat.v1.reshape(self.til_bl_wt, [self.batch_size, 2 * self.edim, -1])

            self.att = tf.compat.v1.matmul(self.a_til_concat, self.til_bl_3dim)
            self.til_bl_b = tf.compat.v1.tile(self.BL_B, [self.batch_size, self.mem_size])
            self.til_bl_3dim = tf.compat.v1.reshape(self.til_bl_b, [-1, self.mem_size, 1])
            self.g = tf.compat.v1.nn.tanh(tf.compat.v1.add(self.att, self.til_bl_3dim))
            self.g_2dim = tf.compat.v1.reshape(self.g, [-1, self.mem_size])
            self.masked_g_2dim = tf.compat.v1.add(self.g_2dim, self.mask)
            self.P = tf.compat.v1.nn.softmax(self.masked_g_2dim)
            self.probs3dim = tf.compat.v1.reshape(self.P, [-1, 1, self.mem_size])

            self.Aout = tf.compat.v1.matmul(self.probs3dim, self.Ain)
            self.Aout2dim = tf.compat.v1.reshape(self.Aout, [self.batch_size, self.edim])

            Cout = tf.compat.v1.matmul(self.hid[-1], self.C)
            til_C_B = tf.compat.v1.tile(self.C_B, [self.batch_size, 1])
            Cout_add = tf.compat.v1.add(Cout, til_C_B)
            self.Dout = tf.compat.v1.add(Cout_add, self.Aout2dim)

            if self.lindim == self.edim:
                self.hid.append(self.Dout)
            elif self.lindim == 0:
                self.hid.append(tf.compat.v1.nn.relu(self.Dout))
            else:
                F = tf.compat.v1.slice(self.Dout, [0, 0], [self.batch_size, self.lindim])
                G = tf.compat.v1.slice(self.Dout, [0, self.lindim], [self.batch_size, self.edim - self.lindim])
                K = tf.compat.v1.nn.relu(G)
                self.hid.append(tf.compat.v1.concat(axis=1, values=[F, K]))

    def build_model(self):
        self.build_memory()

        self.W = tf.compat.v1.Variable(tf.compat.v1.random_uniform([self.edim, 3], minval=-0.01, maxval=0.01))
        self.z = tf.compat.v1.matmul(self.hid[-1], self.W)

        self.loss = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(logits=self.z, labels=self.target)

        self.lr = tf.compat.v1.Variable(self.current_lr)
        self.opt = tf.compat.v1.train.AdagradOptimizer(self.lr)

        params = [self.A, self.C, self.C_B, self.W, self.BL_W, self.BL_B]

        self.loss = tf.compat.v1.reduce_sum(self.loss)

        grads_and_vars = self.opt.compute_gradients(self.loss, params)
        clipped_grads_and_vars = [(tf.compat.v1.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                  for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.compat.v1.control_dependencies([inc]):
            self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        tf.compat.v1.global_variables_initializer().run()
        self.saver = tf.train.Saver()

        self.correct_prediction = tf.compat.v1.argmax(self.z, 1)

    def train(self, data):
        source_data, source_loc_data, target_data, target_label = data
        N = int(math.ceil(len(source_data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, 1], dtype=np.int32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size], dtype=np.int32)
        context = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        mask = np.ndarray([self.batch_size, self.mem_size])

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('Train', max=N)

        rand_idx, cur = np.random.permutation(len(source_data)), 0
        for idx in xrange(N):
            if self.show: bar.next()

            context.fill(self.pad_idx)
            time.fill(self.mem_size)
            target.fill(0)
            mask.fill(-1.0 * np.inf)

            for b in xrange(self.batch_size):
                if cur >= len(rand_idx): break
                m = rand_idx[cur]
                x[b][0] = target_data[m]
                target[b] = target_label[m]
                time[b, :len(source_loc_data[m])] = source_loc_data[m]
                context[b, :len(source_data[m])] = source_data[m]
                mask[b, :len(source_data[m])].fill(0)
                cur = cur + 1

            z, _, loss, self.step = self.sess.run([self.z, self.optim,
                                                   self.loss,
                                                   self.global_step],
                                                  feed_dict={
                                                      self.input: x,
                                                      self.time: time,
                                                      self.target: target,
                                                      self.context: context,
                                                      self.mask: mask})

            cost += np.sum(loss)

        if self.show: bar.finish()
        _, train_acc, _, _, _, _ = self.test(data, True)
        return cost / N / self.batch_size, train_acc

    def test(self, data, is_train=False):
        source_data, source_loc_data, target_data, target_label = data
        N = int(math.ceil(len(source_data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, 1], dtype=np.int32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size], dtype=np.int32)
        context = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        mask = np.ndarray([self.batch_size, self.mem_size])

        context.fill(self.pad_idx)

        m, acc = 0, 0
        predicts, labels = [], []
        sentences, targets = [], []
        for i in xrange(N):
            target.fill(0)
            time.fill(self.mem_size)
            context.fill(self.pad_idx)
            mask.fill(-1.0 * np.inf)

            raw_labels = []
            for b in xrange(self.batch_size):
                if m >= len(target_label): break

                x[b][0] = target_data[m]
                target[b] = target_label[m]
                time[b, :len(source_loc_data[m])] = source_loc_data[m]
                context[b, :len(source_data[m])] = source_data[m]
                mask[b, :len(source_data[m])].fill(0)
                raw_labels.append(target_label[m])
                sentences.append(source_data[m])
                targets.append(target_data[m])
                m += 1

            loss = self.sess.run([self.loss],
                                 feed_dict={
                                     self.input: x,
                                     self.time: time,
                                     self.target: target,
                                     self.context: context,
                                     self.mask: mask})
            cost += np.sum(loss)

            predictions = self.sess.run(self.correct_prediction, feed_dict={self.input: x,
                                                                            self.time: time,
                                                                            self.target: target,
                                                                            self.context: context,
                                                                            self.mask: mask})

            # target predict
            target_predict = self.sess.run([self.input], feed_dict={
                self.input: x,
                self.time: time,
                self.target: target,
                self.context: context,
                self.mask: mask})

            sentence_list = []
            # if is_train:
            #     with open('iphone_train.txt', 'r') as data_file:
            #         lines = data_file.read().split('\n')
            #         for line_no in range(0, len(lines) - 1, 3):
            #             sentence = lines[line_no].lower()
            #             sentence_list.append(sentence)
            if not is_train:
                with open('iphone_test.txt', 'r') as data_file:
                    lines = data_file.read().split('\n')
                    for line_no in range(0, len(lines) - 1, 3):
                        sentence = lines[line_no].lower()
                        sentence_list.append(sentence)

            # if is_train:
            #     for b in xrange(self.batch_size):
            #         if raw_labels[b] != predictions[b]:
            #             print(" predict raw_labels : " + str(raw_labels[b]))
            #             print(" predict by system : " + str(predictions[b]))
            #             print(sentence_list[i] + " \n")
            if not is_train:
                for b in xrange(self.batch_size):
                    if raw_labels[b] != predictions[b]:
                        print(" predict raw_labels : " + str(raw_labels[b]))
                        print(" predict by system : " + str(predictions[b]))
                        print(sentence_list[i] + " \n")

            for b in xrange(self.batch_size):
                if b >= len(raw_labels): break
                predicts.append(predictions[b])
                labels.append(raw_labels[b])
                if raw_labels[b] == predictions[b]:
                    acc = acc + 1

        return cost / float(len(source_data)), acc / float(len(source_data)), predicts, labels, sentences, targets

    def run(self, train_data, test_data):

        print('training...')
        self.sess.run(self.A.assign(self.pre_trained_context_wt))
        self.sess.run(self.ASP.assign(self.pre_trained_target_wt))

        best_acc = 0
        train_acc_list = []
        test_acc_list = []
        for idx in xrange(self.nepoch):
            print('epoch ' + str(idx) + '...')

            train_loss, train_acc = self.train(train_data)
            train_acc_list.append(train_loss)

            test_loss, test_acc, predicts, labels, sentences, targets = self.test(test_data)
            test_acc_list.append(test_loss)

            if best_acc < test_acc * 100:
                best_acc = test_acc * 100

            print('train-loss=%.2f;train-acc=%.2f;test-acc=%.2f;' % (train_loss, train_acc * 100, test_acc * 100))
            print('============================================================================')
            self.log_loss.append([train_loss, test_loss])

            if idx % 10 == 0:
                self.saver.save(self.sess,
                                os.path.join(self.checkpoint_dir, "MemN2N.model"),
                                global_step=self.step.astype(int))

        plt.plot(train_acc_list, 'k-', label='Train Set Accuracy')
        plt.plot(test_acc_list, 'r--', label='Test Set Accuracy')
        plt.title('Train and Test Accuracy')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

        print('best-acc=%.2f' % best_acc)

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Trest mode but no checkpoint found")
