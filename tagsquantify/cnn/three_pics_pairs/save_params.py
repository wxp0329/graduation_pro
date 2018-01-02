# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
import shutil
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tagsquantify.cnn.three_pics_pairs import Three_net_enforce

np.set_printoptions(threshold=np.inf)



def train():
    """Train CIFAR-10 for a number of steps. """
    with tf.Graph().as_default() as g:
        images = tf.placeholder(dtype=tf.float32,
                                shape=[None, 4096])
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction =0.
        drop = tf.placeholder(dtype=tf.bool)
        logits = Three_net_enforce.inference(images, drop)
        saver = tf.train.Saver(tf.all_variables())

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        saver.restore(sess=sess,
                      save_path=r'F:\NUS_dataset\graduate_data\haveL1_checkPoint_64\model.ckpt-20000')

        mydict = dict()
        for i in Three_net_enforce.fen:
            if 'w' in i or 'b' in i:
                mydict[i] = sess.run(Three_net_enforce.fen[i], feed_dict={images: np.random.rand(1, 4096)})

        np.save(r'F:\NUS_dataset\graduate_data\haveL1_checkPoint_64_20000ckpt', mydict)


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    tf.app.run()
