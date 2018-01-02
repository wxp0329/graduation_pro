# encoding=utf-8

import re
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 126,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the NUS data set.
LOSS_LAMBDA = 6.
L1_param=1.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1528431  # contain pics num compute with 219388_2003_indexes_0.2_three_pair.dat
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 2.  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 1e-10 # Learning rate decay factor.

INITIAL_LEARNING_RATE = 1e-10# Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.q
TOWER_NAME = 'tower'


def _activation_summary(x):
    #     """Helper to create summaries for activations.
    #
    #     Creates a summary that provides a histogram of activations.
    #     Creates a summary that measures the sparsity of activations.
    #
    #     Args:
    #       x: Tensor
    #     Returns:
    #       nothing
    #     """
    #     # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    #     # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)

    tf.summary.histogram(tensor_name + '/activations', x)

    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

fen=dict()
def vgg_tuning_layer(feature, dropout):

    with tf.variable_scope('fc7') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        # feature = tf.reshape(feature, [-1, FLAGS.batch_size])
        weights = _variable_with_weight_decay('weights', shape=[4096,4096],
                                              stddev=0.1, wd=0.004)
        biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.))
        local3 = tf.nn.tanh(tf.matmul(feature, weights) + biases, name=scope.name)
        local3 = tf.cond(dropout, lambda: tf.nn.dropout(local3, keep_prob=0.5), lambda: local3)
        fen['fc7_weights']=weights
        fen['fc7_biases']=biases
        _activation_summary(local3)
    return local3

def inference(images, dropout):
    """Build the NUS_dataset model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # with tf.variable_scope('flatten') as scope:
    #     shape = int(np.prod(images.get_shape()[1:]))
    #     weights = _variable_with_weight_decay('weights', shape=[shape, 4096],
    #                                           stddev=0.1, wd=0.004)
    #     biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.))
    #     images=tf.reshape(images,[-1,shape])
    #     images=tf.nn.tanh(tf.matmul(images, weights) + biases, name=scope.name)
    # # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.

        weights = _variable_with_weight_decay('weights', shape=[4096, 1024],
                                              stddev=0.1, wd=0.004)
        biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.))
        local3 = tf.nn.tanh(tf.matmul(images, weights) + biases, name=scope.name)
        local3 = tf.cond(dropout, lambda: tf.nn.dropout(local3, keep_prob=0.5), lambda: local3)
        fen['local3_w']=weights
        fen['local3_b']=biases
        _activation_summary(local3)
    with tf.variable_scope('local3_1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.

        weights = _variable_with_weight_decay('weights', shape=[1024, 512],
                                              stddev=0.1, wd=0.004)
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.))
        local3 = tf.nn.tanh(tf.matmul(local3, weights) + biases, name=scope.name)
        # dropout
        local3 = tf.cond(dropout, lambda: tf.nn.dropout(local3, keep_prob=0.5), lambda: local3)
        fen['local3_1_w'] = weights
        fen['local3_1_b'] = biases
        _activation_summary(local3)
    with tf.variable_scope('local4') as scope:
        # Move everything into depth so we can perform a single matrix multiply.

        weights = _variable_with_weight_decay('weights', shape=[512, 64],
                                              stddev=0.1, wd=0.004)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.))
        hash_in = tf.nn.bias_add(tf.matmul(local3, weights), biases, name=scope.name + '_hash_in')
        local4 = tf.nn.tanh(hash_in)
        fen['local4_w'] = weights
        fen['local4_b'] = biases
        _activation_summary(local4)

    return local4



def loss(imgs):
    """

    :param true_files: 包含tensors的一维数组
    :param false_files: 包含tensors的一维数组
    :return:
    """
    with tf.variable_scope('loss') as scope:
        print('computer losss........................')
        regular = tf.contrib.layers.l1_regularizer(L1_param)(imgs) / FLAGS.batch_size
        i_s = imgs[:int(FLAGS.batch_size / 3)]  # 前三分之一是i
        j_s = imgs[int(FLAGS.batch_size / 3):int(FLAGS.batch_size * 2 / 3)]  # 中间三分之一是j
        k_s = imgs[int(FLAGS.batch_size * 2 / 3):]  # 最后三分之一是k

        i_k_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(i_s, k_s)), axis=1))  # 一列
        i_j_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(i_s, j_s)), axis=1))  # 一列

        loss = tf.reduce_mean(tf.maximum(0., tf.add(tf.subtract(LOSS_LAMBDA, i_k_dist), i_j_dist)))+regular
    tf.add_to_collection('losses', loss)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        # opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.MomentumOptimizer(lr,momentum=0.9)
        grads = opt.compute_gradients(total_loss)
    #
    # # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
    # return opt
