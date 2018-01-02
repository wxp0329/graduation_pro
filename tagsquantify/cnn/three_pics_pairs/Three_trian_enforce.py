# encoding=utf-8
from datetime import datetime
import time, os, sys, threading
import numpy as np
import tensorflow as tf

from tagsquantify.cnn.three_pics_pairs import vgg_feature_reader, Three_net_enforce, vggTransitionPic, \
    Three_input_mem
from tagsquantify.cnn.three_pics_pairs.linux_files import explore
from tagsquantify.cnn.vgg_nets.vgg16train import Vgg16

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', r'F:\NUS_dataset\graduate_data\lmd12_checkPoint_64',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 22000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for CIFAR-10.
        images = tf.placeholder(dtype=tf.float32,shape=[None,4096])
        drop = tf.placeholder(dtype=tf.bool)
        # vgg16=Vgg16(vgg16_npy_path=r'F:\NUS_dataset\tensorflow-vgg_models\vgg16.npy',train_mode=True)
        # logit=vgg16.inference(images)
        # logit=Three_net_enforce.vgg_tuning_layer(images,drop)
        logits = Three_net_enforce.inference(images, drop)

        # Calculate loss.
        loss = Three_net_enforce.loss(logits)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = Three_net_enforce.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(),max_to_keep=100)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        sess = tf.Session(config=config)
        sess.run(init)

        # mydict = np.load(r'F:\NUS_dataset\tensorflow-vgg_models\vgg16.npy',encoding='latin1').item()
        # sess.run(tf.assign(Three_net_enforce.fen['fc7_weights'], mydict['fc7'][0]))
        # sess.run(tf.assign(Three_net_enforce.fen['fc7_biases'], mydict['fc7'][1]))

        # mynet_params= np.load(r'F:\NUS_dataset\graduate_data\alexnet_checkPointdir_fc7_10400.npy').item()
        # for i in mynet_params:
        #     if 'w' in i or 'b' in i:
        #         sess.run(tf.assign(Three_net_enforce.fen[i],mynet_params[i]))
        # saver.restore(sess=sess,
        #               save_path=r'F:\NUS_dataset\graduate_data\alexnet_checkPointdir_fc7\model.ckpt-104000')

        # print('DSH_net.fen[fc7_weights] - mydict[fc7_weights] = {}'.format(
        #     np.sum(np.subtract(sess.run(Three_net_enforce.fen['fc7_weights']), mydict['fc7'][0]))))
        # print()
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        # summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
        #                                         graph_def=sess.get_default)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=g)

        input = Three_input_mem.InputUtil()

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            datas = input.next_batch()
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict={images: datas,drop:True})
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)------ Lmd12')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict={images: datas,drop:True})
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 2000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                # if step % 200 == 0:
                #     small = sess.run(logits, feed_dict={
                #         images: np.load(r'F:\NUS_dataset\teacher_project_data\img_transfer_mat\vgg16_fc7_121.npy'),
                #         drop: False})
                #     big = sess.run(logits, feed_dict={
                #         images: np.load(r'F:\NUS_dataset\teacher_project_data\img_transfer_mat\vgg16_fc7_1000.npy'),
                #         drop: False})
                #     acc = threading.Thread(target=explore.eval_acc, args=(step, small, big,FLAGS.train_dir))
                #     acc.setDaemon(True)
                #     acc.start()


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
