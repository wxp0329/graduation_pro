# encoding=utf-8
from datetime import datetime
import time, os, sys, threading
import numpy as np
import tensorflow as tf
sys.path.append('.')
import vgg16train,vgg_tuning_net,vgg_tuning_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/wangxiaopeng/pool5_tuning_weak_64',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps',100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for CIFAR-10.
        images = tf.placeholder(dtype=tf.float32,shape=[None,14,14,512])
        drop = tf.placeholder(dtype=tf.bool)
        vgg16=vgg16train.Vgg16(vgg16_npy_path=r'/home/wangxiaopeng/tensorflow-vgg_models/vgg16.npy',train_mode=True)
        vgg_feature=vgg16.inference(images)
        # logit=vgg_tuning_net.vgg_tuning_layer(vgg_feature,drop)
        logits = vgg_tuning_net.inference(vgg_feature, drop)

        # Calculate loss.
        loss = vgg_tuning_net.loss(logits)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = vgg_tuning_net.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(),max_to_keep=1000)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess = tf.Session(config=config)
        sess.run(init)

        # mydict = np.load('/home/wangxiaopeng/tensorflow-vgg_models/vgg16.npy',encoding='latin1').item()
        # sess.run(tf.assign(vgg_tuning_net.fen['fc7_weights'], mydict['fc7'][0]))
        # sess.run(tf.assign(vgg_tuning_net.fen['fc7_biases'], mydict['fc7'][1]))

        mynet_params= np.load('../haveL1_checkPoint_64_20000ckpt.npy').item()
        for i in mynet_params:
            if 'w' in i or 'b' in i:
                sess.run(tf.assign(vgg_tuning_net.fen[i],mynet_params[i]))
        # saver.restore(sess=sess,
        #               save_path=r'/home/wangxiaopeng/pool5_tuning_weak/model.ckpt-15000')

        # print('DSH_net.fen[fc7_weights] - mydict[fc7_weights] = {}'.format(
        #     np.sum(np.subtract(sess.run(Three_net_enforce.fen['fc7_weights']), mydict['fc7'][0]))))
        # print()
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        # summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
        #                                         graph_def=sess.get_default)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=g)

        input = vgg_tuning_input.InputUtil()

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
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if step % 500 == 0:
                summary_str = sess.run(summary_op, feed_dict={images: datas,drop:True})
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
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
