import os
import time
from datetime import datetime

import tensorflow as tf
import numpy as np
from data import data_queue

from model.slim_inception_resnet import inception_resnet_v2_arg_scope, inception_resnet_v2
from model.slim_vgg import vgg_19, vgg_arg_scope

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

slim = tf.contrib.slim

train_num = 3320
val_num = 350

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/root/data/tf_record_test',
                    'Directory with the flower data.')
flags.DEFINE_string('ckpt_name', '/root/data/tf_record_test/check_point_2/slim_vgg',
                    'Directory with the ckpt data.')
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_integer('epoch_num', 100, 'Num of epoch')
flags.DEFINE_integer('image_size', 224,
                     'Default image size')
flags.DEFINE_integer('class_num', 5,
                     'Num of class')
flags.DEFINE_string('log_dir', '/root/data/tf_record_test/log/train',
                    'Directory with the log data.')
flags.DEFINE_float('learning_rate', 0.01,
                   'The minimal end learning rate used by a polynomial decay learning rate.')

FLAGS = flags.FLAGS

def validate_run(sess, images, labels, is_training_tensor,
                 val_step_num, val_images, val_labels,
                 total_loss, acc):
    val_loss_list = []
    val_acc_list = []
    for j in range(val_step_num):
        val_images_batch, val_labels_batch = sess.run([val_images, val_labels])
        print ("val batch shape", np.shape(val_images_batch), np.shape(val_labels_batch))

        loss_val, acc_val = sess.run([total_loss, acc],
                                     feed_dict={images: val_images_batch, labels: val_labels_batch,
                                                is_training_tensor:False})
        val_loss_list.append(loss_val)
        val_acc_list.append(acc_val)

    avg_val_loss = np.mean(val_loss_list)
    avg_val_acc = np.mean(val_acc_list)

    return avg_val_loss, avg_val_acc

def losses(logits, one_hot_labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.summary.scalar('loss', cross_entropy_mean)
    return cross_entropy_mean

def main(_):
    # Create global_step
    global_step = tf.contrib.framework.get_or_create_global_step()

    train_images, train_labels =  data_queue.inputs("train", batch_size=FLAGS.batch_size,
                                                    dataset_dir = FLAGS.data_dir, num_epochs=FLAGS.epoch_num,
                                                    width=FLAGS.image_size, height=FLAGS.image_size, is_training=True)

    val_images, val_labels =  data_queue.inputs("validation", batch_size=FLAGS.batch_size,
                                                    dataset_dir = FLAGS.data_dir, num_epochs=FLAGS.epoch_num,
                                                    width=FLAGS.image_size, height=FLAGS.image_size, is_training=False)

    images = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, 3))
    labels = tf.placeholder(tf.int32, shape=(None, ))
    is_training_tensor = tf.placeholder(tf.bool)

    #create vgg+19
    arg_scope = inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_resnet_v2(images, num_classes = FLAGS.class_num,
                                         is_training=is_training_tensor)

    one_hot_labels = tf.one_hot(labels, FLAGS.class_num)
    total_loss = losses(logits, labels)


    predictions = tf.argmax(logits, 1)
    ground_truth = tf.argmax(one_hot_labels,1)
    acc = tf.reduce_mean(tf.cast(tf.equal(predictions, ground_truth), "float"))
    tf.summary.scalar('acc', acc)

    #optimizer setting
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_op = optimizer.minimize(total_loss, global_step=global_step)

    merged = tf.summary.merge_all()

    date_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_dir = FLAGS.log_dir + "/" + date_time_str
    file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

    duration_summary = tf.Summary()
    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_step_num = int(np.ceil((train_num*FLAGS.epoch_num)/FLAGS.batch_size))
        val_step_num = int(np.ceil(val_num/FLAGS.batch_size))
        epoch_step_num = int(np.ceil(train_num/FLAGS.batch_size))
        epoch_num = 0
        print ("threads",threads)
        print ("step num", train_step_num, val_step_num, epoch_step_num)
        train_loss_list = []
        train_acc_list = []

        step = 0

        try:
            for i in range(train_step_num):
                start_time = time.time()
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
                _, loss_train, acc_train = sess.run([train_op, total_loss, acc],
                                                    feed_dict={images: train_images_batch, labels: train_labels_batch,
                                                               is_training_tensor: True})

                train_loss_list.append(loss_train)
                train_acc_list.append(acc_train)
                duration = time.time() - start_time
                step += 1
                print ("train step", step, "shape", np.shape(train_images_batch), np.shape(train_labels_batch), "duration ", duration)

                if (step+1) % epoch_step_num == 0:
                    avg_train_loss = np.mean(train_loss_list)
                    avg_train_acc = np.mean(train_acc_list)
                    train_loss_list = []
                    train_acc_list = []

                    avg_val_loss, avg_val_acc = validate_run(sess, images, labels, is_training_tensor,
                                                             val_step_num, val_images, val_labels,
                                                             total_loss, acc)

                    print ("epoch: ", epoch_num, "train:", avg_train_loss, avg_train_acc,
                           "validate: ", avg_val_loss, avg_val_acc)

                    epoch_num += 1
                    saver.save(sess, FLAGS.ckpt_name)

                if step % 10 == 0:
                    summary = sess.run(merged, feed_dict={images: train_images_batch, labels: train_labels_batch,
                                                          is_training_tensor: True})
                    duration_summary.value.add(tag="step_duration", simple_value=duration)
                    file_writer.add_summary(duration_summary, step)
                    file_writer.add_summary(summary, step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            print ("last step:", step, train_step_num, FLAGS.epoch_num)
            coord.request_stop()

        coord.join(threads)
        file_writer.close()

if __name__ == '__main__':
    tf.app.run()