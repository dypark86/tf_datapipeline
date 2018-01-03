import os
import time
from datetime import datetime

import tensorflow as tf
import numpy as np
from data import generator
import random

from model.slim_inception_resnet import inception_resnet_v2_arg_scope, inception_resnet_v2
from model.slim_vgg import vgg_19, vgg_arg_scope

#os.environ["CUDA_VISIBLE_DEVICES"] = '3'

slim = tf.contrib.slim

train_num = 3320
val_num = 350

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/root/data/tf_record_test',
                    'Directory with the flower data.')
flags.DEFINE_string('ckpt_name', '/root/data/tf_record_test/check_point_2/slim_vgg',
                    'Directory with the ckpt data save.')
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_integer('epoch_num', 2, 'Num of epoch')
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
                 val_step_num, val_gen, total_loss, acc):

    val_loss_list = []
    val_acc_list = []
    for j in range(val_step_num):
        val_images_batch, val_labels_batch = next(val_gen)
        print ("val batch shape", np.shape(val_images_batch), np.shape(val_labels_batch))

        loss_val, acc_val = sess.run([total_loss, acc],
                                     feed_dict={images: val_images_batch, labels: val_labels_batch,
                                                is_training_tensor: False})
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

    filenames, class_names = generator.get_filenames_and_classes(FLAGS.data_dir)
    random.shuffle(filenames)
    training_filenames = filenames[val_num:]
    validation_filenames = filenames[:val_num]
    train_data_len = len(training_filenames)
    val_data_len = len(validation_filenames)

    global_step = tf.contrib.framework.get_or_create_global_step()

    #create placeholder for image, labels, is_training
    images = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, 3))
    labels = tf.placeholder(tf.int32, shape=(None, ))
    is_training_tensor = tf.placeholder(tf.bool)

    #create vgg_19 model
    arg_scope = vgg_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = vgg_19(images, num_classes = FLAGS.class_num,
                                         is_training=is_training_tensor)

    #create loss tensor
    one_hot_labels = tf.one_hot(labels, FLAGS.class_num)
    total_loss = losses(logits, labels)

    #create acc tensor
    predictions = tf.argmax(logits, 1)
    ground_truth = tf.argmax(one_hot_labels,1)
    acc = tf.reduce_mean(tf.cast(tf.equal(predictions, ground_truth), "float"))
    tf.summary.scalar('acc', acc)

    #optimizer setting
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_op = optimizer.minimize(total_loss, global_step=global_step)

    #summary merge
    merged = tf.summary.merge_all()

    #create log dir and file write
    date_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_dir = FLAGS.log_dir + "/" + date_time_str
    file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

    #cretae summary, saver
    duration_summary = tf.Summary()
    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        # create data generator
        train_gen = generator.generator(training_filenames, class_names, train_data_len, FLAGS.batch_size, FLAGS.epoch_num, sess, FLAGS.image_size,
                                        FLAGS.image_size, is_training=True)
        val_gen = generator.generator(validation_filenames, class_names, val_data_len, FLAGS.batch_size, FLAGS.epoch_num, sess, FLAGS.image_size,
                                      FLAGS.image_size, is_training=False)

        #calc train_step_num with batch size and epoch num
        train_step_num = int(np.ceil((train_num*FLAGS.epoch_num)/FLAGS.batch_size))
        #calc 1 epoch val_step_num with batch size
        val_step_num = int(np.ceil(val_num/FLAGS.batch_size))
        #calc 1 epoch train step num
        epoch_step_num = int(np.ceil(train_num/FLAGS.batch_size))
        epoch_num = 0
        train_loss_list = []
        train_acc_list = []
        step = 0

        print ("step num", train_step_num, val_step_num, epoch_step_num)

        for i in range(train_step_num):
            start_time = time.time()
            train_images_batch, train_labels_batch = next(train_gen)

            _, loss_train, acc_train = sess.run([train_op, total_loss, acc],
                                                feed_dict={images: train_images_batch, labels: train_labels_batch,
                                                           is_training_tensor: True})

            train_loss_list.append(loss_train)
            train_acc_list.append(acc_train)
            duration = time.time() - start_time
            step += 1
            print ("train step", step, "shape", np.shape(train_images_batch), np.shape(train_labels_batch), "duration ", duration)

            # each epoch calculate validate loss print loss
            if (step+1) % epoch_step_num == 0:
                avg_train_loss = np.mean(train_loss_list)
                avg_train_acc = np.mean(train_acc_list)
                train_loss_list = []
                train_acc_list = []

                avg_val_loss, avg_val_acc = validate_run(sess, images, labels, is_training_tensor,
                                                         val_step_num, val_gen, total_loss, acc)

                print ("epoch: ", epoch_num, "train:", avg_train_loss, avg_train_acc,
                       "validate: ", avg_val_loss, avg_val_acc)

                epoch_num += 1
                saver.save(sess, FLAGS.ckpt_name)

            # ecah 10 step save log
            if step % 10 == 0:
                summary = sess.run(merged, feed_dict={images: train_images_batch, labels: train_labels_batch,
                                                      is_training_tensor: True})
                duration_summary.value.add(tag="step_duration", simple_value=duration)
                file_writer.add_summary(duration_summary, step)
                file_writer.add_summary(summary, step)

        print ("last step:", step, train_step_num, FLAGS.epoch_num)

        file_writer.close()

if __name__ == '__main__':
    tf.app.run()