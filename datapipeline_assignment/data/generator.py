import os
import numpy as np
#import scipy
import tensorflow as tf
import matplotlib.pyplot as plt
from data import inception_preprocessing

def get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  flower_root = os.path.join(dataset_dir, 'flower_photos')
  directories = []
  class_names = []
  for filename in os.listdir(flower_root):
    path = os.path.join(flower_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)

def generator(filenames, class_names, data_len, batch_size, epoch, sess,
              width=299, height=299, is_training=True):
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    #print (width, height)
    for k in range(epoch):
        i = 0
        while i < data_len:

            if (i + batch_size) > data_len:
                batch_count = data_len - i
            else:
                batch_count = batch_size

            #print (data_len, i, batch_size, batch_count)
            batch_data = np.zeros((batch_count, width, height, 3))
            batch_label = np.zeros((batch_count))

            for j in range(batch_count):
                filename = filenames[i+j]
                image = plt.imread(filename)
                image_tensor = tf.placeholder(dtype=tf.uint8, shape=image.shape, name='image')
                processed_image = sess.run(inception_preprocessing.preprocess_image(image_tensor, height, width, is_training), feed_dict={image_tensor:image})
                batch_data[j] = processed_image

                class_name = os.path.basename(os.path.dirname(filename))
                batch_label[j] = class_names_to_ids[class_name]
                #print (class_name, filename, class_names_to_ids[class_name])
            yield batch_data, batch_label
            i = i + batch_size

