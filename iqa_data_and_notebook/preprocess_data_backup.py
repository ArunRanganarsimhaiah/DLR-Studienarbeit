#!/usr/bin/env python
# coding: utf-8

# In[2]:
import glob
import ntpath
import os
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET

from tensorflow.keras import layers



import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import UpSampling2D,BatchNormalization,Activation,Conv2D,LeakyReLU,Dropout,Dense,Flatten,Reshape
from tensorflow.keras.initializers import RandomNormal as RN
import time
from IPython import display
from iqa_lib import *


# In[9]:


# create .tfrecord dataset from two files with labeled data
#from Desktop.iqa_data_and_notebook.iqa_lib import *


ds = tf.data.TFRecordDataset(['./iqa_labeled_20180212.tfrecord', './iqa_labeled_20180216.tfrecord'])
ds = ds.map(preprocess)


# In[5]:


# create dataframe from .csv files with bounding box information
df = pd.read_csv('./iqa_labeled_20180212.csv')
df = df.append(pd.read_csv('./iqa_labeled_20180216.csv'))
df.head()


# In[7]:


print(categories)
df['label'] = df['class'].map(lambda x: categories[x])
df.head()


# In[8]:

img_str=[]
label_str=[]
box_cols_abs = ['ymin', 'xmin', 'ymax', 'xmax']
box_cols_rel = ['ymin_rel', 'xmin_rel', 'ymax_rel', 'xmax_rel']
colors = np.array([[0., 1., 0.]])
for item in ds:
    img = item[0]
    filename = item[1].numpy().decode('utf-8')
    idx = df['filename'] == filename
    if any(idx):
        boxes_rel = df.loc[idx, box_cols_rel].to_numpy()
        img_boxes = tf.image.draw_bounding_boxes(img[tf.newaxis], np.expand_dims(boxes_rel, axis=0), colors)
        #plt.figure(figsize=(20, 10))
        plt.imshow(np.squeeze(img_boxes))
        plt.axis(False)
        #plt.show()

        boxes_abs = df.loc[idx, box_cols_abs].to_numpy()
        classes = df.loc[idx, 'class'].to_numpy()
        for i in range(boxes_abs.shape[0]):
            #plt.figure()
            img_crop = tf.image.crop_to_bounding_box(img, boxes_abs[i, 0], boxes_abs[i, 1], boxes_abs[i, 2] - boxes_abs[i, 0], boxes_abs[i, 3] - boxes_abs[i, 1])
            img_crop = tf.image.resize(img_crop, size=(128, 128))
            img_str.append(img_crop)
            label_str.append(classes[i])
            plt.imshow(img_crop)
            plt.axis(False)
            plt.title(classes[i])
            #plt.show()
    else:
        #plt.figure(figsize=(20, 10))
        plt.imshow(np.squeeze(img))
        plt.axis(False)
        #plt.show()

        for i in range(3):
            #plt.figure()
            max_dim = tf.cast(tf.reduce_min(tf.shape(img)[:2]), tf.int64)
            crop_size = tf.random.uniform(shape=[2], minval=128, maxval=max_dim, dtype=tf.int64)
            img_crop = tf.image.random_crop(img, size=(crop_size[0], crop_size[1], 3))
            img_crop = tf.image.resize(img_crop, (128, 128))
            img_str.append(img_crop)
            label_str.append('none')
            plt.imshow(img_crop)
            plt.axis(False)
            plt.title('none')
            #plt.show()



# ### Work program for next week
# 
# ##### 1. Get familiar with this notebook and the dataset
# 
# The data provided here consists of all available labeled images as well as a subset of the unlabeled images. Coordinates of bounding boxes and associated class labels can be found in the dataframe loaded from the two csv files.
# 
# ##### 2. Train a variational autoencoder and / or a gan
# 
# Use 128 x 128 snippets for this. The above code shows how to extract snippets with and without errors from the dataset. I would recommend to create a new dataset containing only smaller snippets because loading the larger images from disk requires quite some time. Moreover, it would probably make sense to save the images in the new dataset in grayscale format (1 channel instead of 3).
# 
# ##### 3. Generate some synthetical images using your readily trained model
# 
# ### You do not need to finish this until Tuesday. Take your time and enjoy the weekend! :-)


#img_str=tf.convert_to_tensor(img_str)



#label_str=tf.convert_to_tensor(label_str)

#img_str=img_str.reshape(img_str.shape[0],128,128,1).astype('float32')
img_str=np.array(img_str)
label_str=np.array(label_str)
img_str=tf.image.rgb_to_grayscale(img_str)
print(img_str.shape)
print(label_str.shape)
img_str=(img_str-127.5) /127.5

BUFFER_SIZE = 345
BATCH_SIZE = 32

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(img_str).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

latent_dim=100
num_classes=len(categories)
ki_mean=0.0
ki_stddev=0.02
ks=5
mom=.8
channels=1
drop=0.0
alp=.2
def make_generator():  # Generator mit 5 Conv Schichten; Verkettung der Eingabe (Rauschen und Label)

    model = tf.keras.Sequential()

    model.add(Dense(256 * 4 * 4, kernel_initializer=RN(mean=ki_mean, stddev=ki_stddev), activation="relu",
                    input_dim=latent_dim))
    model.add(Reshape((4, 4, 256)))

    model.add(UpSampling2D())  # 8*8
    model.add(Conv2D(128, kernel_size=ks, kernel_initializer=RN(mean=ki_mean, stddev=ki_stddev), padding="same"))
    model.add(BatchNormalization(momentum=mom))
    model.add(Activation("relu"))

    model.add(UpSampling2D())  # 16*16
    model.add(Conv2D(64, kernel_size=ks, kernel_initializer=RN(mean=ki_mean, stddev=ki_stddev), padding="same"))
    model.add(BatchNormalization(momentum=mom))
    model.add(Activation("relu"))

    model.add(UpSampling2D())  # 32*32
    model.add(Conv2D(32, kernel_size=ks, kernel_initializer=RN(mean=ki_mean, stddev=ki_stddev), padding="same"))
    model.add(BatchNormalization(momentum=mom))
    model.add(Activation("relu"))

    model.add(UpSampling2D())  # 64*64
    model.add(Conv2D(16, kernel_size=ks, kernel_initializer=RN(mean=ki_mean, stddev=ki_stddev), padding="same"))
    model.add(BatchNormalization(momentum=mom))
    model.add(Activation("relu"))

    model.add(UpSampling2D())  # 128*128
    # model.add(Conv2D(8, kernel_size=ks, kernel_initializer=RN(mean=ki_mean, stddev=ki_stddev),padding="same"))
    # model.add(BatchNormalization(momentum=mom))
    # model.add(Activation("relu"))
    # 5CONV layer
    model.add(
        Conv2D(channels, kernel_size=ks, kernel_initializer=RN(mean=ki_mean, stddev=ki_stddev), padding="same"))
    model.add(Activation("tanh"))

    return model


def make_discriminator():  # Generator mit 5 Conv Schichten; Verkettung der Eingabe (Bild und Labeltensor)

    model = tf.keras.Sequential()  # 128*128*7

    model.add(Conv2D(16, kernel_size=ks, kernel_initializer=RN(mean=ki_mean, stddev=ki_stddev), strides=2,
                     input_shape=([128,128,1]), padding="same"))
    model.add(BatchNormalization(momentum=mom))
    model.add(LeakyReLU(alpha=alp))  # 64*64
    model.add(Dropout(drop))

    model.add(
        Conv2D(32, kernel_size=ks, kernel_initializer=RN(mean=ki_mean, stddev=ki_stddev), strides=2, padding="same"))
    model.add(BatchNormalization(momentum=mom))
    model.add(LeakyReLU(alpha=alp))  # 32*32
    model.add(Dropout(drop))

    model.add(
        Conv2D(64, kernel_size=ks, kernel_initializer=RN(mean=ki_mean, stddev=ki_stddev), strides=2, padding="same"))
    model.add(BatchNormalization(momentum=mom))
    model.add(LeakyReLU(alpha=alp))  # 16*16
    model.add(Dropout(drop))

    model.add(
        Conv2D(128, kernel_size=ks, kernel_initializer=RN(mean=ki_mean, stddev=ki_stddev), strides=2, padding="same"))
    model.add(BatchNormalization(momentum=mom))
    model.add(LeakyReLU(alpha=alp))  # 8*8
    model.add(Dropout(drop))

    model.add(
        Conv2D(256, kernel_size=ks, kernel_initializer=RN(mean=ki_mean, stddev=ki_stddev), strides=2, padding="same"))
    model.add(BatchNormalization(momentum=mom))
    model.add(LeakyReLU(alpha=alp))  # 4*4
    model.add(Dropout(drop))

    # Kernel 5x5 zu groß: deshalb Flatten und Dense
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16
generator=make_generator()
discriminator=make_discriminator()

# You will reuse this seed overtime (so it's easier)


# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return generated_images


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      gen_img=train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 10 epochs
    if (epoch + 1) % 10 == 0:
        filename = 'generator_e%03d.h5' % (epoch + 1)
        generator.save('./GAN/models/'+filename)
  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('./GAN/whole_data/image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

train(train_dataset, EPOCHS)