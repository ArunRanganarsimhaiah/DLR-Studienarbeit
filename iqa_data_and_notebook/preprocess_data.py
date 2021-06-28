#!/usr/bin/env python
# coding: utf-8

# In[2]:
import glob
import ntpath
import os

import cv2
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET

from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import UpSampling2D,BatchNormalization,Activation,Conv2D,LeakyReLU,Dropout,Dense,Flatten,Reshape,Input,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal as RN
import time
from IPython import display
from tensorflow.python.keras.utils.np_utils import to_categorical
#from iqa_lib import *
#from Desktop.iqa_data_and_notebook.iqa_lib import *




categories = {'none': 0, 'wrinkle': 1, 'twist': 2, 'foreign material': 3, 'overlap': 4, 'gap': 5}
defects=["0_None", "1_Wrinkle", "2_Twist", "3_Foreign_Body", "4_Overlap", "5_Gap"]
def preprocess(example):
    features = {'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'filename': tf.io.FixedLenFeature([], tf.string, default_value='')}
    parsed_example = tf.io.parse_single_example(example, features)
    image = tf.io.decode_png(parsed_example['image'], channels=3, dtype=tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, parsed_example['filename']

def save_dataset(img, classes, filename,i):

  path_log = "./dataset_zoom/"+classes
  try:
      os.mkdir(path_log)
  except OSError as error:
    print(error)


  img = tf.image.rgb_to_grayscale(img)
  img = np.squeeze(img)
  # Rescale images -1,1 => 0,255
  img = 127.5 * img + 127.5
  cv2.imwrite(os.path.join(path_log, "{}_{}_zoom.png".format(filename,i)), img)


# create .tfrecord dataset from two files with labeled data
ds = tf.data.TFRecordDataset(['./iqa_labeled_20180212.tfrecord', './iqa_labeled_20180216.tfrecord'])
ds = ds.map(preprocess)
# create dataframe from .csv files with bounding box information
df = pd.read_csv('./iqa_labeled_20180212.csv')
df = df.append(pd.read_csv('./iqa_labeled_20180216.csv'))
df.head()
print(categories)
df['label'] = df['class'].map(lambda x: categories[x])
df.head()
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

        boxes_abs[:,0]=np.subtract(boxes_abs[:,0],5)
        boxes_abs[:,1]=np.subtract(boxes_abs[:,1],5)
        boxes_abs[:,2]=np.add(boxes_abs[:,2],5)
        boxes_abs[:,3]=np.add(boxes_abs[:,3],5)
        classes = df.loc[idx, 'class'].to_numpy()
        label=df.loc[idx, 'label'].to_numpy()
        for i in range(boxes_abs.shape[0]):
            #plt.figure()
            img_crop = tf.image.crop_to_bounding_box(img, boxes_abs[i, 0], boxes_abs[i, 1], (boxes_abs[i, 2] - boxes_abs[i, 0]), (boxes_abs[i, 3] - boxes_abs[i, 1]))
            img_crop = tf.image.resize(img_crop, size=(128, 128))
            save_dataset(img_crop,classes[i],filename,i)
            img_str.append(img_crop)
            label_str.append(label[i])
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
            save_dataset(img_crop,'None',filename,i)
            img_str.append(img_crop)
            label_str.append('0')
            plt.imshow(img_crop)
            plt.axis(False)
            plt.title('none')
            #plt.show()





img_str = np.asarray(img_str)
label_str = np.asarray(label_str)
img_str = tf.image.rgb_to_grayscale(img_str)



print(img_str.shape)
print(label_str.shape)
img_str = (img_str - 127.5) / 127.5

BUFFER_SIZE = 345
BATCH_SIZE = 32
EPOCHS = 20
train_dataset = tf.data.Dataset.from_tensor_slices(img_str)#.shuffle(BUFFER_SIZE).batch(
        #BATCH_SIZE)

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



class GAN():
        def __init__(self):

           self.latent_dim=100
           self.num_classes=len(categories)
           self.ki_mean = 0.0
           self.ki_stddev = 0.02
           self.ks = 5
           self.mom = .8
           self.channels = 1
           self.drop = 0.0
           self.alp = .2
           self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
           self.generator = self.make_generator()
           self.discriminator = self.make_discriminator()

           self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
           self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

           self.noise_dim = 100
           self.num_examples_to_generate = 16

           self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])


        def make_generator(self):  # Generator mit 5 Conv Schichten; Verkettung der Eingabe (Rauschen und Label)

            model = tf.keras.Sequential()

            model.add(Dense(256 * 4 * 4, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), activation="relu",
                    input_dim=self.latent_dim))
            model.add(Reshape((4, 4, 256)))

            model.add(UpSampling2D())  # 8*8
            model.add(Conv2D(128, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(Activation("relu"))

            model.add(UpSampling2D())  # 16*16
            model.add(Conv2D(64, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(Activation("relu"))

            model.add(UpSampling2D())  # 32*32
            model.add(Conv2D(32, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(Activation("relu"))

            model.add(UpSampling2D())  # 64*64
            model.add(Conv2D(16, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(Activation("relu"))

            model.add(UpSampling2D())  # 128*128
    # model.add(Conv2D(8, kernel_size=ks, kernel_initializer=RN(mean=ki_mean, stddev=ki_stddev),padding="same"))
    # model.add(BatchNormalization(momentum=mom))
    # model.add(Activation("relu"))
    # 5CONV layer
            model.add(
                        Conv2D(self.channels, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), padding="same"))
            model.add(Activation("tanh"))

            return model


        def make_discriminator(self):  # Generator mit 5 Conv Schichten; Verkettung der Eingabe (Bild und Labeltensor)

            model = tf.keras.Sequential()  # 128*128*7

            model.add(Conv2D(16, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), strides=2,
                     input_shape=([128,128,1]), padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(LeakyReLU(alpha=self.alp))  # 64*64
            model.add(Dropout(self.drop))

            model.add(
        Conv2D(32, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), strides=2, padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(LeakyReLU(alpha=self.alp))  # 32*32
            model.add(Dropout(self.drop))

            model.add(
        Conv2D(64, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), strides=2, padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(LeakyReLU(alpha=self.alp))  # 16*16
            model.add(Dropout(self.drop))

            model.add(
        Conv2D(128, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), strides=2, padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(LeakyReLU(alpha=self.alp))  # 8*8
            model.add(Dropout(self.drop))

            model.add(
        Conv2D(256, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), strides=2, padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(LeakyReLU(alpha=self.alp))  # 4*4
            model.add(Dropout(self.drop))

    # Kernel 5x5 zu groß: deshalb Flatten und Dense
            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid'))

            return model


        def discriminator_loss(self,real_output, fake_output):
            real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
            total_loss = real_loss + fake_loss
            return total_loss

        def generator_loss(self,fake_output):
            return self.cross_entropy(tf.ones_like(fake_output), fake_output)




# You will reuse this seed overtime (so it's easier)


# to visualize progress in the animated GIF)


# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
        @tf.function
        def train_step(self, images):
            noise = tf.random.normal([BATCH_SIZE, self.noise_dim])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = self.generator(noise, training=True)

                real_output = self.discriminator(images, training=True)
                fake_output = self.discriminator(generated_images, training=True)

                gen_loss = self.generator_loss(fake_output)
                disc_loss = self.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))




        def train(self,dataset, epochs):
            for epoch in range(epochs):
                start = time.time()

                for image_batch in dataset:
                    gen_img= self.train_step(image_batch)


                    display.clear_output(wait=True)
                    self.generate_and_save_images(self.generator,
                                  epoch + 1,
                                  self.seed)

    # Save the model every 10 epochs
                if (epoch + 1) % 10 == 0:
                    filename = 'generator_e%03d.h5' % (epoch + 1)
                    self.generator.save('./GAN/models/'+filename)
  # Generate after the final epoch
                display.clear_output(wait=True)
                self.generate_and_save_images(self.generator,
                           epochs,
                           self.seed)


        def generate_and_save_images(self,model, epoch, test_input):
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

DL = []
ACC = []
GL = []
g_summary = []
d_summary = []


X_train=img_str.numpy()
Y_train=label_str
category=Y_train
Y_train = Y_train.reshape(-1, 1) #rows unkown columns 1
Y_train = [int(numeric_string) for numeric_string in Y_train]
category = [int(numeric_string) for numeric_string in category]
category=np.asarray(category)
max_cat = max(Y_train)+1
tensor_shape = (X_train.shape[1], X_train.shape[2], max_cat)
concat_shape = (X_train.shape[1], X_train.shape[2], max_cat+1)
label_arr = []
for i in range(max_cat):
    temp = np.zeros(tensor_shape)
    temp[:,:,i] = 1
    label_arr.append(temp)
label_arr = np.asarray(label_arr)

Y_train = to_categorical(Y_train, max_cat)
#***************************************************************************************************************************************



class CDCGAN():
    def __init__(self):
        # Input definieren
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.num_classes = len(categories)
        self.latent_dim =100
        self.ki_mean = 0.0
        self.ki_stddev = 0.02
        self.ks = 5
        self.mom = .8
        self.channels = 1
        self.drop = 0.0
        self.alp = .2
        self.lr=0.0001
        self.b1=0.5
        self.b2=0.999



        optimizer = Adam(self.lr, self.b1, self.b2)  # ,e )

        # Erstellen und kompilieren des Diskriminators
        self.discriminator = self.make_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Erstellen des Generators
        self.generator = self.make_generator()

        # Die Eingabe des Generator (Rauschen und Label) werden definiert
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))

        # Mit der Eingabe wird ein entsprechendes Bild generiert
        img = self.generator([noise, label])

        # Im zusammengesetzten Modell wird nur der Generator trainiert
        self.discriminator.trainable = False

        # Der Diskriminator erhält ein generiertes Bild mit dem angehängten Labeltensor und bestimmt die Gültigkeit
        label_tensoren = Input(shape=tensor_shape)
        valid = self.discriminator([img, label_tensoren])

        # Im kombinierten modell (Generator und Diskriminator) wird der Generator trainiert, den Diskriminator zu überlisten
        self.combined = Model([noise, label, label_tensoren], valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)


    def make_generator(self):  # Generator mit 5 Conv Schichten; Verkettung der Eingabe (Rauschen und Label)

            model = tf.keras.Sequential()

            model.add(Dense(256 * 4 * 4, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), activation="relu",
                    input_dim=self.latent_dim+self.num_classes))
            model.add(Reshape((4, 4, 256)))

            model.add(UpSampling2D())  # 8*8
            model.add(Conv2D(128, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(Activation("relu"))

            model.add(UpSampling2D())  # 16*16
            model.add(Conv2D(64, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(Activation("relu"))

            model.add(UpSampling2D())  # 32*32
            model.add(Conv2D(32, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(Activation("relu"))

            model.add(UpSampling2D())  # 64*64
            model.add(Conv2D(16, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(Activation("relu"))

            model.add(UpSampling2D())  # 128*128
    # model.add(Conv2D(8, kernel_size=ks, kernel_initializer=RN(mean=ki_mean, stddev=ki_stddev),padding="same"))
    # model.add(BatchNormalization(momentum=mom))
    # model.add(Activation("relu"))
    # 5CONV layer
            model.add(
                        Conv2D(self.channels, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), padding="same"))
            model.add(Activation("tanh"))

            noise = Input(shape=(self.latent_dim,))
            label = Input(shape=(self.num_classes,))
            model_input = Concatenate()([noise, label])

            img = model(model_input)

            return Model([noise, label], img)




    def make_discriminator(self):  # Generator mit 5 Conv Schichten; Verkettung der Eingabe (Bild und Labeltensor)

            model = tf.keras.Sequential()  # 128*128*7

            model.add(Conv2D(16, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), strides=2,
                     input_shape=(concat_shape), padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(LeakyReLU(alpha=self.alp))  # 64*64
            model.add(Dropout(self.drop))

            model.add(
        Conv2D(32, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), strides=2, padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(LeakyReLU(alpha=self.alp))  # 32*32
            model.add(Dropout(self.drop))

            model.add(
        Conv2D(64, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), strides=2, padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(LeakyReLU(alpha=self.alp))  # 16*16
            model.add(Dropout(self.drop))

            model.add(
        Conv2D(128, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), strides=2, padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(LeakyReLU(alpha=self.alp))  # 8*8
            model.add(Dropout(self.drop))

            model.add(
        Conv2D(256, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), strides=2, padding="same"))
            model.add(BatchNormalization(momentum=self.mom))
            model.add(LeakyReLU(alpha=self.alp))  # 4*4
            model.add(Dropout(self.drop))

    # Kernel 5x5 zu groß: deshalb Flatten und Dense
            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid'))

            img = Input(shape=self.img_shape)
            label_tensoren = Input(shape=tensor_shape)

            model_input = Concatenate()([img, label_tensoren])

            validity = model(model_input)

            return Model([img, label_tensoren], validity)

    def train(self, EPOCHS, BATCH_SIZE, SAMPLE_INTERVAL):

        # Korrekte Ergebnisse; Label der Klassen (real / generiert)
        valid = np.ones((BATCH_SIZE, 1))
        fake = np.zeros((BATCH_SIZE, 1))

        path_log = "./CDCGAN/"
        try:
            os.mkdir(path_log)
        except OSError as error:
            print(error)

        # Durchlaufen der Epochen
        for epoch in range(EPOCHS):

            # Berechnungszeit schätzen
            if epoch == 0:
                stamp = time.time()
            if epoch == SAMPLE_INTERVAL:
                scnd_stamp = time.time()
                runtime = (EPOCHS * ((scnd_stamp - stamp) / 10)) / 60
                print("Approx. runtime for current run: {} [min]".format(runtime))

            #  Training des Diskriminators

            # Zufällige Auswahl eine Menge von Bildern und entsptechende Zuweisungen
            idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
            labels = Y_train[idx]
            imgs= X_train[idx]
            label_tensoren = label_arr[category[idx]]

            # Aus dem zufällig generierten Rauschen eine Menge von Bildern generieren
            noise = np.random.normal(0, 1, (BATCH_SIZE, self.latent_dim))
            gen_imgs = self.generator.predict([noise, labels])

            # Trainieren des Diskriminators (real = 1; generiert = 0)
            d_loss_real = self.discriminator.train_on_batch([imgs, label_tensoren], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, label_tensoren], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #  Training des Generators

            # Zufällig eine Menge von Kondition der möglichen Klassen generieren und daraus Labeltensoren und Musterloesungen erstellen
            sampled_labels = np.random.randint(0, self.num_classes, BATCH_SIZE)
            label_tensoren = label_arr[sampled_labels]
            sampled_labels = sampled_labels.reshape(-1, 1)
            sampled_labels = to_categorical(sampled_labels, self.num_classes)

            # Training des Generators (ueberlisten des Diskriminators)
            g_loss = self.combined.train_on_batch([noise, sampled_labels, label_tensoren], valid)

            # Darstellung des Lernprozesses
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # Erfassen der Werte
            DL.append(d_loss[0])
            ACC.append(d_loss[1])
            GL.append(g_loss)

            # Beim Erreichen eines Intervalls werden Probebilder gespeichert
            if epoch % SAMPLE_INTERVAL == 0:
                self.save_imgs(epoch)

            # Ist die training durchgelaufen, werden die Gewichte gespeichert
            if epoch == EPOCHS - 1:

                self.generator.save_weights(path_log + 'generator.h5', overwrite=True)
                self.discriminator.save_weights(path_log + 'discriminator.h5',
                                                overwrite=True)

    def save_imgs(self, epoch):  # Speichern der Probebilder
        r, c = 6, self.num_classes
        fig, axs = plt.subplots(r, c)
        sampled_labels = np.arange(0, c).reshape(-1, 1)
        sampled_labels = to_categorical(sampled_labels, c)
        path_log = "./CDCGAN/Save_images/"
        try:
            os.mkdir(path_log)
        except OSError as error:
            print(error)
        for i in range(r):
            noise = np.random.normal(0, 1, (c, self.latent_dim))
            gen_imgs = self.generator.predict([noise, sampled_labels])
            gen_imgs = 0.5 * gen_imgs + 0.5

            for j in range(c):
                axs[i, j].imshow(gen_imgs[j, :, :, 0], cmap='gray')
                axs[0, j].set_title("{}".format(defects[j]), fontsize=8)
                axs[i, j].axis('off')

        fig.savefig(path_log + "cdcgan_concat_{}.png".format(epoch))
        plt.close()

        # Abspeichern des Lernverlaufs
        fig, axs = plt.subplots(3, 1, sharex=True)
        axs[0].plot(DL)
        axs[0].set_ylabel("D_Loss")
        axs[0].set_title("D & G Loss over Epochs")
        axs[1].plot(ACC)
        axs[1].set_ylabel("Accuracy")
        axs[2].plot(GL)
        axs[2].set_ylabel("G_Loss")
        axs[2].set_xlabel("Epochs")
        try:
            dt = time.asctime(time.localtime())
        except:
            dt = "NO_DAYTIME_AVAILABLE"
        fig.savefig(path_log + "Plot_cdcgan_concat.png")
        plt.close()
        print("Plot erstellt")
        # Leeren der Zwischenspeicher
        if epoch == EPOCHS - 1:
            del DL[:]
            del ACC[:]
            del GL[:]

    def generate_images(self, gen_label, amount):  # Mit einerm trainierten Generator Bilder generieren
        self.generator.load_weights('./CDCGAN/generator.h5')
        label_ar = np.array([gen_label])
        gen_label_ar = label_ar.reshape(-1, 1)
        gen_label_ar = to_categorical(gen_label_ar, self.num_classes)
        path_log = "./CDCGAN/generated_images"
        try:
            os.mkdir(path_log)
        except OSError as error:
            print(error)
        for i in range(amount):
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            gen_img = self.generator.predict([noise, gen_label_ar])
            gen_img = np.squeeze(gen_img)
            # Rescale images -1,1 => 0,255
            gen_img = 127.5 * gen_img + 127.5

            cv2.imwrite(os.path.join(path_log, "{}_Type_{}.png".format(i, gen_label)), gen_img)

        print("Bilder wurden gespeichert!\n")


if __name__ == '__main__':

        cdcganz=CDCGAN()
        #cdcganz.train(51,8,10)
        #cdcganz.generate_images(3,10)
        #gan=GAN()
        #gan.train(train_dataset, EPOCHS)


