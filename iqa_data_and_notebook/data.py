import os
import cv2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Concatenate, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from tensorflow.keras import layers
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.initializers import RandomNormal as RN
import time
from IPython import display
import shutil


AUTOTUNE = tf.data.AUTOTUNE
import pathlib
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator

categories = {'none': 0, 'wrinkle': 1, 'twist': 2, 'foreign material': 3, 'overlap': 4, 'gap': 5}
defects=["0_None", "1_Wrinkle", "2_Twist", "3_Foreign_Body", "4_Overlap", "5_Gap"]
data_dir="./Dataset/"
data_dir=pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.png')))
print(image_count)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,image_size=(128,128),label_mode="int",seed=123,
                                                               batch_size=image_count,color_mode="grayscale")
class_names = train_ds.class_names
print(class_names)
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5,offset=-1)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
X_train,Y_train = next(iter(normalized_ds))
category=Y_train
X_train=X_train.numpy()
Y_train=Y_train.numpy()
Y_train = Y_train.reshape(-1, 1) #rows unkown columns 1
#Y_train = [int(numeric_string) for numeric_string in Y_train]
#category = [int(numeric_string) for numeric_string in category]
category=np.asarray(category)
max_cat = int(max(Y_train)+1)
tensor_shape = (X_train.shape[1], X_train.shape[2], max_cat)
concat_shape = (X_train.shape[1], X_train.shape[2], max_cat+1)
label_arr = []
for i in range(max_cat):
    temp = np.zeros(tensor_shape)
    temp[:,:,i] = 1
    label_arr.append(temp)
label_arr = np.array(label_arr)


Y_train = to_categorical(Y_train, max_cat)
DL = []
ACC = []
GL = []

class AUG():
    def save_imggenrator(self,data,filename):

         it = data.flow(X_train, batch_size=1)
         for i in range(X_train.shape[0]):

             batch=it.next()
             path_log = "./Dataset/" + class_names[category[i]]


             img = np.squeeze(batch[0])
             img = 127.5 * img + 127.5
             cv2.imwrite(os.path.join(path_log, "{}_{}.png".format(filename,i)), img)

    def save_img(self, data, filename):
        for i in range(data.shape[0]):
            path_log = "./Dataset/" + class_names[category[i]]

            img = np.squeeze(data[i])
            img = 127.5 * img + 127.5
            cv2.imwrite(os.path.join(path_log, "{}_{}.png".format(filename, i)), img)
    def aug_process(self):
        # create image data augmentation generator
        width = ImageDataGenerator(width_shift_range=.2)
        self.save_imggenrator(width,'width_shift_range')
        height = ImageDataGenerator(height_shift_range=.2)
        self.save_imggenrator(height, 'height_shift_range')
        horizontal_flip = ImageDataGenerator(horizontal_flip=True)
        self.save_imggenrator(horizontal_flip, 'horizontal_flip')
        vertical_flip = ImageDataGenerator(vertical_flip=True)
        self.save_imggenrator(vertical_flip, 'vertical_flip')
        brightness_range = ImageDataGenerator(brightness_range=[0.2,1.0])
        self.save_imggenrator(brightness_range, 'brightness_range')
        rotation_range = ImageDataGenerator(rotation_range=90)
        self.save_imggenrator(rotation_range, 'rotation_range')
        zoom_range = ImageDataGenerator(zoom_range=[0.5, 1.0])
        self.save_imggenrator(zoom_range, 'zoom_range')


        bright = tf.image.adjust_brightness(X_train, 0.4)
        self.save_img(bright,'adjust_brightness')
        contrast=tf.image.adjust_contrast(X_train, 2)
        self.save_img(contrast, 'adjust_contrast')
        rotated = tf.image.rot90(X_train)
        self.save_img(rotated, 'rot90')
        rot=tfa.image.rotate(X_train,45)
        self.save_img(rot, 'rot45')
        seed=(1,2)
        #jpeg=tf.image.adjust_jpeg_quality(X_train, 75, 95 )
        #self.save_img(jpeg, 'jpeg_quality')


    def train_test_split(self):
        rootdir = '/home/fpds04/sa/code/Dataset'  # path of the original folder
        for i in class_names:
            os.makedirs(rootdir + '/train/' + i)

            os.makedirs(rootdir + '/test/' + i)

            source = rootdir + '/' + i

            allFileNames = os.listdir(source)

            np.random.shuffle(allFileNames)

            test_ratio = 0.10

            train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                       [int(len(allFileNames) * (1 - test_ratio))])

            train_FileNames = [source + '/' + name for name in train_FileNames.tolist()]
            test_FileNames = [source + '/' + name for name in test_FileNames.tolist()]

            for name in train_FileNames:
                shutil.copy(name, rootdir + '/train/' + i)

            for name in test_FileNames:
                shutil.copy(name, rootdir + '/test/' + i)


class IMG_CLASSIFY():
    def train(self):
        data_dir = "./Dataset_aug_train/train/"
        data_dir = pathlib.Path(data_dir)
        image_count = len(list(data_dir.glob('*/*.png')))
        print(image_count)
        batch_size = 128
        img_height = 128
        img_width = 128
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,color_mode="grayscale")
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,color_mode="grayscale")
        class_names = train_ds.class_names
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        num_classes = 4

        model = Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 127.5,offset=-1, input_shape=(img_height, img_width, 1)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        epochs = 100

        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        path_log = "./CDCGAN/"

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=path_log + "Aug_model", histogram_freq=1)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,callbacks=[callback,tensorboard_callback]
        )
        model.save("./CDCGAN/Classify_augdata.h5")
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        path_log = "./CDCGAN/Img_classify/"
        plt.savefig(path_log+"a_plot_{}".format("aug_data"))

    def predict(self):
        testpath="./Dataset/"
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            seed=123,
            image_size=(img_height, img_width),
            batch_size=32, color_mode="grayscale")
        model=keras.models.load_model("Classify_augdata.h5")
        results=model.evaluate(test_ds,batch_size=32)
        print("test loss, test acc:", results)





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

            model = Sequential()

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

            model = Sequential()  # 128*128*7

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
                self.save_imgs(epoch,EPOCHS)

            # Ist die training durchgelaufen, werden die Gewichte gespeichert
            if epoch == EPOCHS - 1:

                self.generator.save_weights(path_log + 'generator.h5', overwrite=True)
                self.discriminator.save_weights(path_log + 'discriminator.h5',
                                                overwrite=True)

    def save_imgs(self, epoch,EPOCHS):  # Speichern der Probebilder
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
        path_log = "./CDCGAN/generated_images/"
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

        aug=AUG()
        #aug.aug_process()
        #aug.train_test_split()
        img_classify=IMG_CLASSIFY()
        img_classify.train()
