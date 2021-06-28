from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal as RN

import matplotlib.pyplot as plt

import numpy as np

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 6
        self.latent_dim = 100
        self.ki_mean = 0.0
        self.ki_stddev = 0.02
        self.ks = 5
        self.mom = .8
        self.drop = 0.0
        self.alp = .2
        self.lr = 0.0001
        self.b1 = 0.5
        self.b2 = 0.999
        self.concat_shape = (128, 128, 7)

        optimizer = Adam(self.lr, self.b1, self.b2)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256 * 4 * 4, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), activation="relu",
                        input_dim=self.latent_dim ))
        model.add(Reshape((4, 4, 256)))

        model.add(UpSampling2D())  # 8*8
        model.add(Conv2D(128, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev),
                         padding="same"))
        model.add(BatchNormalization(momentum=self.mom))
        model.add(Activation("relu"))

        model.add(UpSampling2D())  # 16*16
        model.add(Conv2D(64, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev),
                         padding="same"))
        model.add(BatchNormalization(momentum=self.mom))
        model.add(Activation("relu"))

        model.add(UpSampling2D())  # 32*32
        model.add(Conv2D(32, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev),
                         padding="same"))
        model.add(BatchNormalization(momentum=self.mom))
        model.add(Activation("relu"))

        model.add(UpSampling2D())  # 64*64
        model.add(Conv2D(16, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev),
                         padding="same"))
        model.add(BatchNormalization(momentum=self.mom))
        model.add(Activation("relu"))

        model.add(UpSampling2D())  # 128*128
        # model.add(Conv2D(8, kernel_size=ks, kernel_initializer=RN(mean=ki_mean, stddev=ki_stddev),padding="same"))
        # model.add(BatchNormalization(momentum=mom))
        # model.add(Activation("relu"))
        # 5CONV layer
        model.add(
            Conv2D(self.channels, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev),
                   padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))


        model_input = Concatenate([noise, label])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()  # 128*128*7

        model.add(
            Conv2D(16, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), strides=2,input_shape=self.concat_shape,
                    padding="same"))
        model.add(BatchNormalization(momentum=self.mom))
        model.add(LeakyReLU(alpha=self.alp))  # 64*64
        model.add(Dropout(self.drop))

        model.add(
            Conv2D(32, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), strides=2,
                   padding="same"))
        model.add(BatchNormalization(momentum=self.mom))
        model.add(LeakyReLU(alpha=self.alp))  # 32*32
        model.add(Dropout(self.drop))

        model.add(
            Conv2D(64, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), strides=2,
                   padding="same"))
        model.add(BatchNormalization(momentum=self.mom))
        model.add(LeakyReLU(alpha=self.alp))  # 16*16
        model.add(Dropout(self.drop))

        model.add(
            Conv2D(128, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), strides=2,
                   padding="same"))
        model.add(BatchNormalization(momentum=self.mom))
        model.add(LeakyReLU(alpha=self.alp))  # 8*8
        model.add(Dropout(self.drop))

        model.add(
            Conv2D(256, kernel_size=self.ks, kernel_initializer=RN(mean=self.ki_mean, stddev=self.ki_stddev), strides=2,
                   padding="same"))
        model.add(BatchNormalization(momentum=self.mom))
        model.add(LeakyReLU(alpha=self.alp))  # 4*4
        model.add(Dropout(self.drop))

        # Kernel 5x5 zu groß: deshalb Flatten und Dense
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=(self.img_shape,))
        label = Input(shape=(self.num_classes,))

        #label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        #flat_img = Flatten()(img)

        model_input = Concatenate([img, label])

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        data_dir = "./Dataset/"
        data_dir = pathlib.Path(data_dir)
        image_count = len(list(data_dir.glob('*/*.png')))
        print(image_count)
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, image_size=(128, 128),
                                                                       label_mode="int", seed=123,
                                                                       batch_size=image_count, color_mode="grayscale")
        self.class_names = train_ds.class_names
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        X_train, y_train = next(iter(normalized_ds))
        #X_train = X_train.numpy()
        #y_train = y_train.numpy()

        # Configure input
        #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real,d_acc_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake,d_acc_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 6, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D1 loss: %f,D2 loss:%f acc1.: %.2f%% acc2.: %.2f%%] [G loss: %f]" % (epoch, d_loss_real,d_loss_fake,100*d_acc_real,100*d_acc_fake, g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 6, 6
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 6).reshape(-1, 1)
        path_log = "./CDCGAN/Save_images/"
        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(path_log + "cdcgan_concat_{}.png".format(epoch))
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=20000, batch_size=128, sample_interval=200)
