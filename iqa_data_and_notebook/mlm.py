# example of training an conditional gan on the fashion mnist dataset
import cv2
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU,BatchNormalization,ReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import os
from keras.initializers import RandomNormal



class mlm():
    def __init__(self):
        self.DL1 = []
        self.DL2 = []
        self.GL = []
        self.ACC1=[]
        self.ACC2=[]
        self.class_names=[]




    # define the standalone discriminator model
    def define_discriminator(self,in_shape=(128, 128, 1), n_classes=4):
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        li = Embedding(n_classes, 50)(in_label)
        # scale up to image dimensions with linear activation
        n_nodes = in_shape[0] * in_shape[1]
        li = Dense(n_nodes)(li)
        # reshape to additional channel
        li = Reshape((in_shape[0], in_shape[1], 1))(li)
        # image input
        in_image = Input(shape=in_shape)
        # concat label as a channel
        merge = Concatenate()([in_image, li])
        init = RandomNormal(stddev=0.02)
        # downsample
        fe = Conv2D(16, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(merge)
        fe=BatchNormalization(momentum=.8)(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        #fe = BatchNormalization(momentum=.8)(fe)
        fe=Dropout(.25)(fe)
        # downsample
        fe = Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(fe)
        fe=BatchNormalization(momentum=.8)(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        #fe = BatchNormalization(momentum=.8)(fe)
        fe=Dropout(.25)(fe)
        fe = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(fe)
        fe=BatchNormalization(momentum=.8)(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        #fe = BatchNormalization(momentum=.8)(fe)
        fe=Dropout(.25)(fe)
        fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(fe)
        fe=BatchNormalization(momentum=.8)(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        #fe = BatchNormalization(momentum=.8)(fe)
        fe=Dropout(.25)(fe)
        # flatten feature maps
        fe = Flatten()(fe)
        # dropout
        fe = Dropout(0.4)(fe)
        # output
        out_layer = Dense(1, activation='sigmoid')(fe)
        # define model
        model = Model([in_image, in_label], out_layer)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    # define the standalone generator model
    def define_generator(self,latent_dim, n_classes=4):
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        li = Embedding(n_classes, 50)(in_label)
        # linear multiplication
        n_nodes = 8 * 8
        li = Dense(n_nodes)(li)
        # reshape to additional channel
        li = Reshape((8, 8, 1))(li)
        # image generator input
        in_lat = Input(shape=(latent_dim,))
        # foundation for 7x7 image
        n_nodes = 128 * 8 * 8
        gen = Dense(n_nodes)(in_lat)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Reshape((8, 8, 128))(gen)
        # merge image gen and label input
        merge = Concatenate()([gen, li])
        init = RandomNormal(stddev=0.02)
        # upsample to 14x14
        gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merge)
        gen = BatchNormalization(momentum=.8)(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        #gen = BatchNormalization(momentum=.8)(gen)
        # upsample to 28x28
        gen = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
        gen = BatchNormalization(momentum=.8)(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        #gen = BatchNormalization(momentum=.8)(gen)
        gen = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
        gen = BatchNormalization(momentum=.8)(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        #gen = BatchNormalization(momentum=.8)(gen)
        gen = Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(gen)

        gen = BatchNormalization(momentum=.8)(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
       #gen = BatchNormalization(momentum=.8)(gen)

        # output
        out_layer = Conv2D(1, (8, 8), activation='tanh', padding='same')(gen)
        # define model
        model = Model([in_lat, in_label], out_layer)
        return model

    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self,g_model, d_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # get noise and label inputs from generator model
        gen_noise, gen_label = g_model.input
        # get image output from the generator model
        gen_output = g_model.output
        # connect image output and label input from generator as inputs to discriminator
        gan_output = d_model([gen_output, gen_label])
        # define gan model as taking noise and label and outputting a classification
        model = Model([gen_noise, gen_label], gan_output)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    # load fashion mnist images
    def load_real_samples(self):
        # load dataset
        data_dir = "./Dataset_aug/"
        data_dir = pathlib.Path(data_dir)
        image_count = len(list(data_dir.glob('*/*.png')))
        print(image_count)
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, image_size=(128,128),
                                                                       label_mode="int", seed=123,
                                                                       batch_size=image_count, color_mode="grayscale")
        self.class_names = train_ds.class_names
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        X_train, Y_train = next(iter(normalized_ds))
        X_train=X_train.numpy()
        Y_train=Y_train.numpy()
        return [X_train,Y_train]
    # # select real samples
    def generate_real_samples(self,dataset, n_samples):
        # split into images and labels
        images, labels = dataset
        # choose random instances
        ix = randint(0, images.shape[0], n_samples)
        # select images and labels
        X, labels = images[ix], labels[ix]
        # generate class labels
        y = ones((n_samples, 1))
        return [X, labels], y

    # generate points in latent space as input for the generator
    def generate_latent_points(self,latent_dim, n_samples, n_classes=4):
        # generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        z_input = x_input.reshape(n_samples, latent_dim)
        # generate labels
        labels = randint(0, n_classes, n_samples)
        return [z_input, labels]

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self,generator, latent_dim, n_samples):
        # generate points in latent space
        z_input, labels_input = self.generate_latent_points(latent_dim, n_samples)
        # predict outputs
        images = generator.predict([z_input, labels_input])
        # create class labels
        y = zeros((n_samples, 1))
        return [images, labels_input], y

    # train the generator and discriminator
    def train(self,g_model, d_model, gan_model, dataset, latent_dim, n_epochs=500, n_batch=128):
        bat_per_epo = int(dataset[0].shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        for i in range(n_epochs):
            # enumerate batches over the training set
            for j in range(bat_per_epo):
                # get randomly selected 'real' samples
                [X_real, labels_real], y_real = self.generate_real_samples(dataset, half_batch)
                # update discriminator model weights
                d_loss1, d_acc1 = d_model.train_on_batch([X_real, labels_real], y_real)
                # generate 'fake' examples
                [X_fake, labels], y_fake = self.generate_fake_samples(g_model, latent_dim, half_batch)
                # update discriminator model weights
                d_loss2, d_acc2 = d_model.train_on_batch([X_fake, labels], y_fake)
                # prepare points in latent space as input for the generator
                [z_input, labels_input] = self.generate_latent_points(latent_dim, n_batch)
                # create inverted labels for the fake samples
                y_gan = ones((n_batch, 1))
                # update the generator via the discriminator's error
                g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
                # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f acc1=%.3f acc2=%.3f' %
                      (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss,d_acc1,d_acc2))
                self.DL1.append(d_loss1)
                self.DL2.append(d_loss2)
                self.GL.append(g_loss)
                self.ACC1.append(d_acc1)
                self.ACC2.append(d_acc2)

                latent_points, label = mlm.generate_latent_points(100, 16)
                # specify labels
                labels = np.asarray([x for _ in range(4) for x in range(4)])
                gen_img=g_model.predict([latent_points,labels])
                gen_img = (gen_img + 1)/2
                if i%10==0:
                    self.save_plot(gen_img,4,i)


        # save the generator model
        g_model.save('cgan500_generator.h5')

    def save_plot(self,examples, n,epochs):
        # plot images
        for i in range(n * n):
            # define subplot
            plt.subplot(n, n, 1 + i)
            # turn off axis
            plt.axis('off')
            # plot raw pixel data
            plt.imshow(examples[i, :, :, 0], cmap='gray')
        plt.show()
        path_log = "./CDCGAN/2000/"
        plt.savefig(path_log+"a_plot_{}".format(epochs))
        plt.close()
        # fig, axs = plt.subplots(n, n)
        # for i in range(n):
        #     latent_points, label = mlm.generate_latent_points(100, 36)
        #     # specify labels
        #     labels = np.asarray([x for _ in range(6) for x in range(6)])
        #     gen_img = g_model.predict([latent_points, labels])
        #     gen_img = (gen_img + 1) / 2
        #
        #     for j in range(n):
        #         axs[i, j].imshow(gen_img[j, :, :, 0], cmap='gray')
        #         axs[0, j].set_title("{}".format(self.class_names[j]), fontsize=8)
        #         axs[i, j].axis('off')
        # path_log = "./CDCGAN/300drop/"
        # plt.savefig(path_log + "a_plot_{}".format(epochs))
        # plt.close()


        # fig, axs = plt.subplots(5, 1, sharex=True)
        # axs[0].plot(self.DL1)
        # axs[0].set_ylabel("D1_Loss")
        # axs[0].set_title("D & G Loss over Epochs")
        # axs[1].plot(self.DL2)
        # axs[1].set_ylabel("D2_Loss")
        # axs[2].plot(self.GL)
        # axs[2].set_ylabel("G_Loss")
        # axs[3].plot(self.ACC1)
        # axs[3].set_ylabel("ACC1")
        # axs[4].plot(self.ACC2)
        # axs[4].set_ylabel("ACC2")
        # axs[4].set_xlabel("Epochs")
        # fig.savefig(path_log + "Plot_cdcgan_concat.png")
        # plt.close()

    def generate_images(self,model,label,amount):
        gen_label = []
        path_log = "./CDCGAN/2000/plots/"+self.class_names[label]
        try:
            os.mkdir(path_log)
        except OSError as error:
            print(error)

        for i in range(24):
            gen_label.append(label)
        gen_label=np.array(gen_label)
        latent_points, labels = mlm.generate_latent_points(100, 16)
        X = model.predict([latent_points, gen_label])
        for i in range(amount):
            plt.imshow(X[i,:,:,0],cmap='gray')
            plt.title(self.class_names[label])
            plt.savefig(path_log+'/'+'{}_Type_{}'.format(i,self.class_names[label]))

        # for i in range(amount):
        #     latent_points, labels = mlm.generate_latent_points(100, 36)
        #     X = model.predict([latent_points, gen_label])
        #     X = (X + 1) / 2.0
        #     path_log = "./CDCGAN/plots/"
        #     cv2.imwrite(os.path.join(path_log, "{}_Type_{}.png".format(i, label)), X)





if __name__ == '__main__':
    # size of the latent space
    latent_dim = 100
    mlm=mlm()
    # create the discriminator
    d_model = mlm.define_discriminator()
    # create the generator
    g_model = mlm.define_generator(latent_dim)
    # create the gan
    gan_model = mlm.define_gan(g_model, d_model)
    # load image data
    dataset = mlm.load_real_samples()
    # train model
    mlm.train(g_model, d_model, gan_model, dataset, latent_dim)
    model = load_model('cgan2000drop_generator.h5',compile=False)
    # generate images
    latent_points,label = mlm.generate_latent_points(100,24 )
    # specify labels
    labels = np.asarray([x for _ in range(4) for x in range(4)])
    #label=np.full((100,1),2)

    # generate images
    X = model.predict([latent_points, labels])
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    #gen_label=np.array([0])
   # mlm.generate_images(model,1,10)
    #mlm.generate_images(model, 2, 10)
   # mlm.generate_images(model, 3, 10)
    #mlm.generate_images(model, 4, 10)
    #mlm.generate_images(model, 5, 10)
    #X=X*127.7+127.5
    # plot the result
    #mlm.save_plot(X, 6,1000)