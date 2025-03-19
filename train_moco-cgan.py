import numpy as np
import os

from keras.models import Input, Model
from keras.layers import Activation, Input, Dropout, merge, Concatenate, MaxPooling3D
from keras.layers.convolutional import  Conv3D, UpSampling2D, UpSampling3D, Conv3DTranspose
from tensorflow.keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from utils.data_generator import  data_generator
from networks.discriminator import GanDiscriminator
from networks.CGAN import CGAN
from utils import logger
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
import time
from keras.utils import generic_utils as keras_generic_utils
import scipy


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#limit GPU memory usage
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(allow_growth=True))
set_session(tf.compat.v1.Session(config=config))


# # --------------------------------------------
# # HELPER FUNCTIONS

#data augmentation functions   
def random_rotation(image_array,image_motion_array):
    # pick a random degree of rotation 
    random_degree = np.random.uniform(-6, 6)
    return scipy.ndimage.interpolation.rotate(image_array, random_degree,  reshape=False), scipy.ndimage.interpolation.rotate(image_motion_array, random_degree, reshape=False)

# define the 3d unet architecture 

def conv_block(m, dim, acti, bn, res, do=0):
    n = Conv3D(dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv3D(dim, 3, activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res,do=0)
        m = MaxPooling3D(pool_size=(2,2,2))(n) if mp else Conv3D(dim, 3, strides=2, padding='same')(n)
        m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
        #
        if up:
                m = UpSampling3D(size=(2,2,2))(m)
                m = Conv3D(dim, 2, activation=acti, padding='same')(m)
        else:
                m = Conv3DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, dim, acti, bn, res,do=0)
    else:
            m = conv_block(m, dim, acti, bn, res, do=0.5)
    return m

def UNet(img_shape, out_ch=1, start_ch=4, depth=2, inc_rate=2., activation='relu', dropout=0.5, batchnorm=False, maxpool=True, upconv=False, residual=False):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv3D(out_ch, 1, activation='relu')(o)
    return Model(inputs=i, outputs=o)

# # --------------------------------------------
# # LOADING DATA

#load and prep data
mpath = r'/home/nghiemb/RMC_repos/MoCo_cGAN'
dpath = mpath + r'/data/training_dataset/slices'
spath = mpath + r'/networks'

#Loading complex-valued data
train_labels = np.load(dpath + r'/train/train_GT.npy')
train_corr = np.load(dpath + r'/train/train_corr.npy') #corr for corrupted

val_labels = np.load(dpath + r'/val/val_GT.npy')
val_corr = np.load(dpath + r'/val/val_corr.npy') #corr for corrupted

#Retrieve magnitude-only data
train_labels = np.squeeze(np.abs(train_labels[..., 0:1] + 1j*train_labels[..., 1:2])) #dim: [3072,8,192,224]
train_corr = np.squeeze(np.abs(train_corr[..., 0:1] + 1j*train_corr[..., 1:2])) #dim: [3072,8,192,224]

val_labels = np.squeeze(np.abs(val_labels[..., 0:1] + 1j*val_labels[..., 1:2])) #dim: [768,8,192,224]
val_corr = np.squeeze(np.abs(val_corr[..., 0:1] + 1j*val_corr[..., 1:2])) #dim: [768,8,192,224]

#Transpose data to align with Johnson & Drangova's convention
train_labels = np.transpose(train_labels, axes=(3,2,1,0)) #dim: [224,192,8,3072]
train_corr = np.transpose(train_corr, axes=(3,2,1,0)) #dim: [224,192,8,3072]

val_labels = np.transpose(val_labels, axes=(3,2,1,0)) #dim: [224,192,8,768]
val_corr = np.transpose(val_corr, axes=(3,2,1,0)) #dim: [224,192,8,768]


im_width = train_labels.shape[0] #AP
im_height= train_labels.shape[1] #LR
im_slices= train_labels.shape[2] #SI

img_shape= [im_width,im_height,im_slices,1] #(xdim,ydim,slices,channels)

# # --------------------------------------------
# # DATA AUGMENTATION

#data augmentation options
fliplr=False
rotate=False

# if fliplr:
#     train_corr=np.concatenate([train_corr, np.flip(train_corr,3)], axis=0)
#     train_labels=np.concatenate([train_labels, np.flip(train_labels,3)], axis=0)
#     im_count=np.shape(train_corr)[0]

# #random shuffle of training examples
# arr=np.arange(np.shape(train_corr)[0])
# np.random.shuffle(arr)
# train_corr=train_corr[arr,...]
# train_labels=train_labels[arr,...]

# print('start augmentation rotations')
# if rotate:
#     im_count=np.shape(train_corr)[0]
#     tempx=np.zeros(np.shape(train_corr))
#     tempy=np.zeros(np.shape(train_labels))
#     for i in range(im_count):
#         [tempx[i,:,:,:,:], tempy[i,:,:,:,:]]=random_rotation(train_corr[i,:,:,:,:],train_labels[i,:,:,:,:])
#     train_corr=np.concatenate([train_corr,tempx])
#     train_labels=np.concatenate([train_labels,tempy])

# print('done data augmentation')
# # width, height of images to work with.


# --------------------------------------------
# INITIATING MODELS

# ----------------------
# GENERATOR
# Our generator is a 3D  U-NET with skip connections
# ----------------------
# input/oputputt channels in image
input_channels = 1
output_channels = 1

# image dims
input_img_dim =(im_width, im_height, im_slices, input_channels)
output_img_dim = ( im_width,im_height,im_slices,  output_channels)


generator_nn =UNet(img_shape, out_ch=1, start_ch=64, depth=3, inc_rate=2., activation='relu', dropout=0.5, batchnorm=True, maxpool=True, upconv=True, residual=True)

generator_nn.summary()

# ----------------------
#  GAN DISCRIMINATOR
discriminator_nn = GanDiscriminator(output_img_dim=output_img_dim)
discriminator_nn.summary()

# disable training while we put it through the GAN
discriminator_nn.trainable = False

# ------------------------
# Define Optimizers
opt_discriminator = Adam(lr=5e-5,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
opt_generator = Adam(lr=5E-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

# -------------------------
# compile generator
generator_nn=multi_gpu_model(generator_nn, gpus=2)
generator_nn.compile(loss='mse', optimizer=opt_generator)

# ----------------------
# MAKE FULL CGAN
# ----------------------
cgan_nn = CGAN(generator_model=generator_nn,
                  discriminator_model=discriminator_nn,input_img_dim)

cgan_nn.summary()

# ---------------------
# Compile CGAN
# we use a combination of mae and bin_crossentropy
loss = ['mae', 'binary_crossentropy']

loss_weights = [1, 1]
cgan_nn=multi_gpu_model(cgan_nn,gpus=2)
cgan_nn.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_generator)

# ---------------------
discriminator_nn=multi_gpu_model(discriminator_nn,gpus=2)
discriminator_nn.trainable =True
discriminator_nn.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

# ------------------------
# RUN TRAINING
batch_size =4
nb_epoch = 50
n_images_per_epoch =np.shape(train_corr)[0]

print('start training')
for epoch in range(0, nb_epoch):
    #
    print('Epoch {}'.format(epoch))
    batch_counter = 1
    start = time.time()
    progbar = keras_generic_utils.Progbar(n_images_per_epoch)
    #
    # init the datasources again for each epoch
    tng_gen = data_generator(train_corr, train_labels,batch_size=batch_size)
    val_gen = data_generator(val_corr,val_labels, batch_size=batch_size)
    #
    for mini_batch_i in range(0, n_images_per_epoch, batch_size):
        #
        X_train_decoded_imgs, X_train_original_imgs = next(tng_gen)
        # generate a batch of data and feed to the discriminator
        # some images that come out of here are real and some are fake
        # X is image patches for each image in the batch
        # Y is a 1x2 vector for each image. (means fake or not)
        X_discriminator, y_discriminator = patch_utils.get_disc_batch(X_train_original_imgs,
                                                          X_train_decoded_imgs,
                                                          generator_nn,
                                                          batch_counter,
                                                          patch_dim=sub_patch_dim)
        # Update the discriminator
        disc_loss = discriminator_nn.train_on_batch(X_discriminator, y_discriminator)
        # create a batch to feed the generator
        X_gen_target, X_gen = next(patch_utils.gen_batch(X_train_original_imgs, X_train_decoded_imgs, batch_size))
        y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
        y_gen[:, 1] = 1
        # Freeze the discriminator
        discriminator_nn.trainable = False
        # trainining GAN
        # print('calculating GAN loss...')
        gen_loss = cgan_nn.train_on_batch(X_gen, [X_gen_target, y_gen])
        # Unfreeze the discriminator
        discriminator_nn.trainable = True
        # counts batches we've ran through for generating fake vs real images
        batch_counter += 1
        # print losses
        D_log_loss = disc_loss
        gen_total_loss = gen_loss[0].tolist()
        gen_mae = gen_loss[1].tolist()
        gen_log_loss = gen_loss[2].tolist()
        #
        progbar.add(batch_size, values=[("Dis logloss", D_log_loss),
                                        ("Gen total", gen_total_loss),
                                        ("Gen L1 (mae)", gen_mae),
                                        ("Gen logloss", gen_log_loss)])
        #
    #       
    print('Epoch %s/%s, Time: %s' % (epoch + 1, nb_epoch, time.time() - start))
    #
    # ------------------------------
    # save weights on every 10th epoch
    if epoch % 10 == 0:
        generator_nn.save('models_best')
    #
    val_loss=generator_nn.evaluate(val_corr,val_labels,batch_size)
    #
    print('val_loss = %s' %val_loss)
    

