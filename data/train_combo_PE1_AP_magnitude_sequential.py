import os
import pathlib as plib
import scipy.io
import numpy as np
import math
import datetime

import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import  Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.python.keras.backend import set_session

# from read_data import read_data
# from Correction_Multi_input_complex import Correction_Multi_input
from Correction_Multi_input import Correction_Multi_input

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#limit GPU memory usage
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(allow_growth=True))
set_session(tf.compat.v1.Session(config=config))

#-------------------------------------------------------------------------------
#Helper functions
def save_model(path_weight, model,md = 'lstm'):
	model_json = model.to_json()
	with open(path_weight+r"/model_"+md+".json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights(path_weight+r"/model_"+md+".h5")
	print("The model is successfully saved")

# def load_model(path_weight, md = 'lstm'):
# 	json_file = open(path_weight+r"/model_"+md+".json", 'r')
# 	loaded_model_json = json_file.read()
# 	json_file.close()
# 	loaded_model = model_from_json(loaded_model_json)
# 	loaded_model.load_weights(path_weight+r"/model_"+md+".h5")
# 	print("Loaded model from disk")
# 	return loaded_model

def ssim_score(y_true, y_pred):
	score = K.mean(tf.image.ssim(y_true, y_pred, max_val))
	return score

def ssim_loss(y_true, y_pred):
	#loss_ssim = 1.0 - K.mean((tf.image.ssim(y_true, y_pred, 255.0)+1.0)/2.0)## SSIM range is between -1~1 so --> +1/2 is added
	loss_ssim = 1.0 - K.mean(tf.image.ssim(y_true, y_pred, max_val))
	return loss_ssim

def scheduler(epoch):
	ep = 10
	if epoch < ep:
		return learningRate
	else:
		return learningRate * math.exp(0.1 * (ep - epoch)) # lr decreases exponentially by a factor of 10

#Helper Class
class SaveNetworkProgress(keras.callbacks.Callback):
    def __init__(self, dirname):
        self.dirname = dirname
        #
    def on_train_begin(self, logs={}):
        self.epoch_ind = []
        self.losses = []
        self.val_losses = []
        #
    def on_epoch_end(self, epoch, logs={}):
        print("Finished Epoch {}".format(epoch))
        self.epoch_ind.append(epoch)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        np.save(self.dirname, dict([('val_losses', self.val_losses), \
                                    ('losses',self.losses), \
                                    ('epoch_ind', self.epoch_ind)]))

#-------------------------------------------------------------------------------
#Set neural network parameters
nb_epoch      = 50
learningRate  = 0.001 # 0.001
optimizer     = Adam(learning_rate=learningRate)
batch_size    = 10 #reduced batch-size to accomomdate for 3x larger training dataset size
Height        = 192     # input image dimensions
Width         = 224
max_val       = 1.0

#Defining paths

cnn_path = r'/home/nghiemb/PyMoCo/cnn/3DUNet_SAP'
dpath = cnn_path + r'/weights/PE1_AP/Complex/{}/train_n240_sequential/slices'.format('combo')
wpath = os.path.join(dpath, 'magnitude_half_dataset')
plib.Path(wpath).mkdir(parents=True, exist_ok=True)

datestring = datetime.date.today().strftime("%Y-%m-%d")

#-------------------------------------------------------------------------------
def main():
	print('Reading Data ... ')
	train_data_init = np.load(dpath + r"/train/current_train.npy")
	crop_ind_train = train_data_init.shape[0] // 2
	# crop_ind_train = -1
	train_data = train_data_init[:crop_ind_train,:,:,0]; del train_data_init
	train_before = np.load(dpath + r"/train/before_train.npy")
	train_before = abs(train_before[:crop_ind_train,:,:,0] + 1j*train_before[:crop_ind_train,:,:,1])
	train_after = np.load(dpath + r"/train/after_train.npy")
	train_after = abs(train_after[:crop_ind_train,:,:,0] + 1j*train_after[:crop_ind_train,:,:,1])
	train_label = np.load(dpath + r"/train/current_train_GT.npy")
	train_label = abs(train_label[:crop_ind_train,:,:,0] + 1j*train_label[:crop_ind_train,:,:,1])
	#
	valid_data_init = np.load(dpath + r"/val/current_val.npy")
	crop_ind_val = valid_data_init.shape[0] // 2
	# crop_ind_val = -1
	valid_data = valid_data_init[:crop_ind_val,:,:,0]; del valid_data_init
	valid_before = np.load(dpath + r"/val/before_val.npy")
	valid_before = abs(valid_before[:crop_ind_val,:,:,0] + 1j*valid_before[:crop_ind_val,:,:,1])
	valid_after = np.load(dpath + r"/val/after_val.npy")
	valid_after = abs(valid_after[:crop_ind_val,:,:,0] + 1j*valid_after[:crop_ind_val,:,:,1])
	valid_label = np.load(dpath + r"/val/current_val_GT.npy")
	valid_label = abs(valid_label[:crop_ind_val,:,:,0] + 1j*valid_label[:crop_ind_val,:,:,1])
	#
	print('---------------------------------')
	print('Model Training ...')
	print('---------------------------------')
	#
	model = Correction_Multi_input(Height, Width)
	print(model.summary())
	#---------------------------------------------------------------------------
	#Defining callbacks
	csv_logger = CSVLogger(wpath+r'/Loss_Acc.csv', append=True, separator=' ')
	reduce_lr = LearningRateScheduler(scheduler)
	#
	progress_fname = os.path.join(wpath, datestring + r'_progress.npy')
	save_progress = SaveNetworkProgress(progress_fname)
	checkpoint_fpath = os.path.join(wpath, datestring + r'_weights-{epoch:02d}.hdf5')
	checkpoint = ModelCheckpoint(checkpoint_fpath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
	callbacks_list = [csv_logger, reduce_lr,checkpoint, save_progress]
	#---------------------------------------------------------------------------
	#Model training
	model.compile(loss=ssim_loss, optimizer=optimizer, metrics=[ssim_score,'mse'])
	hist = model.fit(x = [train_before, train_data, train_after],  # train_CE
					y = train_label,
					batch_size = batch_size,
					shuffle = True,#False,
					epochs = nb_epoch, #100,
					verbose = 2,          # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
					validation_data=([valid_before, valid_data, valid_after], valid_label),   # test_CE
					callbacks=callbacks_list)
	print('Saving Model...')
	save_model(wpath, model,'CorrectionUNet_') # to save the weight - 'CNN_iter_'+str(i)
	#


if __name__ == "__main__":
	main()


'''
import numpy as np
import matplotlib.pyplot as plt

loss_array = np.loadtxt('Loss_Acc.csv', dtype=str)

epoch_array = loss_array[1:,0].astype(np.int)+1
train_loss = loss_array[1:,1].astype(np.float)
train_mse = loss_array[1:,2].astype(np.float)
train_ssim = loss_array[1:,3].astype(np.float)
val_loss = loss_array[1:,4].astype(np.float)
val_mse = loss_array[1:,5].astype(np.float)

#Plotting the Loss Function
#Training Loss = (1-SSIM)

plt.figure()
plt.plot(epoch_array, train_loss, label = "Train")
plt.plot(epoch_array, val_loss, label = "Val")
plt.xlabel("Epoch")
plt.ylabel("Loss (1-SSIM)")
plt.title("Training Loss for Magnitude Component")
plt.legend(loc = 'upper right')
plt.show()

#Plotting the MSE
plt.figure()
plt.plot(epoch_array, train_mse, label = "Train")
plt.plot(epoch_array, val_mse, label = "Val")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE for Magnitude Component")
plt.legend(loc = 'upper right')
plt.show()

'''
