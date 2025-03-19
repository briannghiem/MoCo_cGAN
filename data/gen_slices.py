'''
Created August 6, 2023

Generating training dataset for Stacked U-Nets with Self-Assisted Priors **Modified to take in and output complex images
(Al-masni et al, 2022, https://github.com/Yonsei-MILab/MRI-Motion-Artifact-Correction-Self-Assisted-Priors)

Loading simulated motion-corrupted image volumes (see gen_data_h4h.py, data stored in /cluster/projects/uludag/Brian/data/cc/train_3D/corrupted)
Creating training dataset of axial slices

*On h4h cluster, needed the following allocation: salloc -p veryhimem -c 4 -t 2:00:00 --mem 100G
'''

import os
import glob
import re
import numpy as np
import pathlib as plib
from time import time

#-------------------------------------------------------------------------------
#Loading npy files with proper alphanumeric sorting
def atoi(text):
    '''
    From https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    '''
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    From https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

#-------------------------------------------------------------------------------
#Helper functions
def load_mask(path):
    mask = np.load(path)
    if mask.shape[0] == 180: #hard-coding LR dimension for Calgary-Campinas dataset
        mask = mask[5:-5,...] #choosing to crop outlier 180-LR dim to 170
    return mask

def load_GTDat(path, mode = 'current', str_out=None): #Loading the groundtruth image data
    print(str_out)
    m_out = np.load(path) #no need to transpose for this new dataset
    # m_out = np.transpose(np.load(path), axes = (2,0,1))
    if m_out.shape[0] == 180: #hard-coding LR dimension for Calgary-Campinas dataset
        m_out = m_out[5:-5,...] #choosing to crop outlier 180-LR dim to 170
    m_out /= np.max(abs(m_out.flatten()))
    return m_out

def load_TrainDat(path, mode = 'current', str_out=None): #Loading the training data
    print(str_out)
    m_out = np.load(path,allow_pickle=1)[3] #m_files[i], Mtraj, s_corrupted, m_corrupted, m_corrupted_loss
    if m_out.shape[0] == 180: #hard-coding LR dimension for Calgary-Campinas dataset
        m_out = m_out[5:-5,...] #choosing to crop outlier 180-LR dim to 170
    return m_out

def save_Sens(dpath, spath): #Loading the coil sensitivity profiles
    C_out = np.load(dpath)
    C_out = np.transpose(C_out, axes = (3,2,0,1))
    np.save(spath, C_out)

def save_Kdat(dpath, spath): #Loading the corrupted k-space data
    s_out = np.load(dpath,allow_pickle=1)[2]
    np.save(spath, s_out)

def save_Mtraj(dpath, spath): #Loading the training data
    Mtraj_out = np.load(dpath,allow_pickle=1)[1]
    np.save(spath, Mtraj_out)

def save_m_corrupted(dpath, spath): #Loading the training data
    m_corrupted = np.load(dpath,allow_pickle=1)[3]
    np.save(spath, m_corrupted)

def save_U(dpath, spath): #Loading the training data
    U = np.load(dpath,allow_pickle=1)[5]
    np.save(spath, U)

#---------------------------------------------------------
def slice_mode(array, mode): #for generating datasets for adjacent
    if mode == 'current':
        array = array[:,1:-1,...]
    elif mode == 'before':
        array = array[:,:-2,...]
    elif mode == 'after':
        array = array[:,2:,...]
    return array

def vol2slice(array, crop=True): #transform array of volumes to AXIAL slices
    array = np.transpose(array, axes = (0,3,1,2)) #Nsubjects, SI, LR, AP
    if crop:
        n_SI = array.shape[1]
        array = array[:,n_SI//4:-n_SI//4,:,:] #cropping out top and bottom quarter of axial slices
    array = array.reshape((array.shape[0] * array.shape[1],array.shape[2], array.shape[3]))
    array_out = np.concatenate((np.real(array)[..., None], np.imag(array)[..., None]), axis = 3)
    return array_out

def vol2slab(array, nslices = 8, crop=True): #transform array of volumes to AXIAL slices
    '''
    For the MoCo cGAN network (Johnson and Drangova, MRM 2019)
    Need to subdivide volumes into slabs of 8 axial slices
    '''
    array = np.transpose(array, axes = (0,3,1,2)) #Nsubjects, SI, LR, AP
    if crop:
        n_SI = array.shape[1]
        array = array[:,n_SI//4:-n_SI//4,:,:] #cropping out top and bottom quarter of axial slices
    n_SI_cropped = array.shape[1]
    n_subjects = array.shape[0]
    nslabs = int(n_SI_cropped / nslices)
    array = array.reshape((array.shape[0] * nslabs, nslices, array.shape[2], array.shape[3])) #concatenate all axial slabs from all subjects
    array_out = np.concatenate((np.real(array)[..., None], np.imag(array)[..., None]), axis = -1)
    return array_out

def gen_AdjSlice(array, shape_val = (60,256//2,192,224), mode = 'current'):
    array_reshape = array.reshape(shape_val)
    array_crop = slice_mode(array_reshape, mode)
    array_slices = array_crop.reshape((array_crop.shape[0] * array_crop.shape[1],array_crop.shape[2], array_crop.shape[3], array_crop.shape[4]))
    return array_slices

def split_dat(array, train_inds, val_inds):
    train_array = array[train_inds,...]
    val_array = array[val_inds,...]
    return train_array, val_array

def pad_dat(array, pad_x, pad_y):
    array_pad = np.pad(array, ((0,0), (0,0), (pad_x,pad_x), (pad_y,pad_y), (0,0)))
    return array_pad

#-------------------------------------------------------------------------------
#------------------------------------LOAD DATA----------------------------------
#-------------------------------------------------------------------------------
#File paths
dpath = r'/home/nghiemb/Data/CC'
spath_init = r'/home/nghiemb/RMC_repos/MoCo_cGAN/data/training_dataset'

GT_path = os.path.join(dpath,'m_complex')
mask_path = os.path.join(dpath, 'mask')
C_path = os.path.join(dpath, 'sens')

spath = os.path.join(spath_init, 'slices')
plib.Path(spath).mkdir(parents=True, exist_ok=True)

nsims = 2 #number of sims per subject per motion lv
nlvs = 2 #number of motion levels
ntest = 7

GT_files = sorted(glob.glob(GT_path + r'/*.npy'))[:-ntest]
mask_files = sorted(glob.glob(mask_path + r'/*.npy'))[:-ntest] #reserve last 7 for test
sens_files = sorted(glob.glob(C_path + r'/*.npy'))[:-ntest]


#-------------------------------------------------------------------------------
#-----------------------------------LABEL DATA----------------------------------
#-------------------------------------------------------------------------------
#Loading groundtruth images, with masking based on estimated coil sensitivity profiles
print("Loading masks")
mask_store = [load_mask(mask_path) for mask_path in mask_files]

print("Loading Groundtruth data")
label_dat_init = np.array([load_GTDat(GT_fname, str_out = str(i+1))*mask_store[i] for i, GT_fname in enumerate(GT_files)]) #masked groundtruth image
label_dat = vol2slab(label_dat_init) #transform array of volumes to array of AXIAL slabs (stacks of 8 slices)
del label_dat_init

pad_x = int((np.ceil(label_dat.shape[2]/32) * 32 - label_dat.shape[2])/2)
pad_y = int((np.ceil(label_dat.shape[3]/32) * 32 - label_dat.shape[3])/2)

label_dat = pad_dat(label_dat, pad_x, pad_y)
# label_dat = label_dat[..., None] #Need to add 5th dimension
label_dat = np.tile(label_dat, (nsims*nlvs,1,1,1,1))
np.save(spath + r"/label_data.npy", label_dat) #18 GB


#Split into train and validation datasets
ntrain = int(label_dat.shape[0] * 0.8); nval = int(label_dat.shape[0] * 0.2)
inds_range = [i for i in range(label_dat.shape[0])]
#
train_inds = np.random.choice(inds_range, ntrain, replace=0).tolist()
val_inds = list(set(inds_range) - set(train_inds))
#
np.save(spath + r"/train_inds.npy", train_inds) #NB. reuse same train_inds!
np.save(spath + r"/val_inds.npy", val_inds)
#

#If not first time generating dataset, then reload indices
train_inds = np.load(spath + r"/train_inds.npy")
val_inds = np.load(spath + r"/val_inds.npy")

#Label dataset
GT_train, GT_val = split_dat(label_dat, train_inds, val_inds)
del label_dat
plib.Path(spath + r"/train").mkdir(parents=True, exist_ok=True)
plib.Path(spath + r"/val").mkdir(parents=True, exist_ok=True)

np.save(spath + r"/train/train_GT.npy", GT_train)
np.save(spath + r"/val/val_GT.npy", GT_val)
del GT_train, GT_val

#-------------------------------------------------------------------------------
#-----------------------------------TRAIN DATA----------------------------------
#-------------------------------------------------------------------------------

nsims_train = 2 #number of sims per subject per motion lv
nlvs = 2 #number of motion levels

dpath_moderate = os.path.join(spath_init, 'moderate') #**************************
dpath_mild = os.path.join(spath_init, 'mild') #**************************

print("Loading Corrupted data")
m_files_full_moderate = sorted(glob.glob(dpath_moderate + r'/train_dat*.npy'), key = natural_keys) #alphanumeric order
# m_files_moderate = [files for j in range(nsims_train) for files in m_files_full_moderate[j*67:(j+1)*67][:-ntest]] ####I did not simulate motion for the reserved test subjects, so no need for this line
m_files_full_mild = sorted(glob.glob(dpath_mild + r'/train_dat*.npy'), key = natural_keys) #alphanumeric order
# m_files_mild = [files for j in range(nsims_train) for files in m_files_full_mild[j*67:(j+1)*67][:-ntest]] ####I did not simulate motion for the reserved test subjects, so no need for this line

m_files = [*m_files_full_moderate, *m_files_full_mild]

corr_dat_init = [load_TrainDat(m_path, str_out = str(i+1)) for i,m_path in enumerate(m_files)]

pad_x = int((np.ceil(corr_dat_init[0].shape[0]/32) * 32 - corr_dat_init[0].shape[0])/2) #pad LR dim from 170 to 192
pad_y = int((np.ceil(corr_dat_init[0].shape[1]/32) * 32 - corr_dat_init[0].shape[1])/2) #pad AP dim from 218 to 224

train_inds = np.load(spath + r"/train_inds.npy")
val_inds = np.load(spath + r"/val_inds.npy")

corr_dat_vol = vol2slab(np.array(corr_dat_init)); del corr_dat_init #transform array of volumes to slabs of AXIAL slices
corr_dat_pad = pad_dat(corr_dat_vol, pad_x, pad_y); del corr_dat_vol
# corr_dat = corr_dat_pad[..., None]; del corr_dat_pad #Need to add 4th dimension for train script
corr_dat = corr_dat_pad
np.save(spath + r"/corr_data.npy", corr_dat) #4 GB if single precision
#

#---------------------------------------
print("Saving the adjacent slices")
#Current slices
train_corr, val_corr = split_dat(corr_dat, train_inds, val_inds)
del corr_dat
np.save(spath + r"/train/train_corr.npy", train_corr) #2.4 G
np.save(spath + r"/val/val_corr.npy", val_corr) #0.6 G
del train_corr, val_corr
