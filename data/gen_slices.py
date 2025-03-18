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

def load_TrainDat_Subrecon(path, mode = 'current', str_out=None): #Loading the training data (s_corrupted)
    print(str_out)
    m_out = np.load(path,allow_pickle=1)[2] #m_files[i], Mtraj, s_corrupted, m_corrupted, m_corrupted_loss
    if m_out.shape[0] == 180: #hard-coding LR dimension for Calgary-Campinas dataset
        m_out = m_out[:,5:-5,...] #choosing to crop outlier 180-LR dim to 170
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
    array_pad = np.pad(array, ((0,0), (pad_x,pad_x), (pad_y,pad_y), (0,0)))
    return array_pad

#-------------------------------------------------------------------------------
#------------------------------------LOAD DATA----------------------------------
#-------------------------------------------------------------------------------
#File paths
dpath = r'/home/nghiemb/Data/CC'
cnn_path = r'/home/nghiemb/PyMoCo/cnn/3DUNet_SAP'
spath_init = cnn_path + r'/weights/PE1_AP/Complex/{}/train_n240_sequential'.format('combo')

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
label_dat = vol2slice(label_dat_init) #transform array of volumes to array of AXIAL slices
del label_dat_init

pad_x = int((np.ceil(label_dat.shape[1]/32) * 32 - label_dat.shape[1])/2)
pad_y = int((np.ceil(label_dat.shape[2]/32) * 32 - label_dat.shape[2])/2)

label_dat = pad_dat(label_dat, pad_x, pad_y)
# label_dat = label_dat[..., None] #Need to add 5th dimension
label_dat = np.tile(label_dat, (nsims*nlvs,1,1,1))
np.save(spath + r"/label_data.npy", label_dat) #18 GB

#Creating datasets for adjacent slices
label_current = gen_AdjSlice(label_dat, shape_val = (60*nsims*nlvs,256//2,192,224,2), mode = 'current')[...,None]
np.save(spath + r"/label_dat_current.npy", label_current) #3.4 GB
del label_dat

#Split into train and validation datasets
ntrain = int(label_current.shape[0] * 0.8); nval = int(label_current.shape[0] * 0.2)
inds_range = [i for i in range(label_current.shape[0])]
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
GT_train_current, GT_val_current = split_dat(label_current, train_inds, val_inds)
del label_current
plib.Path(spath + r"/train").mkdir(parents=True, exist_ok=True)
plib.Path(spath + r"/val").mkdir(parents=True, exist_ok=True)

np.save(spath + r"/train/current_train_GT.npy", GT_train_current)
np.save(spath + r"/val/current_val_GT.npy", GT_val_current)
del GT_train_current, GT_val_current

#-------------------------------------------------------------------------------
#-----------------------------------TRAIN DATA----------------------------------
#-------------------------------------------------------------------------------

nsims_train = 2 #number of sims per subject per motion lv
nlvs = 2 #number of motion levels

dpath_moderate = os.path.join(spath_init, 'moderate') #**************************
dpath_severe = os.path.join(spath_init, 'large') #**************************

print("Loading Corrupted data")
m_files_full_moderate = sorted(glob.glob(dpath_moderate + r'/train_dat*.npy'), key = natural_keys) #alphanumeric order
m_files_moderate = [files for j in range(nsims_train) for files in m_files_full_moderate[j*67:(j+1)*67][:-ntest]]
m_files_full_severe = sorted(glob.glob(dpath_severe + r'/train_dat*.npy'), key = natural_keys) #alphanumeric order
m_files_severe = [files for j in range(nsims_train) for files in m_files_full_severe[j*67:(j+1)*67][:-ntest]]

m_files = [*m_files_moderate, *m_files_severe]

corr_dat_init = [load_TrainDat(m_path, str_out = str(i+1)) for i,m_path in enumerate(m_files)]

pad_x = int((np.ceil(corr_dat_init[0].shape[0]/32) * 32 - corr_dat_init[0].shape[0])/2)
pad_y = int((np.ceil(corr_dat_init[0].shape[1]/32) * 32 - corr_dat_init[0].shape[1])/2)

train_inds = np.load(spath + r"/train_inds.npy")
val_inds = np.load(spath + r"/val_inds.npy")

corr_dat_vol = vol2slice(np.array(corr_dat_init)); del corr_dat_init #transform array of volumes to array of AXIAL slices
corr_dat_pad = pad_dat(corr_dat_vol, pad_x, pad_y); del corr_dat_vol
corr_dat = corr_dat_pad[..., None]; del corr_dat_pad #Need to add 4th dimension for train script
np.save(spath + r"/corr_data.npy", corr_dat) #4 GB if single precision
#
corr_current = gen_AdjSlice(corr_dat, shape_val = (60*nsims_train*nlvs,256//2,192,224,2), mode = 'current')
np.save(spath + r"/corr_dat_current.npy", corr_current)

corr_before = gen_AdjSlice(corr_dat, shape_val = (60*nsims_train*nlvs,256//2,192,224,2), mode = 'before')
np.save(spath + r"/corr_dat_before.npy", corr_before)

corr_after = gen_AdjSlice(corr_dat, shape_val = (60*nsims_train*nlvs,256//2,192,224,2), mode = 'after')
np.save(spath + r"/corr_dat_after.npy", corr_after)
del corr_dat
#

print("Saving the adjacent slices")
#Current slices
train_current, val_current = split_dat(corr_current, train_inds, val_inds)
del corr_current
np.save(spath + r"/train/current_train.npy", train_current) #2.4 G
np.save(spath + r"/val/current_val.npy", val_current) #0.6 G
del train_current, val_current
#
#Before slices
train_before, val_before = split_dat(corr_before, train_inds, val_inds)
del corr_before
np.save(spath + r"/train/before_train.npy", train_before)
np.save(spath + r"/val/before_val.npy", val_before)
del train_before, val_before
#
#After slices
train_after, val_after = split_dat(corr_after, train_inds, val_inds)
del corr_after
np.save(spath + r"/train/after_train.npy", train_after)
np.save(spath + r"/val/after_val.npy", val_after)
del train_after, val_after

'''
#-------------------------------------------------------------------------------
#----------------------------------TESTING DATA---------------------------------
#-------------------------------------------------------------------------------

GT_files = sorted(glob.glob(GT_path + r'/*.npy'))[-ntest:]
mask_files = sorted(glob.glob(mask_path + r'/*.npy'))[-ntest:] #reserve last 7 for test
sens_files = sorted(glob.glob(C_path + r'/*.npy'))[-ntest:]


print("Loading Corrupted data")
m_files_full_moderate = sorted(glob.glob(dpath_moderate + r'/train_dat*.npy'), key = natural_keys) #alphanumeric order
m_files_moderate = [files for j in range(nsims_test) for files in m_files_full_moderate[j*67:(j+1)*67][-ntest:]]
m_files_full_severe = sorted(glob.glob(dpath_severe + r'/train_dat*.npy'), key = natural_keys) #alphanumeric order
m_files_severe = [files for j in range(nsims_test) for files in m_files_full_severe[j*67:(j+1)*67][-ntest:]]

# m_files = [*m_files_moderate, *m_files_severe]
motion_lv = 'severe'
m_files = [*m_files_severe]

# #Loading groundtruth images, with masking based on estimated coil sensitivity profiles
for i in range(len(m_files)):
    t1 = time()
    print('Test {}'.format(i+1))
    mask = load_mask(mask_files[np.mod(i,ntest)])
    label_dat = load_GTDat(GT_files[np.mod(i,ntest)])*mask
    label_dat = vol2slice(label_dat[None,...], crop=0) #transform array of volumes to array of AXIAL slices
    #
    pad_x = int((np.ceil(label_dat.shape[1]/32) * 32 - label_dat.shape[1])/2)
    pad_y = int((np.ceil(label_dat.shape[2]/32) * 32 - label_dat.shape[2])/2)
    #
    label_dat = pad_dat(label_dat, pad_x, pad_y)
    label_dat = label_dat[..., None] #Need to add 4th dimension
    label_current = gen_AdjSlice(label_dat, shape_val = (1,256,192,224,2), mode = 'current')[...,None]
    #
    plib.Path(spath + r"/test/{}/Test{}".format(motion_lv, i+1)).mkdir(parents=True, exist_ok=True)
    #
    np.save(spath + r"/test/{}/Test{}/current_test_GT.npy".format(motion_lv, i+1), label_current)
    # save_Sens(sens_files[np.mod(i,ntest)], spath + r"/test/{}/Test{}/sens.npy".format(motion_lv, i+1))
    # save_Kdat(m_files[i], spath + r"/test/{}/Test{}/s_corrupted.npy".format(motion_lv, i+1))
    # save_Mtraj(m_files[i], spath + r"/test/{}/Test{}/Mtraj.npy".format(motion_lv, i+1))
    save_m_corrupted(m_files[i], spath + r"/test/{}/Test{}/m_corrupted.npy".format(motion_lv, i+1))
    save_U(m_files[i], spath + r"/test/{}/Test{}/U.npy".format(motion_lv, i+1))
    t2 = time()
    print("Elapsed time: {} sec".format(t2 - t1))
    # #
    # #Loading simualted corrupted images (n = 80)
    # corr_dat = load_TrainDat(m_files[i])
    # corr_dat = vol2slice(corr_dat[None,...], crop=0) #transform array of volumes to array of AXIAL slices
    # corr_dat = pad_dat(corr_dat, pad_x, pad_y)
    # corr_dat = corr_dat[..., None] #Need to add 4th dimension
    # corr_current = gen_AdjSlice(corr_dat, shape_val = (1,256,192,224,2), mode = 'current')[...,None]
    # corr_after = gen_AdjSlice(corr_dat, shape_val = (1,256,192,224,2), mode = 'after')[...,None]
    # corr_before = gen_AdjSlice(corr_dat, shape_val = (1,256,192,224,2), mode = 'before')[...,None]
    # np.save(spath + r"/test/{}/Test{}/current_test.npy".format(motion_lv, i+1), corr_current)
    # np.save(spath + r"/test/{}/Test{}/after_test.npy".format(motion_lv, i+1), corr_after)
    # np.save(spath + r"/test/{}/Test{}/before_test.npy".format(motion_lv, i+1), corr_before)

# current_train_GT_sub = current_train_GT[400:500,...]
# current_train_GT_sub = abs(current_train_GT_sub[...,0,0] + 1j*current_train_GT_sub[...,1,0])

# fig, axes = plt.subplots(4,25)
# for i, ax in enumerate(axes.flatten()):
#     ax.imshow(current_train_GT_sub[i], cmap = "gray")
# 
# plt.show()
'''