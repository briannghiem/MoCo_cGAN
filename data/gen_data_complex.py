'''
Created: Feb 16, 2025

T1w MPRAGE Calgary-Campinas Data
Create training dataset with simulated 3D motion

Details:
- loading data from 4 healthy subjects, from Calgary-Campinas Dataset
- with each subject, simulate 20 motion cases
- motion ranges from +/- 7 mm or deg; for each motion trajectory, the max extent
  is randomly selected from aforementioned range, and the trajectory is then
  randomly generated given the selected max
- motion trajectory has temporal resolution of 8 TRs
- using a sequentially-ordered Cartesian pattern, with R = 2

'''

import os
os.chdir('/home/nghiemb/PyMoCo')

import pathlib as plib
from time import time
from functools import partial
import numpy as np

import jax
import jax.numpy as xp
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="0" #turn off GPU pre-allocation
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import encode.encode_op as eop
import recon.recon_op as rec
import cnn.run_unet as cnn
import utils.metrics as mtc
import motion.motion_sim as msi

#-------------------------------------------------------------------------------
#Helper functions

#Helper Functions
def _gen_traj_dof(rand_key, motion_lv, dof, nshots, motion_specs):
    '''
    Input:
        rand_key=jax.random.PRNGKey object,
        motion_lv={'mild','moderate','severe'},
        dof={'Tx','Ty','Tz','Rx','Ry','Rz'}
        nshots=int # of motion states
    Output:
        xp.array of motion trajectory for a given DOF
    '''
    p_val = motion_specs[motion_lv][dof][1]
    p_array = xp.array([p_val/2, 1-p_val, p_val/2])
    opts = xp.array([-1,0,1]) #move back, stay, move fwd
    maxval = motion_specs[motion_lv][dof][0]
    minval = maxval / 2
    array = jax.random.choice(rand_key, a = opts, shape=(nshots-1,), p = p_array) #binary array
    array = xp.concatenate((xp.array([0]), array)) #ensure first motion state is origin
    vals = jax.random.uniform(rand_key, shape=(nshots,),minval=minval, maxval=maxval) #displacements
    return xp.cumsum(array * vals) #absolute value of motion trajectory

def _gen_traj(rand_keys, motion_lv, nshots, motion_specs):
    '''
    Input:
        rand_key=jax.random.PRNGKey object,
        motion_lv={'mild','moderate','severe'},
        nshots=int # of motion states
    Output:
        xp.array of motion trajectory across all 6 DOFs
    '''
    out_array = xp.zeros((nshots, 6))
    out_array = out_array.at[:,0].set(_gen_traj_dof(rand_keys[0], motion_lv, 'Tx', nshots, motion_specs))
    out_array = out_array.at[:,1].set(_gen_traj_dof(rand_keys[1], motion_lv, 'Ty', nshots, motion_specs))
    out_array = out_array.at[:,2].set(_gen_traj_dof(rand_keys[2], motion_lv, 'Tz', nshots, motion_specs))
    out_array = out_array.at[:,3].set(_gen_traj_dof(rand_keys[3], motion_lv, 'Rx', nshots, motion_specs))
    out_array = out_array.at[:,4].set(_gen_traj_dof(rand_keys[4], motion_lv, 'Ry', nshots, motion_specs))
    out_array = out_array.at[:,5].set(_gen_traj_dof(rand_keys[5], motion_lv, 'Rz', nshots, motion_specs))
    return out_array

def _gen_seq(i,j,k,dof):
    a1 = (i+j+dof+1)**2 + (5*j)**2 + (17*i)**2 + (k*1206)**2 #including the exponent to guarantee different random value than training dataset
    return a1

def _gen_key(i, j, k):
    return [jax.random.PRNGKey(_gen_seq(i,j,k,dof)) for dof in range(6)]

#-------------------------------------------------------------------------------
#Set up motion sim specs
# motion_lv_list = ['mild', 'moderate', 'severe']
mild_specs = {'Tx':[0.1,0.1],'Ty':[0.2,0.15],'Tz':[0.2,0.15],\
            'Rx':[0.2,0.15],'Ry':[0.1,0.1],'Rz':[0.1,0.1]} #[max_rate, prob]
moderate_specs = {'Tx':[0.2,0.1],'Ty':[0.4,0.2],'Tz':[0.4,0.2],\
            'Rx':[0.5,0.2],'Ry':[0.2,0.1],'Rz':[0.2,0.1]} #[max_rate, prob]
severe_specs1 = {'Tx':[0.4,0.15],'Ty':[0.9,0.3],'Tz':[0.9,0.3],\
            'Rx':[1,0.3],'Ry':[0.5,0.15],'Rz':[0.5,0.15]} #[max_rate, prob]
severe_specs2 = {'Tx':[0.8,0.3],'Ty':[1.8,0.6],'Tz':[1.8,0.6],\
            'Rx':[2,0.6],'Ry':[1.0,0.3],'Rz':[1.0,0.3]} #Double the max_rate and probability
severe_specs3 = {'Tx':[1.6,0.6],'Ty':[3.6,1.0],'Tz':[3.6,1.0],\
            'Rx':[4,1.0],'Ry':[2.0,0.6],'Rz':[2.0,0.6]} #Quadruple the probability
# motion_specs = {'mild':mild_specs,'moderate':moderate_specs,'severe':severe_specs}
motion_specs = {'mild':moderate_specs,'moderate':severe_specs1,\
                'large':severe_specs2,'extreme':severe_specs3}

motion_lv_list = ['mild', 'moderate']

#-------------------------------------------------------------------------------
#-------------------------Image Acquisition Simulation--------------------------
#-------------------------------------------------------------------------------
#Load data
dpath = r'/home/nghiemb/Data/CC'
# cnn_path = r'/home/nghiemb/PyMoCo/cnn/3DUNet_SAP'
# spath = cnn_path + r'/weights/PE1_AP/Complex/{}/train_n240_sequential'.format('combo')
spath = r'/home/nghiemb/RMC_repos/MoCo_cGAN/data/training_dataset'
m_files = sorted(os.listdir(os.path.join(dpath,'m_complex'))) #alphanumeric order
C_files = sorted(os.listdir(os.path.join(dpath, 'sens')))

nsims = 2 # number of motion simulations per subject per motion level
IQM_store = []
#-------------------------------------------------------------------------------
t1 = time()
count = 1
for i in range(len(m_files)):
    t3 = time()
    print("Subject {}".format(str(i+1)))
    #---------------------------------------------------------------------------
    #Load data
    m_fname = os.path.join(dpath,'m_complex',m_files[i])
    C_fname = os.path.join(dpath,'sens',C_files[i])
    m_GT = xp.load(m_fname)
    C = xp.load(C_fname); C = xp.transpose(C, axes = (3,2,0,1))
    res = xp.array([1,1,1])
    #
    m_GT = m_GT / xp.max(abs(m_GT.flatten())) #rescale
    mask = rec.getMask(C)
    plib.Path(os.path.join(dpath,'mask')).mkdir(parents=True, exist_ok=True)
    mask_name = os.path.join(dpath,'mask','mask_' + m_files[i][10:])
    xp.save(mask_name, mask)
    #---------------------------------------------------------------------------
    #Sampling pattern, for Calgary-Campinas brain data (12 coils, [PE:218,RO:256,SL:170])
    PE1 = m_GT.shape[0] #LR
    PE2 = m_GT.shape[1] #AP
    RO = m_GT.shape[2] #SI
    # #
    # Rs = 1
    # TR_shot = 16
    # order = 'sequential'
    # U_array = xp.transpose(msi.make_samp(xp.transpose(m_GT, (1,0,2)), \
    #                                 Rs, TR_shot, order='sequential'), (0,2,1,3)).astype('int16')
    # U = eop._U_Array2List(U_array, m_GT.shape)
    #
    #Using same sampling pattern as SAP UNet (versions trained in 2024, before I padded image to 256x256x192 in 2025)
    m_U = xp.zeros((PE2, PE1, RO))
    n_states_init = 10
    Rs = 1
    TR_shot = int(xp.round(PE1 / (Rs*n_states_init)))
    order = 'interleaved'
    U_array = xp.transpose(msi.make_samp(m_U, Rs, TR_shot, order), axes = (0,2,1,3)) #PE1 along AP
    del m_U
    U = eop._U_Array2List(U_array, m_GT.shape)
    #
    #---------------------------------------------------------------------------
    #Generate motion trace
    for j, motion_lv in enumerate(motion_lv_list):
        for k in range(nsims):
            print("Sim {} for Subject {}".format(str(j+1), str(i+1)))
            rand_keys = _gen_key(i, j, k)
            Mtraj_GT = _gen_traj(rand_keys, motion_lv, len(U), motion_specs)
            Mtraj_init = xp.zeros((len(U), 6))
            #-------------------------------------------------------------------
            R_pad = (10,10,10)
            batch = 1
            s_corrupted = eop.Encode(m_GT, C, U, Mtraj_GT, res, batch=batch)
            #-------------------------------------------------------------------
            #Reconstruct image via EH, since data is fully-sampled
            m_corrupted = eop.Encode_Adj(s_corrupted, C, U, Mtraj_init, res, batch=batch)
            m_corrupted_PE = mtc.evalPE(m_corrupted, m_GT, mask=mask)
            m_corrupted_SSIM = mtc.evalSSIM(m_corrupted, m_GT, mask=mask)
            m_corrupted_loss = [m_corrupted_PE, m_corrupted_SSIM]
            IQM_store.append(m_corrupted_loss)
            xp.save(spath + r'/IQM_store.npy', IQM_store)
            #
            #save the filename, motion trajectory, simulated k-space and image
            output = [m_files[i], Mtraj_GT, s_corrupted, m_corrupted, m_corrupted_loss, U]
            s_path_temp = os.path.join(spath, motion_lv)
            plib.Path(s_path_temp).mkdir(parents=True, exist_ok=True)
            xp.save(s_path_temp + r'/train_dat{}.npy'.format((i+1)+(k*67)), output)
        #
    #
    t4 = time()
    print("Time elapsed for Subject {}: {} sec".format(str(i+1), str(t4 - t3)))
    #
    #

print("Finished simulating training data")
t2 = time()
print("Total elapsed time: {} min".format(str((t2 - t1)/60)))





'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def plot_views(img, vmax = 1.0):
    if vmax == "auto": #if auto, set as max val of volume
        vmax = abs(img.flatten().detach().cpu()).max()
    #
    fig, axes = plt.subplots(1,3)
    for i, ax in enumerate(axes):
        if i==0:
            ax.imshow(img[img.shape[0]//2,:,:], cmap = "gray", vmax = vmax)
        if i==1:
            ax.imshow(img[:,img.shape[1]//2,:], cmap = "gray", vmax = vmax)
        if i==2:
            ax.imshow(img[:,:,img.shape[2]//2], cmap = "gray", vmax = vmax)
        #
    plt.show()


def plot_Mtraj(T_GT, R_GT, T, R, img_dims, rescale = 0, spath = None):
    Nx, Ny, Nz = img_dims
    if rescale:
        Tx_scale = (Nx/2)
        Ty_scale = (Ny/2)
        Tz_scale = (Nz/2)
        R_scale = 1/(np.pi/180)
    else:
        Tx_scale = 1; Ty_scale = 1; Tz_scale = 1
        R_scale = 1
    #
    plt.figure()
    plt.plot(T_GT[0]*Tx_scale, '--r', alpha = 0.75, label="Tx - GT")
    plt.plot(T_GT[1]*Ty_scale, '--b', alpha = 0.75, label="Ty - GT")
    plt.plot(T_GT[2]*Tz_scale, '--g', alpha = 0.75, label="Tz - GT")
    plt.plot(T[0]*Tx_scale, 'r', label="Tx")
    plt.plot(T[1]*Ty_scale, 'b', label="Ty")
    plt.plot(T[2]*Tz_scale, 'g', label="Tz")
    # plt.legend(loc="lower left")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel("Translations (mm)")
    plt.xlabel("Shot Index")
    if spath is None:
        plt.show()
    else:
        plt.savefig(spath + r'_Translations.png')
    #
    plt.figure()
    plt.plot(R_GT[0]*R_scale, '--r', alpha = 0.75, label="Rx - GT")
    plt.plot(R_GT[1]*R_scale, '--b', alpha = 0.75, label="Ry - GT")
    plt.plot(R_GT[2]*R_scale, '--g', alpha = 0.75, label="Rz - GT")
    plt.plot(R[0]*R_scale, 'r', label="Rx")
    plt.plot(R[1]*R_scale, 'b', label="Ry")
    plt.plot(R[2]*R_scale, 'g', label="Rz")
    # plt.legend(loc="upper left")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel("Rotations (deg)")
    plt.xlabel("Shot Index")
    if spath is None:
        plt.show()
    else:
        plt.savefig(spath + r'_Rotations.png')

        

import numpy as np
dat = np.load("train_dat2.npy", allow_pickle=1)
# dat = np.load("train_dat69.npy", allow_pickle=1)

m_corrupted = dat[3]
Nx = m_corrupted.shape[0]
Ny = m_corrupted.shape[1]
Nz = m_corrupted.shape[2]
img_dims = [Nx,Ny,Nz]

plot_views(abs(m_corrupted))


Mtraj_GT = dat[1]; Mtraj_final = Mtraj_GT
T_GT = [Mtraj_GT[:,0], Mtraj_GT[:,1], Mtraj_GT[:,2]]
R_GT = [Mtraj_GT[:,3], Mtraj_GT[:,4], Mtraj_GT[:,5]]

T_final = [Mtraj_final[:,0], Mtraj_final[:,1], Mtraj_final[:,2]]
R_final = [Mtraj_final[:,3], Mtraj_final[:,4], Mtraj_final[:,5]]

plot_Mtraj(T_GT, R_GT, T_final, R_final, img_dims, rescale = 0)


'''


'''

import os
import glob
import re
import numpy as np
import pathlib as plib
from time import time

#-------------------------------------------------------------------------------
#Helper functions
#Loading npy files with proper alphanumeric sorting
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

#-------------------------------------------------------------------------------

nsims_train = 2 #number of sims per subject per motion lv
nlvs = 2 #number of motion levels
ntest = 7

mpath = r'/cluster/projects/uludag/Brian/data/cc/train_3D'
dpath_moderate = os.path.join(mpath, 'corrupted', 'PE1_AP', 'Complex', 'moderate') #**************************
dpath_severe = os.path.join(mpath, 'corrupted', 'PE1_AP', 'Complex', 'severe') #**************************

print("Loading Corrupted data")
m_files_full_moderate = sorted(glob.glob(dpath_moderate + r'/train_dat*.npy'), key = natural_keys) #alphanumeric order
m_files_moderate = [files for j in range(nsims_train) for files in m_files_full_moderate[j*67:(j+1)*67][:-ntest]]
m_files_full_severe = sorted(glob.glob(dpath_severe + r'/train_dat*.npy'), key = natural_keys) #alphanumeric order
m_files_severe = [files for j in range(nsims_train) for files in m_files_full_severe[j*67:(j+1)*67][:-ntest]]

m_files = [*m_files_moderate, *m_files_severe]

iter = 1
PE_severe = []
Mtraj_severe = []
for name in m_files_severe:
    print("Loading File {}".format(iter))
    dat_temp = np.load(name, allow_pickle=1)
    PE_severe.append(dat_temp[-2])
    Mtraj_severe.append(dat_temp[1])
    iter+=1

PE_severe = np.asarray(PE_severe)
np.save(dpath_severe + r'/PE_severe.npy', PE_severe)
np.save(dpath_severe + r'/Mtraj_severe.npy', Mtraj_severe)

iter = 1
PE_moderate = []
Mtraj_moderate = []
for name in m_files_moderate:
    print("Loading File {}".format(iter))
    dat_temp = np.load(name, allow_pickle=1)
    PE_moderate.append(dat_temp[-2])
    Mtraj_moderate.append(dat_temp[1])
    iter+=1

PE_moderate = np.asarray(PE_moderate)
np.save(dpath_moderate + r'/PE_moderate.npy', PE_moderate)
np.save(dpath_moderate + r'/Mtraj_moderate.npy', Mtraj_moderate)


'''


'''
import numpy as np
import matplotlib.pyplot as plt

def plot_views(m):
    fig, axes = plt.subplots(1,3)
    for i, ax in enumerate(axes):
        if i==0:
            temp1 = ax.imshow(m[m.shape[0]//2,:,:], cmap = "gray")
            fig.colorbar(temp1, ax=ax)
        if i==1:
            temp2 = ax.imshow(m[:,m.shape[1]//2,:], cmap = "gray")
            fig.colorbar(temp2, ax=ax)
        elif i==2:
            temp3 = ax.imshow(m[:,:,m.shape[2]//2], cmap = "gray")
            fig.colorbar(temp3, ax=ax)
        #
    #
    plt.show()

train_dat = np.load("train_dat2.npy", allow_pickle=1)
m_path, Mtraj, s_corrupted, m_corrupted, m_corrupted_loss, U = train_dat

m_corrupted_mag = abs(m_corrupted)
m_corrupted_phase = np.angle(m_corrupted)

plot_views(m_corrupted_mag)
plot_views(m_corrupted_phase)



# scp brian.nghiem@h4huhndata1.uhnresearch.ca:/cluster/projects/uludag/Brian/data/cc/train_3D/corrupted/PE1_AP/Complex/severe/train_dat2.npy .

'''