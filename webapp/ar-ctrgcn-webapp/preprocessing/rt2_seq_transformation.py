import os
import os.path as osp
import numpy as np
import pickle
import logging
import h5py
#from sklearn.model_selection import train_test_split

num_joints = 17

def align_frames(skes_joints):
    num_skes = len(skes_joints)
    #max_num_frames = 300
    max_num_frames = skes_joints[0].shape[0]
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, (num_joints * 4 * 2)), dtype=np.float32)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        """num_bodies = 1 if ske_joints.shape[1] == (num_joints * 3) else 2
        if num_bodies == 1:
            aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints,
                                                               np.zeros_like(ske_joints)))[:300, :]
        else:
            aligned_skes_joints[idx, :num_frames] = ske_joints"""
        aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints, np.zeros_like(ske_joints)))[:max_num_frames, :]


    return aligned_skes_joints

def seq_translation(skes_joints):

    for idx, ske_joints in enumerate(skes_joints):
        
        num_frames = ske_joints.shape[0]
        
        i = 0 
        while i < num_frames:
            if np.any(ske_joints[i,:] != 0):
                break
            i += 1
        
        centroid = (ske_joints[:,20:22] + ske_joints[:,24:26] + ske_joints[:,44:46] + ske_joints[:,48:50]) / 4
    
        for f in range(num_frames):   
                ske_joints[f,:2] -= centroid[f]
                ske_joints[f,4:6] -= centroid[f]
                ske_joints[f,8:10] -= centroid[f]
                ske_joints[f,12:14] -= centroid[f]
                ske_joints[f,16:18] -= centroid[f]
                ske_joints[f,20:22] -= centroid[f]
                ske_joints[f,24:26] -= centroid[f]
                ske_joints[f,28:30] -= centroid[f]
                ske_joints[f,32:34] -= centroid[f]
                ske_joints[f,36:38] -= centroid[f]
                ske_joints[f,40:42] -= centroid[f]
                ske_joints[f,44:46] -= centroid[f]
                ske_joints[f,48:50] -= centroid[f]
                ske_joints[f,52:54] -= centroid[f]
                ske_joints[f,56:58] -= centroid[f]
                ske_joints[f,60:62] -= centroid[f]
                ske_joints[f,64:66] -= centroid[f]

        skes_joints[idx] = ske_joints  # Update
        
    return skes_joints 

def transform(skes_joints):
    skes_joints = seq_translation(skes_joints)
    skes_joints = align_frames(skes_joints)

    return skes_joints

if __name__ == '__main__':
    pass