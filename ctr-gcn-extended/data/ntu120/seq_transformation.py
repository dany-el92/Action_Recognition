# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import os.path as osp
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

root_path = './'
stat_path = osp.join(root_path, 'statistics')
setup_file = osp.join(stat_path, 'setup.txt')
camera_file = osp.join(stat_path, 'camera.txt')
performer_file = osp.join(stat_path, 'performer.txt')
replication_file = osp.join(stat_path, 'replication.txt')
label_file = osp.join(stat_path, 'label.txt')
skes_name_file = osp.join(stat_path, 'skes_available_name.txt')

denoised_path = osp.join(root_path, 'denoised_data')
raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
frames_file = osp.join(denoised_path, 'frames_cnt.txt')

save_path = './'


if not osp.exists(save_path):
    os.mkdir(save_path)


def remove_nan_frames(ske_name, ske_joints, nan_logger):
    num_frames = ske_joints.shape[0]
    valid_frames = []

    for f in range(num_frames):
        if not np.any(np.isnan(ske_joints[f])):
            valid_frames.append(f)
        else:
            nan_indices = np.where(np.isnan(ske_joints[f]))[0]
            nan_logger.info('{}\t{:^5}\t{}'.format(ske_name, f + 1, nan_indices))

    return ske_joints[valid_frames]

def seq_translation(skes_joints):
    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        #print(ske_joints.shape)
        #print(ske_joints[:,0])
        i = 0  # get the "real" first frame of actor1
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


def align_frames(skes_joints, frames_cnt):
    """
    Align all sequences with the same frame length.

    """
    num_skes = len(skes_joints)
    max_num_frames = frames_cnt.max()
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 136), dtype=np.float32)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]

        aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints, np.zeros_like(ske_joints)))

    return aligned_skes_joints


def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 120))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1

    return labels_vector


def split_train_val(train_indices, method='sklearn', ratio=0.05):
    """
    Get validation set by splitting data randomly from training set with two methods.
    In fact, I thought these two methods are equal as they got the same performance.

    """
    if method == 'sklearn':
        return train_test_split(train_indices, test_size=ratio, random_state=10000)
    else:
        np.random.seed(10000)
        np.random.shuffle(train_indices)
        val_num_skes = int(np.ceil(0.05 * len(train_indices)))
        val_indices = train_indices[:val_num_skes]
        train_indices = train_indices[val_num_skes:]
        return train_indices, val_indices


def split_dataset(skes_joints, label, performer, camera, evaluation, save_path):
    train_indices, test_indices = get_indices(performer, camera, evaluation)
    m = 'sklearn'  # 'sklearn' or 'numpy'
    # Select validation set from training set
    #train_indices, val_indices = split_train_val(train_indices, m)

    # Save labels and num_frames for each sequence of each data set
    train_labels = label[train_indices]
    test_labels = label[test_indices]
    
    train_x = skes_joints[train_indices]
    train_y = one_hot_vector(train_labels)
    test_x = skes_joints[test_indices]
    test_y = one_hot_vector(test_labels)

    save_name = 'NTU120_%s.npz' % evaluation
    np.savez(save_name, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)

    # Save data into a .h5 file
    # h5file = h5py.File(osp.join(save_path, 'NTU_%s.h5' % (evaluation)), 'w')
    # Training set
    # h5file.create_dataset('x', data=skes_joints[train_indices])
    # train_one_hot_labels = one_hot_vector(train_labels)
    # h5file.create_dataset('y', data=train_one_hot_labels)
    # Validation set
    # h5file.create_dataset('valid_x', data=skes_joints[val_indices])
    # val_one_hot_labels = one_hot_vector(val_labels)
    # h5file.create_dataset('valid_y', data=val_one_hot_labels)
    # Test set
    # h5file.create_dataset('test_x', data=skes_joints[test_indices])
    # test_one_hot_labels = one_hot_vector(test_labels)
    # h5file.create_dataset('test_y', data=test_one_hot_labels)

    # h5file.close()


def get_indices(performer, camera, evaluation='CS'):
    test_indices = np.empty(0)
    train_indices = np.empty(0)

    if evaluation == 'CS':  # Cross Subject (Subject IDs)
        train_ids = [ 1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19,
                     41, 43, 44, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                    64, 65, 66, 68, 102, 103, 104, 105, 106]
        test_ids = [ 3, 6, 7, 10, 11, 12, 20, 21, 22, 23,
                    24, 26, 29, 30, 32, 33, 36, 37, 39, 40,
                    61, 62, 67]

        # Get indices of test data
        for idx in test_ids:
            temp = np.where(performer == idx)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(performer == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(int)
    else:  # Cross View (Camera IDs)
        train_ids = [2, 3]
        test_ids = 1
        # Get indices of test data
        temp = np.where(camera == test_ids)[0]  # 0-based index
        test_indices = np.hstack((test_indices, temp)).astype(int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(camera == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(int)
     
    return train_indices, test_indices


if __name__ == '__main__':
    camera = np.loadtxt(camera_file, dtype=int)  # camera id: 1, 2, 3
    performer = np.loadtxt(performer_file, dtype=int)  # subject id: 1~40
    label = np.loadtxt(label_file, dtype=int) - 1  # action label: 0~59

    frames_cnt = np.loadtxt(frames_file, dtype=int)  # frames_cnt
    skes_name = np.loadtxt(skes_name_file, dtype=np.string_)

    with open(raw_skes_joints_pkl, 'rb') as fr:
        skes_joints = pickle.load(fr)  # a list

    skes_joints = seq_translation(skes_joints)

    skes_joints = align_frames(skes_joints, frames_cnt)  # aligned to the same frame length

    evaluations = ['CS', 'CV']
    for evaluation in evaluations:
        split_dataset(skes_joints, label, performer, camera, evaluation, save_path)
