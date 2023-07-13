import os
import os.path as osp
import numpy as np
import pickle
import logging


num_joints = 17
noise_spr_thres1 = 0.8
noise_spr_thres2 = 0.69754
noise_len_thres = 11

def denoising_by_length(ske_name, bodies_data):
    noise_info = str()
    new_bodies_data = bodies_data.copy()
    for (bodyID, body_data) in new_bodies_data.items():
        length = len(body_data['interval'])
        if length <= noise_len_thres:
            noise_info += 'Filter out: %s, %d (length).\n' % (bodyID, length)
            del bodies_data[bodyID]

    if noise_info != '':
        noise_info += '\n'

    return bodies_data, noise_info

def get_valid_frames_by_spread(points):
    num_frames = points.shape[0]
    valid_frames = []
    for i in range(num_frames):
        x = points[i, :, 0]
        y = points[i, :, 1]
        if (x.max() - x.min()) <= noise_spr_thres1 * (y.max() - y.min()):  # 0.8
            valid_frames.append(i)
    return valid_frames

def denoising_by_spread(ske_name, bodies_data):
    noise_info = str()
    denoised_by_spr = False  # mark if this sequence has been processed by spread.
    new_bodies_data = bodies_data.copy()
    for (bodyID, body_data) in new_bodies_data.items():
        if len(bodies_data) == 1:
            break
        valid_frames = get_valid_frames_by_spread(body_data['joints'].reshape(-1, num_joints, 3))
        num_frames = len(body_data['interval'])
        num_noise = num_frames - len(valid_frames)
        if num_noise == 0:
            continue
        ratio = num_noise / float(num_frames)
        motion = body_data['motion']
        if ratio >= noise_spr_thres2:  # 0.69754
            del bodies_data[bodyID]
            denoised_by_spr = True
            noise_info += 'Filter out: %s (spread rate >= %.2f).\n' % (bodyID, noise_spr_thres2)

        else:
            joints = body_data['joints'].reshape(-1, num_joints, 3)[valid_frames]
            body_data['motion'] = min(motion, np.sum(np.var(joints.reshape(-1, 3), axis=0)))
            noise_info += '%s: motion %.6f -> %.6f\n' % (bodyID, motion, body_data['motion'])

        if noise_info != '':
            noise_info += '\n'

    return bodies_data, noise_info, denoised_by_spr

def denoising_bodies_data(bodies_data):

    ske_name = bodies_data['name']
    bodies_data = bodies_data['data']

    # Step 1: Denoising based on frame length.
    bodies_data, noise_info_len = denoising_by_length(ske_name, bodies_data)
    if len(bodies_data) == 1:  # only has one bodyID left after step 1
        return bodies_data.items(), noise_info_len

    # Step 2: Denoising based on spread.
    bodies_data, noise_info_spr, denoised_by_spr = denoising_by_spread(ske_name, bodies_data)

    if len(bodies_data) == 1:
        return bodies_data.items(), noise_info_len + noise_info_spr

    bodies_motion = dict()  # get body motion
    for (bodyID, body_data) in bodies_data.items():
        bodies_motion[bodyID] = body_data['motion']

    # Sort bodies based on the motion
    # bodies_motion = sorted(bodies_motion.items(), key=lambda x, y: cmp(x[1], y[1]), reverse=True)
    bodies_motion = sorted(bodies_motion.items(), key=lambda x: x[1], reverse=True)
    denoised_bodies_data = list()
    for (bodyID, _) in bodies_motion:
        denoised_bodies_data.append((bodyID, bodies_data[bodyID]))

    return denoised_bodies_data, noise_info_len + noise_info_spr

def remove_missing_frames(ske_name, joints, colors):
    num_frames = joints.shape[0]
    num_bodies = colors.shape[1]
    if num_bodies == 2:
        missing_indices_1 = np.where(joints[:, :(num_joints * 3)].sum(axis=1) == 0)[0]
        missing_indices_2 = np.where(joints[:, (num_joints * 3):].sum(axis=1) == 0)[0]
        cnt1 = len(missing_indices_1)
        cnt2 = len(missing_indices_2)
        start = 1 if 0 in missing_indices_1 else 0
        end = 1 if num_frames - 1 in missing_indices_1 else 0
        if max(cnt1, cnt2) > 0:
            if cnt1 > cnt2:
                info = '{}\t{:^10d}\t{:^6d}\t{:^6d}\t{:^5d}\t{:^3d}'.format(ske_name, num_frames,
                                                                            cnt1, cnt2, start, end)
                #missing_skes_logger1.info(info)
            else:
                info = '{}\t{:^10d}\t{:^6d}\t{:^6d}'.format(ske_name, num_frames, cnt1, cnt2)
                #missing_skes_logger2.info(info)

    valid_indices = np.where(joints.sum(axis=1) != 0)[0]  # 0-based index
    missing_indices = np.where(joints.sum(axis=1) == 0)[0]
    num_missing = len(missing_indices)
    if num_missing > 0:
        joints = joints[valid_indices]
        #colors[missing_indices] = np.nan
        global missing_count
        missing_count += 1
    colors = []
    return joints, colors

def get_two_actors_points(bodies_data):
    ske_name = bodies_data['name']
    num_frames = bodies_data['num_frames']
    bodies_data, noise_info = denoising_bodies_data(bodies_data)  # Denoising data


    bodies_data = list(bodies_data)
    #if len(bodies_data) == 1:
    if 1 == 1:
        bodyID, body_data = bodies_data[0]
        joints, colors = get_one_actor_points(body_data, num_frames)

    else:
        sp = num_joints * 3 * 2  # njoints * xyz * 2 actors
        joints = np.zeros((num_frames, sp), dtype=np.float32)
        #colors = np.ones((num_frames, 2, num_joints, 2), dtype=np.float32) * np.nan
        bodyID, actor1 = bodies_data[0]
        start1, end1 = actor1['interval'][0], actor1['interval'][-1]
        joints[start1:end1 + 1, :(num_joints * 3)] = actor1['joints'].reshape(-1, (num_joints * 3))
        #colors[start1:end1 + 1, 0] = actor1['colors']
        del bodies_data[0]

        start2, end2 = [0, 0]
        while len(bodies_data) > 0:
            bodyID, actor = bodies_data[0]
            start, end = actor['interval'][0], actor['interval'][-1]
            if min(end1, end) - max(start1, start) <= 0:
                joints[start:end + 1, :(num_joints*3)] = actor['joints'].reshape(-1, (num_joints*3))
                #colors[start:end + 1, 0] = actor['colors']
                start1 = min(start, start1)
                end1 = max(end, end1)

            elif min(end2, end) - max(start2, start) <= 0:
                joints[start:end + 1, (num_joints * 3):] = actor['joints'].reshape(-1, (num_joints * 3))
                #colors[start:end + 1, 1] = actor['colors']
                start2 = min(start, start2)
                end2 = max(end, end2)
            del bodies_data[0]
    colors = []
    return joints, np.array(colors)

def get_one_actor_points(body_data, num_frames):

    joints = np.zeros((num_frames, (num_joints * 4)), dtype=np.float32)
    #colors = np.ones((num_frames, 1, num_joints, 2), dtype=np.float32) * np.nan
    start, end = body_data['interval'][0], body_data['interval'][-1]
    joints[start:end + 1] = body_data['joints'].reshape(-1, (num_joints * 4))
    #colors[start:end + 1, 0] = body_data['colors']

    return joints

def get_raw_denoised_data(bodies_data):
    num_bodies = len(bodies_data['data'])
    #if num_bodies == 1:
    if num_bodies == 1:
        num_frames = bodies_data['num_frames']
        body_data = list(bodies_data['data'].values())[0]
        joints = get_one_actor_points(body_data, num_frames[0])

    return joints

if __name__ == '__main__':
    pass