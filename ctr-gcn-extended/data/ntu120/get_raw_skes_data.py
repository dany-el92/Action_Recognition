# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os.path as osp
import os
import numpy as np
import pickle
import pandas as pd
from numba import jit, cuda

coord = pd.read_pickle('ntu120_xsub_val.pkl')
coord = pd.DataFrame(coord)
coordalt = pd.read_pickle('ntu120_xsub_train.pkl')
coordalt = pd.DataFrame(coordalt)

@jit(target_backend='cuda')                                                
                                          
def get_raw_bodies_data(vidname):
    
        print("Video: ",vidname)

        coord2 = pd.DataFrame(coord.loc[coord['frame_dir']==vidname]['total_frames'])
        if coord2.empty:
             coord2 = pd.DataFrame(coordalt.loc[coordalt['frame_dir']==vidname]['total_frames'])      
    
        if coord2.empty:
            num_frames=0
        else:
            num_frames=coord2.values[0]
        print("Total frames: ",num_frames)
    
        bodies_data = dict()
        valid_frames = -1  # 0-based index

        #print("1")
        
        """for root,dirs,files in os.walk('./runs'):
            #print("2")
            for file in files:
                #print("3")
                root_tmp=os.path.split(root)
                passed=root_tmp[1]
                passed_split=passed.split('_')[0]
                #print("4")
                if passed_split==vidname:
                    #print("5")
                    if os.path.getsize(os.path.join(root,'distances.csv'))>0:
                        #print("6")
                        tmpfile=pd.read_csv(os.path.join(root,'distances.csv'),header=None)
                        #print("7")
                        break"""
        
        tmpfile = None
        if num_frames > 0:
            tmpfile = pd.read_csv(f"runs/{vidname}_rgb/distances.csv",header=None)
       
        #print("8")
        framecnt=0
        while framecnt<num_frames:
            #print("Frame: ",framecnt)
            #current_line += 1      
            valid_frames += 1
    
            bodyID=0
            
            coord3 = pd.DataFrame(coord.loc[coord['frame_dir']==vidname]['keypoint'])
            if coord3.empty:
                 coord3 = pd.DataFrame(coordalt.loc[coordalt['frame_dir']==vidname]['keypoint']) 

            joints = np.zeros((1, 17, 4), dtype=np.float32)
            coord4 = coord3.values[0]
            coord5 = coord4[0]

            tmpx=coord5[:,framecnt,:,0]
            tmpy=coord5[:,framecnt,:,1]

            #tmpboth=coord5[:,framecnt,:,:]
            tmpx2=pd.DataFrame(tmpx)
            tmpy2=pd.DataFrame(tmpy)  
            #tmpboth2=pd.DataFrame(tmpboth)
            #tmpdist=pd.DataFrame(tmpfile.iloc[framecnt,j])
            #print(tmpdist.iloc[framecnt,0])

            num_joints=17
            for j in range(num_joints): 

                joints[0, j, 0] = np.array(tmpx2[j].iloc[0], dtype=np.float32)  
                joints[0, j, 1] = np.array(tmpy2[j].iloc[0], dtype=np.float32)
                #joints[0, j, :] = np.array(tmpboth2[j], dtype=np.float32)
                joints[0, j, 2] = np.array(tmpfile.iloc[framecnt,j])
                joints[0, j, 3] = np.array(tmpfile.iloc[framecnt,-1])

            if bodyID not in bodies_data:                
                body_data = dict()
                body_data['joints'] = joints[0, np.newaxis]  # ndarray: (1, 17, 4)
                body_data['interval'] = [valid_frames]  # the index of the first frame
            else:    
            # Stack each body's data of each frame along the frame order      
                body_data['joints'] = np.vstack((body_data['joints'], joints[0, np.newaxis]))
                pre_frame_idx = body_data['interval'][-1]
                body_data['interval'].append(pre_frame_idx + 1)  # add a new frame index
       
            bodies_data[bodyID] = body_data  # Update bodies_data
            framecnt+=1
            
        return {'name': vidname, 'data': bodies_data, 'num_frames': num_frames}


def get_raw_skes_data():
    # # save_path = './data'
    # # skes_path = '/data/pengfei/NTU/nturgb+d_skeletons/'
    # stat_path = osp.join(save_path, 'statistics')
    #
    # skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
    # save_data_pkl = osp.join(save_path, 'raw_skes_data.pkl')

    skes_name = np.loadtxt(skes_name_file, dtype=str)
    

    num_files = skes_name.size
    print('Found %d available skeleton files.' % num_files)

    raw_skes_data = []
    frames_cnt = np.zeros(num_files, dtype=int)

    print(skes_name)
    for (idx, ske_name) in enumerate(skes_name):     
      #if ske_name == 'S001C003P005R002A005':
        bodies_data = get_raw_bodies_data(ske_name)
        raw_skes_data.append(bodies_data)
        frames_cnt[idx] = bodies_data['num_frames'] 
        print('Processed:',ske_name)
        if (idx + 1) % 20 == 0:
            print('Processed: %.2f%% (%d / %d)' % \
                  (100.0 * (idx + 1) / num_files, idx + 1, num_files))
                  
    with open(save_data_pkl, 'wb') as fw:
        pickle.dump(raw_skes_data, fw, pickle.HIGHEST_PROTOCOL)
    np.savetxt(osp.join(save_path, 'raw_data', 'frames_cnt.txt'), frames_cnt, fmt='%d')

    print('Saved raw bodies data into %s' % save_data_pkl)
    print('Total frames: %d' % np.sum(frames_cnt))

  
if __name__ == '__main__':
    save_path = './'

    stat_path = osp.join(save_path, 'statistics')
    if not osp.exists('./raw_data'):
        os.makedirs('./raw_data')

    skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
    save_data_pkl = osp.join(save_path, 'raw_data', 'raw_skes_data.pkl')

    get_raw_skes_data()