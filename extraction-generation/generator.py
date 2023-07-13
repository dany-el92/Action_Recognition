import argparse
import os
import os.path as osp
from csv import writer

import abc
import cv2
import mmcv
import numpy as np
import torch
from mmcv import DictAction
from PIL import Image
import pandas as pd
import math
import json

from pathlib import Path

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

args = abc.abstractproperty()
args.det_score_thr = 0.45

classes = json.load(open('classes.json'))
groups = json.load(open('groups.json'))

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument(
        '--rgb-stdet-config',
        default=('configs/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py'),
        help='rgb-based spatio temporal detection config file path')
    parser.add_argument(
        '--rgb-stdet-checkpoint',
        default=('https://download.openmmlab.com/mmaction/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb'
                 '_20201217-16378594.pth'),
        help='rgb-based spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--skeleton-stdet-checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'posec3d_ava.pth'),
        help='skeleton-based spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/'
                 'faster_rcnn/faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--skeleton-config',
        default='configs/skeleton/posec3d/'
        'slowonly_r50_u48_240e_ntu120_xsub_keypoint.py',
        help='skeleton-based action recognition config file path')
    parser.add_argument(
        '--skeleton-checkpoint',
        default='https://download.openmmlab.com/mmaction/skeleton/posec3d/'
        'posec3d_k400.pth',
        help='skeleton-based action recognition checkpoint file/url')
    parser.add_argument(
        '--rgb-config',
        default='configs/recognition/tsn/'
        'tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py',
        help='rgb-based action recognition config file path')
    parser.add_argument(
        '--rgb-checkpoint',
        default='https://download.openmmlab.com/mmaction/recognition/'
        'tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/'
        'tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth',
        help='rgb-based action recognition checkpoint file/url')
    parser.add_argument(
        '--use-skeleton-stdet',
        action='store_true',
        help='use skeleton-based spatio temporal detection method')
    parser.add_argument(
        '--use-skeleton-recog',
        action='store_true',
        help='use skeleton-based action recognition method')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.45,
        help='the threshold of human detection score')
    parser.add_argument(
        '--action-score-thr',
        type=float,
        default=0.45,
        help='the threshold of action prediction score')
    parser.add_argument(
        '--video',
        default='',
        help='video file/url')
    parser.add_argument(
        '--label-map-stdet',
        default='tools/data/ava/label_map.txt',
        help='label map file for spatio-temporal action detection')
    parser.add_argument(
        '--label-map',
        default='tools/data/kinetics/label_map_k400.txt',
        help='label map file for action recognition')
    parser.add_argument(
        '--device', type=str, default='cuda', help='CPU/CUDA device option')
    parser.add_argument(
        '--out-filename',
        default='demo/test_stdet_recognition_output.mp4',
        help='output filename')
    parser.add_argument(
        '--predict-stepsize',
        default=8,
        type=int,
        help='give out a spatio-temporal detection prediction per n frames')
    parser.add_argument(
        '--output-stepsize',
        default=1,
        type=int,
        help=('show one frame per n frames in the demo, we should have: '
              'predict_stepsize % output_stepsize == 0'))
    parser.add_argument(
        '--output-fps',
        default=24,
        type=int,
        help='the fps of demo video output')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name



def main():
    args = parse_args()

    #model = torch.hub.load('ultralytics/yolov5', 'yolov5x6') 
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5m_Objects365.pt') 
    model.conf = 0.45
    coord = pd.read_pickle('ntu120_xsub_val.pkl')
    coord = pd.DataFrame(coord)
    coordalt = pd.read_pickle('ntu120_xsub_train.pkl')
    coordalt = pd.DataFrame(coordalt)
    
    
    rgb_stdet_config = mmcv.Config.fromfile(args.rgb_stdet_config)
    rgb_stdet_config.merge_from_dict(args.cfg_options)
        
    directory='./tmp'
    subdirs_names_array=[]
    for path, subdirs, files in os.walk(directory):
        for name in subdirs:
            subdirs_names=(os.path.join(path, name))   
            subdirs_names_array.append(subdirs_names)
    
    for x in subdirs_names_array:

        framepathtmp=x
        if Path(framepathtmp.replace("tmp","runs")).is_dir():
            print("skipping directory:",framepathtmp)
            continue

        print("ANALYZING VIDEO:",framepathtmp)
        vidnametmp = os.path.split(x)
        tail=vidnametmp[1]
        vidname = (tail).rsplit('_', 1)[0]
        
        
        tmp_action_code = (x).rsplit('_', 1)[0]
        action_code = tmp_action_code[-4:]
        
        model.classes = classes[action_code]
                    
        if not os.path.exists(osp.join('./runs', osp.basename(osp.splitext(x)[0]))):
            os.makedirs(osp.join('./runs', osp.basename(osp.splitext(x)[0])))
            
        consistency_log=open('./runs/inconsistent_dftodetect_skeleton_count_log.txt','a')    
        #classes_txt=open('./runs/detected_classes.txt','a')    
         
            
        distances_csv_joined = osp.join('./runs', osp.basename(osp.splitext(x)[0]), 'distances.csv')
        
        
        if os.path.exists(distances_csv_joined):
            os.remove(distances_csv_joined)
            
 
        num_frame=len(os.listdir(framepathtmp))

          
        frame_tmpl = osp.join(osp.splitext(x)[0], 'img_{0:06d}.jpg') 
        
        
        with open(distances_csv_joined, 'a+', newline='') as write_distances_joined:
            distances_csv_writer_joined = writer(write_distances_joined)

            cnt=0
            df_control=0
            for frame in os.listdir(framepathtmp):   
                    
                    #print("outer frame loop:",frame)

                    for frame in os.listdir(framepathtmp): 

                        #print("inner frame loop:",frame)
 
                        try:                         
                                       
                            if(cnt==num_frame):
                                     #print("frame cnt control MAX REACHED:",cnt)
                                 break
                             
                            frame_tmpl_passed=frame_tmpl.format(cnt+1)
                              
                            # print("\n --------------- STARTING FRAME ANALYSIS FOR: ",frame_tmpl_passed)
                            frame_pass=Image.open(frame_tmpl_passed)
                     
                             
                             
                             #print("vidname=",vidname)
                             #print(coord.loc[coord['frame_dir']==vidname]['keypoint'])
                            
                            coord2 = pd.DataFrame(coord.loc[coord['frame_dir']==vidname]['keypoint'])
                            if coord2.empty:
                                 coord2 = pd.DataFrame(coordalt.loc[coordalt['frame_dir']==vidname]['keypoint'])
                            
                                
                            
                            coord3 = coord2.values[0]
                             
                            coord4 = coord3[0]
                             
                            tmpx=coord4[:,cnt,:,0]
          
                             #control for .pkl detected skeletons number 
                            for x in coord4:
                                 df_control+=1
                             #print("dataframe skeleton rows:",df_control)  
                             
                            tmpx2=tmpx.T
                             
                            tmpy=coord4[:,cnt,:,1]
                            tmpy2=tmpy.T
                                        
                             #SKELETON COORDINATES
                            xmax = np.amax(tmpx)             
                            xmin = np.amin(tmpx)                
                            ymax = np.amax(tmpy)                  
                            ymin = np.amin(tmpy)                  
                             
                            skel_bbox =[]
                            skel_bbox.extend([xmin-50,ymin-50,xmax+50,ymax+50])
                             #print("\n skeleton_bbox, expanded by 50:",skel_bbox)
          
                             #PASSING SKELETON BBOX CROP 
                            im = frame_pass.crop(skel_bbox)
                             #YOLO OBJECT DETECTION
                            results = model(im)
                             
                            df=results.pandas().xyxy[0] 
                            #results.save()
                             
                             #EXCLUDING PERSON FROM YOLO DETECTION
                            dfiltered=df.loc[df['class'] != 0]
                                         
                                 #print("\n detection results: \n")
                                 #print(results.pandas().xyxy[0])
                        except:
                            missing_error=open('./runs/error.txt','a')
                            missing_error.write(framepathtmp + "\n")
                            missing_error.close()
                            cnt+=1
                            break
                                         
                        try:
                           # print("\n filtered results (person excluded):")
                           # print(df.loc[df['class'] != 0])              
                       
                                                                    
                            dfiltered=df.loc[df['class'] != 0]
                            
                            #OBJECT BBOX COORDINATES - RELATIVE TO CROP
                            xomin=dfiltered.iloc[0].at['xmin']
                            
                            if xomin is None:
                                xomin=0   
     
                            xomax=dfiltered.iloc[0].at['xmax']
                            if xomax is None:
                                xomax=0  
     
                            yomin=dfiltered.iloc[0].at['ymin']
                            if yomin is None:
                                yomin=0  
     
                            yomax=dfiltered.iloc[0].at['ymax']
                            if yomax is None:
                                yomax=0
                            

                            
                            
                            #OBJECT CLASS TYPE
                            obj_class_num_raw=dfiltered.iloc[0].at['class']
                            
                            #OBJECT BBOX COORDINATES - ABSOLUTE FRAMEWISE
                            xominf = xmin-50+xomin
                            xomaxf = xmin-50+xomax
                            yominf = ymin-50+yomin
                            yomaxf = ymin-50+yomax
                                

                                      
                            #OBJECT CENTER VARIABLES - ABSOLUTE FRAMEWISE
                            xoavgf=(xomaxf+xominf)/2
                           # print("\n X object center: ")
                          #  print(xoavgf)
                            yoavgf=(yomaxf+yominf)/2
                           # print("Y object center: ")
                           # print(yoavgf)

                            
                            #NORMALIZING OBJECT CENTER COORDINATES
                            x_obj_center_norm = xoavgf/1920
                            y_obj_center_norm = yoavgf/1080
                            
                        except:
                            pass
                                   
                        trial1 = np.concatenate((tmpx2,tmpy2),axis=1)
     
                        totdistances_joined=[]
                        
                        if dfiltered.empty:
                            #print("\n ONLY PEOPLE DETECTED, APPENDING EMPTY DATA ROWS \n")
                            distances_csv_writer_joined.writerow(np.zeros(18))
                            cnt+=1
                            break
                        
                       
                        for x,y,*rest in trial1:
                            #CALCULATING EUCLIDEAN DISTANCE AND NORMALIZING SKELETON CHECKPOINT COORDINATES
                            totdist=math.dist([x_obj_center_norm,y_obj_center_norm],[(x/1920),(y/1080)])
                            totdistances_joined.append(totdist)
             
                        obj_class_num_joined = 0
                        #JOINING OBJECT CLASS CODES BY ACTION CODES
                        for i in groups:
                            if obj_class_num_raw in groups[i]:
                                obj_class_num_joined = i
                                break
                        
                        totdistances_joined.append(obj_class_num_joined) 
                         
                        #appending detected object class number to detected classes log file
                        #classes_txt.write(str(obj_class_num) + "\n")
                        
                        #print("\n skeleton keypoint distances from object center:",totdistances_raw)
                        distances_csv_writer_joined.writerow(totdistances_joined)
                       # print("--------------- END OF FRAME ANALYSIS ---------------")
       
                        target_dir2= osp.join('./runs', vidname,'img_{:06d}.jpg')
                        target_dir2 = target_dir2.format(cnt+1)
                        #results.save(save_dir=target_dir2)
     
                        cnt+=1

                    if(cnt==num_frame):
                         #control for frame count
                         #print(".pkl dataframe skeletons df control:",df_control)
                        # print("frame cnt control:",cnt)
                         
                         if df_control<cnt:
                             if df_control>0:
                                 consistency_log.write(str(x) + "\n")
                                 cnt_min_fix=0
                                 tmp_diff=abs(df_control-cnt)
                                 while tmp_diff>cnt_min_fix:
                                     ffix2=open(distances_csv_joined, 'r+')
                                     lines2=ffix2.readlines()
                                     lines2.pop()
                                     ffix2.close()
                                     ffix2=open(distances_csv_joined, 'w+')
                                     ffix2.writelines(lines2)
                                     ffix2.close()
                                     tmp_diff=tmp_diff-1
                             break        
                         break 
                     
                         if df_control>cnt:
                             if df_control>0:
                                 consistency_log.write(str(x) + "\n")
                                 cnt_fix=0
                                 tmp_diff=abs(df_control-cnt)
                                 while tmp_diff>cnt_fix:
                                     totdistances_joined.append(np.zeros(18))
                                     tmp_diff=tmp_diff-1
                                 break    
                         break  
                    
                    
            torch.cuda.empty_cache()
            write_distances_joined.close()
            consistency_log.close()
            #classes_txt.close()

    #deleting duplicates from detected classes log file
    #with open('./runs/detected_classes.txt', "r") as fc:
    #    lines = dict.fromkeys(fc.readlines())
    #with open('./runs/detected_classes.txt', "w") as fc:
     #   fc.writelines(lines)
    
    #fc.close()    
    #classes_txt.close()

    with open('./runs/inconsistent_dftodetect_skeleton_count_log.txt', "r") as fc2:
        lines = dict.fromkeys(fc2.readlines())
    with open('./runs/inconsistent_dftodetect_skeleton_count_log.txt', "w") as fc2:
        fc2.writelines(lines)

    fc2.close()
    consistency_log.close()
    
    if os.path.exists('./runs/error.txt'):
        with open('./runs/error.txt', "r") as fc3:
            lines = dict.fromkeys(fc3.readlines())
        with open('./runs/error.txt', "w") as fc3:
            fc3.writelines(lines)
    
        fc3.close()
    
if __name__ == '__main__':
    main()
