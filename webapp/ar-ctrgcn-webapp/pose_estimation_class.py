import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import numpy as np
import math
import json
import torch
import os


class PoseDetector:

    def __init__(self, mode = False, complexity = 2, smooth_lm=True, enable_seg=False, smooth_seg=False, detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.smooth_lm = smooth_lm
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.complexity = complexity

        #self.mpPose = mp.solutions.pose
        #self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth_lm, self.enable_seg, self.smooth_seg, self.detectionCon, self.trackCon)

        base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_heavy.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.modelObj = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5m_Objects365.pt')
        #self.modelObj.to(device)
        self.modelObj.conf = 0.45

        self.groups = json.load(open('groups.json'))
            


    def getCoordinates(self, img):

        """imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)"""

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        self.results = self.detector.detect(image)

        relLmList = [] #relative coordinates
        
        num_keypoint = 17
        rxa=np.zeros(shape=(17,1))
        rya=np.zeros(shape=(17,1))

        if self.results.pose_landmarks:
            
            landmarks = []
            landmarks.append(self.results.pose_landmarks[0][0])
            landmarks.append(self.results.pose_landmarks[0][2])
            landmarks.append(self.results.pose_landmarks[0][5])
            landmarks.append(self.results.pose_landmarks[0][7])
            landmarks.append(self.results.pose_landmarks[0][8])
            landmarks.append(self.results.pose_landmarks[0][11])
            landmarks.append(self.results.pose_landmarks[0][12])
            landmarks.append(self.results.pose_landmarks[0][13])
            landmarks.append(self.results.pose_landmarks[0][14])
            landmarks.append(self.results.pose_landmarks[0][15])
            landmarks.append(self.results.pose_landmarks[0][16])
            landmarks.append(self.results.pose_landmarks[0][23])
            landmarks.append(self.results.pose_landmarks[0][24])
            landmarks.append(self.results.pose_landmarks[0][25])
            landmarks.append(self.results.pose_landmarks[0][26])
            landmarks.append(self.results.pose_landmarks[0][27])
            landmarks.append(self.results.pose_landmarks[0][28])

            for i in range(0,num_keypoint):
                rxa[i]=landmarks[i].x
                rya[i]=landmarks[i].y

                trial1 = np.concatenate((rxa,rya),axis=1)

        image_height, image_width, _ = img.shape
        landmarkCoords = []
        for landmark in landmarks:
            landmarkCoords.append([ landmark.x * image_width, landmark.y * image_height ])
        landmarkCoords = np.array(landmarkCoords)

        xmax = np.amax(landmarkCoords[:,0])
        xmin = np.amin(landmarkCoords[:,0])
        ymax = np.amax(landmarkCoords[:,1])
        ymin = np.amin(landmarkCoords[:,1])

        body_centre = np.array([
            (landmarkCoords[5,0] + landmarkCoords[6,0] + landmarkCoords[11,0] + landmarkCoords[12,0]) / 4,
            (landmarkCoords[5,1] + landmarkCoords[6,1] + landmarkCoords[11,1] + landmarkCoords[12,1]) / 4
        ])

        skel_bbox =[]
        skel_bbox.extend([xmin-50,ymin-50,xmax+50,ymax+50])
        img2 = Image.fromarray(img)
        im = img2.crop(skel_bbox)

        
        #YOLO OBJECT DETECTION
        results = self.modelObj(im)
         
        df=results.pandas().xyxy[0] 

        print("yolo results", df)
        #results.save()
         
        dfiltered=df.loc[df['class'] != 0]

        xomin, xomax, yomin, yomax = 0,0,0,0
        xomin_temp, xomax_temp, yomin_temp, yomax_temp = 0,0,0,0

        obj_class_num_raw = 0  
        nearest_obj_dist = None

        #OBJECT BBOX COORDINATES - RELATIVE TO CROP
        if not dfiltered.empty:
            #for each row in dataframe
            for index, row in dfiltered.iterrows():
                try:
                    xomin_temp=row['xmin']
                    if xomin_temp is None:
                        xomin_temp=0   
            
                    xomax_temp=row['xmax']
                    if xomax_temp is None:
                        xomax_temp=0  
            
                    yomin_temp=row['ymin']
                    if yomin_temp is None:
                        yomin_temp=0  
            
                    yomax_temp=row['ymax']
                    if yomax_temp is None:
                        yomax_temp=0
                except:
                    pass

                object_centre = np.array([xomin_temp + (xomax_temp-xomin_temp)/2, yomin_temp + (yomax_temp-yomin_temp)/2])

                #convert object centre to absolute coordinates
                object_centre[0] = object_centre[0] + skel_bbox[0]
                object_centre[1] = object_centre[1] + skel_bbox[1]

                #object_current_dist = np.linalg.norm(object_centre - body_centre)
                object_current_dist = math.dist(body_centre,object_centre)

                if nearest_obj_dist is None or object_current_dist < nearest_obj_dist:
                    nearest_obj_dist = object_current_dist
                    xomin, xomax, yomin, yomax = xomin_temp, xomax_temp, yomin_temp, yomax_temp
                    obj_class_num_raw = row['class']

      
        """
        #OBJECT CLASS TYPE
        if not dfiltered.empty:
            obj_class_num_raw=dfiltered.iloc[0].at['class']"""
                
        try:
            
            #OBJECT BBOX COORDINATES - ABSOLUTE FRAMEWISE
            xominf = xmin-50+xomin
            xomaxf = xmin-50+xomax
            yominf = ymin-50+yomin
            yomaxf = ymin-50+yomax
              
                    
            #OBJECT CENTER VARIABLES - ABSOLUTE FRAMEWISE
            xoavgf=(xomaxf+xominf)/2
            yoavgf=(yomaxf+yominf)/2

            tmpcenter=[]
            tmpcenter.append(xoavgf)
            tmpcenter.append(yoavgf)
            
            #NORMALIZING OBJECT CENTER COORDINATES
            x_obj_center_norm = xoavgf/image_width
            y_obj_center_norm = yoavgf/image_height
        except:
            pass

          
        for x,y in trial1:
            
            #CALCULATING EUCLIDEAN DISTANCE AND NORMALIZING SKELETON CHECKPOINT COORDINATES
            totdist=math.dist([x_obj_center_norm,y_obj_center_norm],[(x/image_width),(y/image_height)])
            
            obj_class_num_joined = 0
            for i in self.groups:
                if obj_class_num_raw in self.groups[i]:
                    obj_class_num_joined = i
                    break

            relLmList.append([x/image_width, y/image_height, totdist, obj_class_num_joined])
    
        return relLmList     

    def __del__(self):
        #self.pose.close()
        del self.mode
        del self.smooth_lm
        del self.enable_seg
        del self.smooth_seg
        del self.detectionCon
        del self.trackCon
        del self.complexity
        #del self.mpPose
        #del self.pose
