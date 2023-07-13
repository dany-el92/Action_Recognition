# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 13:28:24 2023

@author: admin
"""
import os.path as osp
import numpy as np
import pandas as pd
import os                



outfile = "cleaned_file.txt"
stat_path = './statistics'
skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
infile = skes_name_file

delete_list = []
   
coord = pd.read_pickle('ntu120_xsub_val.pkl')
coord = pd.DataFrame(coord)
coordalt = pd.read_pickle('ntu120_xsub_train.pkl')
coordalt = pd.DataFrame(coordalt)

with open("analyzed_videos.txt", "r") as txt_file:
    for line in txt_file:
        
        line=line.rstrip('\n')
        print("analyzing:",line)
       
        
        coord2 = pd.DataFrame(coord.loc[coord['frame_dir']==line]['total_frames'])
        if coord2.empty:
             coord2 = pd.DataFrame(coordalt.loc[coordalt['frame_dir']==line]['total_frames'])      
        
        if coord2.empty:
            missing_error=open('./missing_videos.txt','a')
            missing_error.write(line + "\n")
            missing_error.close()
            delete_list.append(line)
        
with open(infile) as fin, open(outfile, "w+") as fout:
    for line in fin:
        for word in delete_list:
            line = line.replace(word, "")
        fout.write(line)