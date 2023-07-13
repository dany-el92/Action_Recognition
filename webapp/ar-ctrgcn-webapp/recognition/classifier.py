#!/usr/bin/env python
from __future__ import print_function

import pickle
import random
import sys
from collections import OrderedDict
import traceback
import numpy as np


# torch
import torch
import torch.nn as nn

from recognition.feeders import tools

from recognition.model.ctrgcn import Model

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def import_class(import_str):
        mod_str, _sep, class_str = import_str.rpartition('.')
        __import__(mod_str)
        try:
            return getattr(sys.modules[mod_str], class_str)
        except AttributeError:
            raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Classifier():

    def __init__(self, workdir, weights, window_size=64, p_interval=[0.5, 1],
                    test_batchsize=256, device=0, model='recognition.model.ctrgcn.Model', model_args=dict()):
        
        model_args['num_class'] = 120
        model_args['num_point'] = 17
        model_args['num_person'] = 2
        model_args['graph'] = 'recognition.graph.ntu_rgb_d.Graph'
        model_args['graph_args'] = dict()
        model_args['graph_args']['labeling_mode'] = 'spatial'


        self.workdir = workdir
        self.weights = weights
        self.window_size = window_size
        self.p_interval = p_interval
        self.test_batchsize = test_batchsize
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model_args = model_args
        self.main_model = None
        self.global_steps = 0


    def load_model(self):

        model = Model(**self.model_args)
        weights = torch.load(self.weights)
        model.load_state_dict(weights)

        self.main_model = model

    
    def eval(self, data):


        self.main_model.eval()
        valid_frame_num = np.sum(data.sum(0).sum(-1).sum(-1) != 0)


        N, T, _ = data.shape
        data = data.reshape((N, T, 2, 17, 4))
        data = data.transpose(0, 4, 1, 3, 2)
        data = tools.valid_crop_resize(data[0], valid_frame_num, self.p_interval, self.window_size)
        
        """from recognition.feeders.bone_pairs import ntu_pairs
        bone_data_numpy = np.zeros_like(data)
        for v1, v2 in ntu_pairs:
            bone_data_numpy[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]
        data = bone_data_numpy"""


        with torch.no_grad():
            data = data[np.newaxis, ...]
            data = torch.tensor(data)
            data = data.float().to(self.device)
            #data = data.float().cuda(0)

            output = self.main_model(data)
            softmax_fn = nn.Softmax(1)
            output = softmax_fn(output)
            scores_desc = torch.argsort(output.data, 1, descending=True)
            top5_idx = scores_desc[0, :5]

            output = output[0, top5_idx]
            #output = output[:, top5_idx]

        return top5_idx.tolist(), output.tolist()


    #
    # def getitem(self):
    #     data_numpy = self.data[index]
    #     label = self.label[index]
    #     data_numpy = np.array(data_numpy)
    #     valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
    #     # reshape Tx(MVC) to CTVM
    #     data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
    #     if self.random_rot:
    #         data_numpy = tools.random_rot(data_numpy)
    #     if self.bone:
    #         from .bone_pairs import ntu_pairs
    #         bone_data_numpy = np.zeros_like(data_numpy)
    #         for v1, v2 in ntu_pairs:
    #             bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
    #         data_numpy = bone_data_numpy
    #     if self.vel:
    #         data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
    #         data_numpy[:, -1] = 0
    #
    #     return data_numpy, label, index