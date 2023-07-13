import sys
import numpy as np

sys.path.extend(['../'])
from recognition.graph import tools

num_node = 17
#num_node = 25
self_link = [(i, i) for i in range(num_node)]
"""inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]"""

"""inward_ori_index = [(1, 2), (2, 3), (4, 3), (8, 4), (5, 1), (6, 5), (7, 6), (9, 7), (10, 11), (13, 12),
                    (12, 14), (16, 14), (15, 13), (17, 15),
                    (25, 13), (27, 25), (29, 27), (24, 25), (24, 12), (26, 24), (28, 26),
                    (22, 16), (20, 16), (18, 20), (16, 18), (23, 17), (21, 17), (19, 21), (17, 19),
                    (32, 28), (30, 32), (28, 30), (33, 29), (31, 33), (29, 31)]"""

inward_ori_index = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (6, 7),
    (5, 7), (6, 12), (7, 13), (6, 8), (8, 10), (7, 9),
    (9, 11), (12, 13), (12, 14), (14, 16), (13, 15), (15, 17)]


inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
