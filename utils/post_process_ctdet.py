#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post Processing raw data of DETR model:
    Raw data: outputs[
                    <ConstOutput: names[center_heatmap] shape[1,80,128,128] type: f32>,
                    <ConstOutput: names[width_height] shape[1,2,128,128] type: f32>,
                    <ConstOutput: names[regression] shape[1,2,128,128] type: f32>
                    ]>>
"""

import numpy as np
from .utils_set import sigmoid
from .utils_set import nms_loacl_maxima
from .utils_set import bbox_scale

def ctdet_out_process(res, shape, conf, k, filter_list):
    
    scale = (shape[0]/128, shape[1]/128)
    
    hm = sigmoid(res(0).data[0]) # for each pixel in output 128*128
    wh = res(1).data[0]
    reg = res(2).data[0]
    
    hm = nms_loacl_maxima(hm)
    m = len(hm[hm != 0.0])  # Number of non-zeros in hm
    
    if m < k:
        label, y, x = np.unravel_index(np.nonzero(hm), hm.shape)
    else:
        label, y, x = np.unravel_index(np.argsort(hm, axis=None)[-k:], hm.shape)
        
    # hm_maxk_ind (label, y, x)
    scores = hm[label, y, x]  
    w = wh[0][y, x]
    h = wh[1][y, x]    
    reg_x = reg[0][y, x]
    reg_y = reg[1][y, x]
    
    # xmin, ymin, w, h
    results = np.vstack((label, scores, x + reg_x - w/2, y + reg_y - h/2, w, h), dtype=np.float32).T
    
    if filter_list:
        results = results[np.isin(results[:, 0], filter_list)]
    
    results = results[results[:, 1] >= conf]
    if len(results) == 0:
        return np.empty((0, 6), dtype=np.float32)
    
    # does not need further nms for bboxes
    results[:, 2:] = bbox_scale(results[:, 2: ], scale)

    return results