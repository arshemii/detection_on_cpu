#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post Processing raw data of DETR model:
    Raw data: outputs[
                    <ConstOutput: names[] shape[1,84,8400] type: f32>
                    ]>>
"""


import numpy as np
from .utils_set import yolo_84_to_6_features
from .utils_set import center_wh_to_xywh
from .utils_set import bbox_scale
from .utils_set import non_max_suppression_fast


def yolo_out_process(res, shape, iou, conf, filter_list):
    res = res(0).data[0]
    # 84 features for 8400 box
    scale = (shape[0]/640, shape[1]/640)

    
    res = yolo_84_to_6_features(res, conf)
    # bbox layout is: xcenter, ycenter, w, h
    # res layout is label, score, bbox
    
    if len(res) == 0:
        return np.empty((0, 6), dtype=np.float32)
    
    
    res[:, 2:] = center_wh_to_xywh(res[:, 2:])
    
    if filter_list:
        res = res[np.isin(res[:, 0], filter_list)]
    
    res[:, 2:] = bbox_scale(res[:, 2:], scale)
    
    unique_labels = np.unique(res[:, 0])
    results = []    
    
    for label in unique_labels:
        results.append(non_max_suppression_fast(res[res[:, 0] == label], iou))
        
    if results:
        results = np.vstack(results).astype(np.float32)
    else:
        results = np.empty((0, 6), dtype=np.float32)
        
    return results
    