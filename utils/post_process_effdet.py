#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post Processing raw data of DETR model:
    Raw data: outputs[
                    <ConstOutput: names[detections:0, detections] shape[1,1,100,7] type: f32>
                    ]>>
"""

import numpy as np
from .utils_set import xyxy_wh_to_xywh
from .utils_set import bbox_scale
from .utils_set import non_max_suppression_fast


def effdet_out_process(res, shape, iou, conf, nms, filter_list):
    res = res(0).data[0][0][:, 1:]
    res = res[res[:, 1] != 0]
    # Res will be in shape: [100, 6]
        # label
        # score
        # x_min
        # y_min
        # x_max
        # y_max
    # by efault res is filtered by: iou = 0.6, conf = 0.2  
    if filter_list:
        res = res[np.isin(res[:, 0], filter_list)]
    
    if len(res) == 0:
        return np.empty((0, 6), dtype=np.float32)
    
    res[:, 2:] = xyxy_wh_to_xywh(res[:, 2:])
    res[:, 2:] = bbox_scale(res[:, 2:], shape)
    
    
    if iou >= 0.6 or conf <= 0.2 or nms == False:
        # why in this situation nms shall be done???
        return res
    
    res = res[res[:, 1] >= conf]
    
    if len(res) == 0:
        return np.empty((0, 6), dtype=np.float32)
    
    unique_labels = np.unique(res[:, 0])
    results = []    
    
    for label in unique_labels:
        results.append(non_max_suppression_fast(res[res[:, 0] == label], iou))
        
    if results:
        results = np.vstack(results).astype(np.float32)
    else:
        results = np.empty((0, 6), dtype=np.float32)
        
    
    return results



