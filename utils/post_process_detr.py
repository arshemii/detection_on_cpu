#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post Processing raw data of DETR model:
    Raw data: outputs[
                    <ConstOutput: names[scores] shape[1,100,92] type: f32>,
                    <ConstOutput: names[boxes] shape[1,100,4] type: f32>
                    ]>> 
"""

import numpy as np
from .utils_set import softmax, center_wh_to_xywh
from .utils_set import bbox_scale
from .utils_set import non_max_suppression_fast


def detr_out_process(res, shape, iou, conf, filter_list):
    # Operations:
        # obtains logits with res(0)
        # convert to numpy f16
        # softmax on classes for each bbox
        # drop last class --> no object
    logits = softmax(res(0).data[0].astype(np.float32), axis=-1)[:, :-1]
    boxes = res(1).data[0].astype(np.float32)
    # box layout:
        # x_center normalized
        # y_center normalized
        # w normalized
        # h normalized
    boxes = center_wh_to_xywh(boxes)
    boxes = bbox_scale(boxes, shape)
        
    scores = np.max(logits, axis = -1, keepdims=True)
    labels = np.argmax(logits, axis = -1, keepdims=True)
    
    res = np.hstack((labels, scores, boxes), dtype = np.float32)
    
    if filter_list:
        res = res[np.isin(res[:, 0], filter_list)]
    
    res = res[res[:, 1] >= conf]
    
    del logits, scores, labels, boxes
    
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

            

        
        
