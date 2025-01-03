#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script defines functions for different heads after object detection

All detections in this script are filtered out
"""
import numpy as np
from . import utils_set as ut
import cv2




def print_report(detections, class_names, shape):
    """
    This function prints the detections for each frame
    
    
    Args:
        detections: A numpy array float32 of shape [N, length of F]
                    N is number of detections in a frame
                    F is features: label, score, x, y, w, and h
                    ** x and y are wrt the frame top-left
                    ** If depends on size, provide in original format
        class_names is the list
        shape: shape (h, w) of the frame : 480*640

    """
    
    if not np.shape(detections)[0] == 0:
        
        print("#########################################################")
        for dets in detections:
            detections = ut.topleft_to_center(dets, shape)
            # Converting to the center of the object
            dets[2] = dets[2] + dets[4]/2
            dets[3] = dets[5] + dets[5]/2
            if class_names:
                print(f"{class_names[int(dets[0])]} in x: {int(dets[2])}, y: {int(dets[3])}, with score: {dets[1]:.2f}")
            else:
                print(f"Class ID: {int(dets[0])} in x: {int(dets[2])}, y: {int(dets[3])}, with score: {dets[1]:.2f}")
            
    else:
        print("No object detected in the frame")
            
            
            
def capture_annotation_no_depth(detections, rgb_frame, class_names):
    """
    This functions creates annotations in the array of RGB image.
    
    Arguments:
        detections: A numpy array float32 of shape [N, length of F]
                    N is number of detections in a frame
                    F is features: label, score, x, y, w, and h
                    ** x and y are wrt the frame top-left
                    ** If depends on size, provide in original format
        rgb_frame: NHWC RGB captured frame in uint8 or 16 numpy array dtype
        class_names is the list
        
        
    Returns:
        annot_array: an array of layout HWC with bboxes

    """
    annot_array = rgb_frame[0]
    del rgb_frame
    for dets in detections:
        
        cv2.rectangle(annot_array, (int(dets[2]), int(dets[3]))
                      ,(int(dets[2] + dets[4]), int(dets[3] + dets[5]))
                      , (0, 0, 255), 1, cv2.LINE_4)
    
        if class_names:
            cv2.putText(annot_array
                        , f"{class_names[int(dets[0])]}"
                        , (int(dets[2]), int(dets[3]+3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5
                        , (0, 0, 255), 1, cv2.LINE_4)
        else:
            cv2.putText(annot_array
                        , f"Class ID: {int(dets[0])}"
                        , (int(dets[2]), int(dets[3]+3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5
                        , (0, 0, 255), 1, cv2.LINE_4)
    
    return annot_array




