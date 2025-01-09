#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script defines functions for different heads after object detection

All detections in this script are filtered out


"""
import numpy as np
from . import utils_set as ut
import cv2


def add_obj_map(top_array, obj, depth_frame, rgb_ints):
    
    scale = 0.5  # Convert cm to pixel size of top_array
    
    # xc, yc, w, h, d
    object_pose_cm = ut.bbox_to_cm_report(obj[1:], rgb_ints, depth_frame)
    
    obj_center_x = object_pose_cm[0] + object_pose_cm[2]/2
    
    # calculate height in top-view image
    h_top = np.sqrt(np.square(object_pose_cm[4]) - np.square(obj_center_x))
    
    center = (int(scale*obj_center_x), int(scale*(570 - h_top)))
    
    # Just filter objects inside the 6*4 meter boundary
    if not object_pose_cm[0] > 399 and not h_top > 569:
        cv2.circle(top_array, center, 8, 255, 1)
        cv2.putText(top_array, str(int(obj[0])), (center[0] - 5,
                                             center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.3, 255, 1)
    
    return top_array


def mapping_cam_related(detections, top_array, depth_frame, rgb_ints):
    
    """
    This function maps the report and add objects to a top-view map.
    The camera is stationary in the center-bottom and other objects move
    ** 300*200 represents 600 cm * 400 cm --> scale = 300/600
    
    
    Args:
        detections: A numpy array float32 of shape [N, length of F]
                    N is number of detections in a frame
                    F is features: label, score, x, y, w, and h
                    ** x and y are wrt the frame top-left
                    ** If depends on size, provide in original format
        top_array: A 300*200 numpy array from previous step
        depth_frame is distance frame in meter
                
    Returns:
        A (300,200) numpy array with only gray channel.
        This numpy array is top-view anc can be visualized with cv2
    """

    if not np.shape(detections)[0] == 0:
        dets = np.delete(detections, 1, axis=1)
        top_array[:285, :] = 0
        for obj in dets:
            top_array = add_obj_map(top_array, obj, depth_frame, rgb_ints)
    

    return top_array




def print_report(detections, depth_frame, rgb_ints, class_names):
    """
    This function prints the detections for each frame
    
    
    Args:
        detections: A numpy array float32 of shape [N, length of F]
                    N is number of detections in a frame
                    F is features: label, score, x, y, w, and h
                    ** x and y are wrt the frame top-left
                    ** If depends on size, provide in original format
        depth_frame is distance frame in meter
        rgb_ints is the intrinsics of the RGB camera
        class_names is the list

    """
    if not np.shape(detections)[0] == 0:
        print("#########################################################")
        for dets in detections:
            object_pose_cm = ut.bbox_to_cm_report(dets[2:], rgb_ints, depth_frame)
            # Converting to the center of the object
            object_pose_cm[0] = object_pose_cm[0] + object_pose_cm[2]/2
            object_pose_cm[1] = object_pose_cm[1] - object_pose_cm[3]/2
            if class_names:
                print(f"{class_names[int(dets[0])]} in x: {int(object_pose_cm[0])}, y: {int(object_pose_cm[1])}, d: {int(object_pose_cm[4])} with score: {dets[1]:.2f}")
            else:
                print(f"Class ID: {int(dets[0])} in x: {int(object_pose_cm[0])}, y: {int(object_pose_cm[1])}, d: {int(object_pose_cm[4])} with score: {dets[1]:.2f}")
            
    else:
        print("No object detected in the frame")
    
    
def capture_annotation_depth(detections, rgb_frame, depth_frame, class_names):
    """
    This functions creates annotations in the array of RGB image.
    
    Arguments:
        detections: A numpy array float16 of shape [N, length of F]
                    N is number of detections in a frame
                    F is features: label, scale, x, y, w, and h
                    ** x and y are wrt the frame top-left
                    ** If depends on size, provide in original format
        rgb_frame: NHWC RGB captured frame in uint8 or 16 numpy array dtype
        depth_frame is distance frame in meter [480, 640]
        class_names is the list
        
        
    Returns:
        annot_array: an array of layout HWC with bboxes

    """
    annot_array = rgb_frame[0]
    del rgb_frame
    for dets in detections:
        
        dep = 100*depth_frame[int(dets[3] + dets[5]/2)
                                          ,int(dets[2] + dets[4]/2)]
        
        cv2.rectangle(annot_array, (int(dets[2]), int(dets[3]))
                      ,(int(dets[2] + dets[4]), int(dets[3] + dets[5]))
                      , (0, 0, 255), 1, cv2.LINE_4)
    
        if class_names:
            cv2.putText(annot_array
                        , f"{class_names[int(dets[0])]} d: {int(dep)} cm"
                        , (int(dets[2]), int(dets[3]+3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5
                        , (0, 0, 255), 1, cv2.LINE_4)
            
        else:
            cv2.putText(annot_array
                        , f"Class ID: {int(dets[0])} d: {int(dep)} cm"
                        , (int(dets[2]), int(dets[3]+3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5
                        , (0, 0, 255), 1, cv2.LINE_4)
    
    return annot_array
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
