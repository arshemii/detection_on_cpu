#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The main inference pipeline for object detection 
"""

import argparse 
from utils.compiled_creator import CompiledModel
from utils import heads_depth as hd
from utils import heads_no_depth as hnd
from utils.post_process_ctdet import ctdet_out_process
from utils.post_process_detr import detr_out_process
from utils.post_process_effdet import effdet_out_process
from utils.post_process_yolo import yolo_out_process
from pathlib import Path
import cv2
from utils import utils_set as ut
import openvino as ov
import time
import numpy as np




def print_sts(total_time
              ,capture_total_time,
              cap_pp_total_time,
              res_pp_total_time,
              abs_inf_total_time,
              head_tot_time,
              args):
    
    total_functions = capture_total_time +\
        cap_pp_total_time +\
            res_pp_total_time +\
                abs_inf_total_time +\
                    head_tot_time
    
    lag_time = total_time - total_functions
    
    if lag_time >= 0:
        capture_total_time = 100*(capture_total_time/total_time)
        cap_pp_total_time = 100*(cap_pp_total_time/total_time)
        res_pp_total_time = 100*(res_pp_total_time/total_time)
        abs_inf_total_time = 100*(abs_inf_total_time/total_time)
        head_tot_time = 100*(head_tot_time/total_time)
        lag_time = 100*(lag_time/total_time)
    else:
        capture_total_time = 100*(capture_total_time/total_functions)
        cap_pp_total_time = 100*(cap_pp_total_time/total_functions)
        res_pp_total_time = 100*(res_pp_total_time/total_functions)
        abs_inf_total_time = 100*(abs_inf_total_time/total_functions)
        head_tot_time = 100*(head_tot_time/total_functions)
    
    print(f"Model: {args.model}, Device: {args.device}, and Head: {args.head}")
    print("--------------------------------------------------------------------------------------------")
    print(f"Capturing a frame occupies {int(capture_total_time)} percent of each iteration")
    print(f"Preprocessing a frame occupies {int(cap_pp_total_time)} percent of each iteration")
    print(f"Inference with a frame occupies {int(abs_inf_total_time)} percent of each iteration")
    print(f"Post Processing a result occupies {int(res_pp_total_time)} percent of each iteration")
    print(f"Head of results occupies {int(head_tot_time)} percent of each iteration")
    if lag_time >= 0:
        print(f"Lag time occupies {int(lag_time)} percent of each iteration")
    
    






def main(args):
    # Uncomment for test purposes
    # args = argparse.Namespace()
    # args.model = 'yolov8n'
    # args.device = 'rgbd'
    args.iou = 0.45
    args.conf = 0.40
    # args.head = 'map'
    args.ROOT = Path.cwd().resolve()
    args.coco90 = [] # check background
    args.coco80 = [] # check backgrounds
    args.filter_labels = None # example: [1.0, 4.0, ... , 16.0]
    args.h = 480
    args.w = 640
    args.fps = 6
    args.k = 50 # for ctdet
    args.nms = True # for effdet
    args.rgb_ints = {
        'fx': 617.323486328125,
        'fy': 617.6768798828125,
        'ppx': 330.5740051269531,
        'ppy': 235.93508911132812,
        } 
    
    
    
    init_model = CompiledModel(args.model, args.ROOT)
    model, input_layer = init_model.compile_it()
    
    a_infer = model.create_infer_request()
    b_infer = model.create_infer_request()

    a_infer.share_inputs = True
    a_infer.share_outputs = True
    b_infer.share_inputs = True
    b_infer.share_outputs = True
    
    if args.device == 'webcam':
        cap = cv2.VideoCapture(0)
        while True:
            is_frame, frame = cap.read(0)
            if is_frame:
                shape = (frame.shape[0], frame.shape[1])
                break
        frame = ut.frame_preprocessing(frame, args.model)
        
    if args.device == 'rgbd':
        shape = (args.h, args.w)
        pipe, align, depth_scale = ut.rgbd_init(args.h, args.w, args.fps)
        frameset = pipe.wait_for_frames()
        aligned_frames = align.process(frameset)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        frame, depth = ut.rgbd_preprocessing(color_frame, depth_frame, depth_scale, args.model)
        

    a_infer.set_tensor(input_layer, ov.Tensor(frame))
    a_infer.start_async()
    
    ti = time.time()  # For all functionalities
    capture_total_time = 0 # total time of capturing frames
    cap_pp_total_time = 0 # total time frame preprocessing
    res_pp_total_time = 0 # total time of results post processing
    abs_inf_total_time = 0 # Total time of network
    head_tot_time = 0 # total time of head processing
    
    
    frame_counter = 0
    while True:
        if args.device == 'webcam':
            t_cap = time.time()
            is_frame, frame_next = cap.read()
            capture_total_time = capture_total_time + time.time() - t_cap
            if not is_frame:
                continue
            t_cap_pp = time.time()
            frame_next = ut.frame_preprocessing(frame_next, args.model)
            cap_pp_total_time = cap_pp_total_time + time.time() - t_cap_pp
            
        if args.device == 'rgbd':
            t_cap = time.time()
            frameset = pipe.wait_for_frames()
            aligned_frames = align.process(frameset)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            capture_total_time = capture_total_time + time.time() - t_cap
            t_cap_pp = time.time()
            frame_next, depth_next = ut.rgbd_preprocessing(color_frame, depth_frame, depth_scale, args.model)
            cap_pp_total_time = cap_pp_total_time + time.time() - t_cap_pp
            
        t_inf = time.time()
        b_infer.set_tensor(input_layer, ov.Tensor(frame_next))
        b_infer.start_async()
        a_infer.wait()
        res = a_infer.get_output_tensor
        abs_inf_total_time = abs_inf_total_time + time.time() - t_inf
        
        total_time = time.time() - ti
        frame_counter = frame_counter + 1
        print(f"{int(100*frame_counter/1000)} percent of the test is passes")
        if frame_counter == 1000:
            break
        
        if args.model == 'yolov5nu' or args.model == 'yolov8n' or args.model == 'yolo11n':
            t_res = time.time()
            res = yolo_out_process(res, shape, args.iou, args.conf, args.filter_labels)
            res_pp_total_time = res_pp_total_time + time.time() - t_res
        elif args.model == 'detr':
            t_res = time.time()
            res = detr_out_process(res, shape, args.iou, args.conf, args.filter_labels)
            res_pp_total_time = res_pp_total_time + time.time() - t_res
        elif args.model == 'ctdet':
            t_res = time.time()
            res = ctdet_out_process(res, shape, args.conf, args.k, args.filter_labels)
            res_pp_total_time = res_pp_total_time + time.time() - t_res
        else:
            t_res = time.time()
            res = effdet_out_process(res, shape, args.iou, args.conf, args.nms, args.filter_labels)
            res_pp_total_time = res_pp_total_time + time.time() - t_res
            
            
        if args.device == 'rgbd':
            if args.head == 'vis':
                t_head = time.time()
                if 'yolo' in args.model:
                    frame_vis = hd.capture_annotation_depth(res, (frame_next*255.0).astype(np.uint8), depth_next, args.coco80)
                else:
                    frame_vis = hd.capture_annotation_depth(res, frame_next.astype(np.uint8), depth_next, args.coco80)
                cv2.imshow("Detection on RGB-D", frame_vis)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                head_tot_time = head_tot_time + time.time() - t_head
                
            elif args.head == 'map':
                t_head = time.time()
                if frame_counter == 1:
                    top_map = ut.init_array_topview((600, 400))
                top_map = hd.mapping_cam_related(res, top_map, depth_next, args.rgb_ints)
                cv2.imshow("Detection on map", top_map)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                head_tot_time = head_tot_time + time.time() - t_head
                
            else:
                t_head = time.time()
                hd.print_report(res, depth_next, args.rgb_ints, args.coco80)
                head_tot_time = head_tot_time + time.time() - t_head
               
                
        else:
            if args.head == 'vis':
                t_head = time.time()
                if 'yolo' in args.model:
                    frame_vis = hnd.capture_annotation_no_depth(res, (frame_next*255.0).astype(np.uint8), args.coco80)
                else:
                    frame_vis = hnd.capture_annotation_no_depth(res, frame_next.astype(np.uint8), args.coco80)
                cv2.imshow("Detection on webcam", frame_vis)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                head_tot_time = head_tot_time + time.time() - t_head
                
            else:
                t_head = time.time()
                hnd.print_report(res, args.coco80, shape)
                head_tot_time = head_tot_time + time.time() - t_head
            

        frame = frame_next
        a_infer, b_infer = b_infer, a_infer

    ###########################
    print_sts(total_time
                  ,capture_total_time,
                  cap_pp_total_time,
                  res_pp_total_time,
                  abs_inf_total_time,
                  head_tot_time,
                  args)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for Detection inference. Use flags:")
    
    
    parser.add_argument(
        '--model', 
        choices=['yolov5nu', 'yolov8n', 'yolo11n', 'ctdet', 'detr', 'effdet'],
        required=True,
        help="Network to be used for object detection"
    )
    
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['webcam', 'rgbd'],
        required=True,
        help="Device for capturing"
    )
    
    parser.add_argument(
        '--head', 
        type=str,
        choices=['text', 'vis', 'map'],
        required=True,
        help="Detection results Head"
    )
    
    args = parser.parse_args()
    
    if args.device == 'webcam':
        if args.head == 'map':
            raise ValueError("Mapping is only accepted for RGBD inputs!")
    
    main(args)