#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The main inference pipeline for object detection (ROS-subsciber node)

# fixed inference app with:
        model: yolo11n (Best model)
        device: rgbd
        iou = 0.40
        conf = 0.45

# Publish:
        sensor_msgs/Image
        std_msgs/String
"""

import argparse 
from utils.compiled_creator import CompiledModel
from utils.post_process_yolo import yolo_out_process
from pathlib import Path
import cv2
from utils import utils_set as ut
from utils import heads_depth as hd
import openvino as ov
import time
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

latest_color = None
latest_depth = None
fx = None
fy = None
ppx = None
ppy = None
width = None
height = None

def color_callback(msg):
    global latest_color
    latest_color = msg
    
def depth_callback(msg):
    global latest_depth
    latest_depth = msg

def cinfo_callback(msg):
    global fx, fy, ppx, ppy, width, height
    fx = msg.K[0]
    fy = msg.K[4]
    ppx = msg.K[2]
    ppy = msg.K[5]
    width = msg.width
    height = msg.height

def main():
    args = argparse.Namespace()
    args.model = 'yolo11n'
    args.device = 'rgbd'
    args.iou = 0.40
    args.conf = 0.45
    args.filter_labels = None
    args.ROOT = Path.cwd().resolve()
    args.coco90 = [] # check background
    args.coco80 = [] # check backgrounds
    args.msg_layout = 'RGB'

    #time.sleep(0.3)
    rospy.loginfo("Start subscribing to Color and Depth stream ...")

    rospy.init_node('detection_node', anonymous=True)
    bridge = CvBridge()
    rospy.Subscriber('/device_0/sensor_1/Color_0/image/data', Image, color_callback)
    rospy.Subscriber('/device_0/sensor_0/Depth_0/image/data', Image, depth_callback)    
    rospy.Subscriber('/device_0/sensor_1/Color_0/info/camera_info', CameraInfo, cinfo_callback)
    # rospy.spin()

    rospy.loginfo("Subscribing has been started...")

    pub_img = rospy.Publisher('annotated_img', Image, queue_size=10)
    pub_map = rospy.Publisher('top_map', Image, queue_size=10)

    time.sleep(0.3)
    rospy.loginfo("Gathering sensor info ...")

    attempts = 0
    max_attempts = 10
    while fx is None and attempts < max_attempts:
        rospy.sleep(0.1)
        rospy.loginfo("No camera info received")
        attempts += 1

    if fx is None:
        rospy.logwarn("Camera info not received after 10 attempts. Using predefined values.")
        args.rgb_ints = {
            'fx': 617.323486328125,
            'fy': 617.6768798828125,
            'ppx': 330.5740051269531,
            'ppy': 235.93508911132812,
            }
        args.h = 480
        args.w = 640

    else:
        args.rgb_ints = {
            'fx': fx,
            'fy': fy,
            'ppx': ppx,
            'ppy': ppy,
            }
        
        args.h = height
        args.w = width
        
    args.depth_scale = 0.001

    time.sleep(0.1)
    rospy.loginfo("Compiling the model ...")

    init_model = CompiledModel(args.model, args.ROOT, args.msg_layout)
    model, input_layer = init_model.compile_it()
    
    a_infer = model.create_infer_request()
    b_infer = model.create_infer_request()

    a_infer.share_inputs = True
    a_infer.share_outputs = True
    b_infer.share_inputs = True
    b_infer.share_outputs = True
        

    time.sleep(0.1)
    rospy.loginfo("Detection just started ...")

    # device already is rgbd
    shape = (args.h, args.w)
    color_frame = bridge.imgmsg_to_cv2(latest_color, "bgr8")
    depth_frame = bridge.imgmsg_to_cv2(latest_depth, "16UC1")
    frame, depth = ut.rgbd_preprocessing(color_frame, depth_frame, args.depth_scale, args.model)
        
    fps = 0
    a_infer.set_tensor(input_layer, ov.Tensor(frame))
    a_infer.start_async()
    ti = time.time()
    frame_counter = 0
    j = 0

    while not rospy.is_shutdown():
    
        # device is rgbd by default
        color_frame = bridge.imgmsg_to_cv2(latest_color, "bgr8")
        depth_frame = bridge.imgmsg_to_cv2(latest_depth, "16UC1")
        frame_next, depth_next = ut.rgbd_preprocessing(color_frame, depth_frame, args.depth_scale, args.model)
            
        b_infer.set_tensor(input_layer, ov.Tensor(frame_next))
        b_infer.start_async()
        a_infer.wait()
        res = a_infer.get_output_tensor
        
        total_time = time.time() - ti
        frame_counter = frame_counter + 1
        fps = frame_counter / total_time #Absolute, not average
        j = j + 1
        if j % 10 == 0:
            j = 0
            rospy.loginfo(f"Average FPS is: {fps:.2f}")
        
        # model: yolo11n
        res = yolo_out_process(res, shape, args.iou, args.conf, args.filter_labels)
           
        # yolo capture annotation
        frame_vis = hd.capture_annotation_depth(res, (frame_next*255.0).astype(np.uint8), depth_next, args.coco80)
        out_img = bridge.cv2_to_imgmsg(frame_vis, encoding="bgr8")

        if frame_counter == 1:
            top_map = ut.init_array_topview((600, 400))
        top_map = hd.mapping_cam_related(res, top_map, depth_next, args.rgb_ints)
        out_map = bridge.cv2_to_imgmsg(top_map, encoding="mono8")

        hd.print_report(res, depth_next, args.rgb_ints, args.coco80)

        # publish annotated image
        pub_img.publish(out_img)
        pub_map.publish(out_map)
            
        frame = frame_next
        a_infer, b_infer = b_infer, a_infer

    rospy.spin()

if __name__ == "__main__":
    rospy.loginfo("Start a new detection session: YOLO11n, with RealSense D435...")
    main()