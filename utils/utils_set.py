
"""
utils for inference
"""
import numpy as np
from scipy.ndimage import maximum_filter
import pyrealsense2 as rs


def bbox_to_cm_report(one_object_bbox, rgb_ints, scaled_depth_frame):
    """
    Arguments:
        one_object_bbox in format: xmin, ymin, w, h from top-left, and label
        rgb_ints intrinsics of color frame
        scaled_depth_frame is distance in meter
    
    Return:
        A bbox in cm in an array of 5 (x, w, h, and distance)
               with respect to the center of the image
    """
    
    fx = rgb_ints['fx']
    fy = rgb_ints['fy']
    ppx = rgb_ints['ppx']
    ppy = rgb_ints['ppy']
    
    cm_report = np.zeros((5, 1), dtype = np.float32)
    
    
    # Depth
    cm_report[4] = 100*scaled_depth_frame[int(one_object_bbox[1] + one_object_bbox[3]/2)
                                      ,int(one_object_bbox[0] + one_object_bbox[2]/2)]
    
    
    # xmin, ymin, w, and h
    cm_report[0] = (one_object_bbox[0] - ppx) * cm_report[4] / fx
    cm_report[1] = (-one_object_bbox[1] + ppy) * cm_report[4] / fy
    cm_report[2] = (one_object_bbox[2]) * cm_report[4] / fx
    cm_report[3] = (one_object_bbox[3]) * cm_report[4] / fy
    
    
    
    return cm_report

def init_array_topview(shape):
    
    image = np.zeros(shape, dtype = np.uint8)
    
    height = shape[0]
    width = shape[1]
    
    icon_height = 12
    icon_width = 15
    icon_top = height - icon_height
    icon_left = (width - icon_width) // 2

    # Draw the camera icon
    # Camera base (rectangle)
    image[icon_top:height, icon_left:icon_left + icon_width] = 255

    # Camera lens (circle)
    lens_radius = 4
    lens_center = (icon_top + icon_height // 2, icon_left + icon_width // 2)
    for y in range(icon_top, height):
        for x in range(icon_left, icon_left + icon_width):
            if (y - lens_center[0]) ** 2 + (x - lens_center[1]) ** 2 <= lens_radius ** 2:
                image[y, x] = 0
    
    # Camera flash (small rectangle on top of the base)
    flash_height = 3
    flash_width = 5
    flash_top = icon_top - flash_height
    flash_left = lens_center[1] - flash_width // 2
    image[flash_top:icon_top, flash_left:flash_left + flash_width] = 255
    
    return image




def topleft_to_center(detections, shape):
    """
    This function converts [0, width] and [o, height] to:
        [-width/2, +width/2] and [-height/2, +height/2]
    
    
    Args:
        detections: A numpy array float16 of shape [length of F]
                    F is features: score, label, xmin, ymin, w, and h
                    ** x and y are wrt the frame center
    Returns:
        detections: A numpy array float16 of shape [N, length of F]
                    N is number of detections in a frame
                    F is features: score, label, xcenter, ycenter, w, and h
    """
    detections[2] = detections[2] - shape[1]/2
    detections[3] = detections[3] - shape[0]/2
    
    return detections

    
def softmax(array, axis):
    # Performs the softmax on a numpy array
    array = np.exp(array - np.max(array, axis=axis, keepdims=True))
    array /= array.sum(axis=axis, keepdims=True)
    
    return array
    

def sigmoid(x):
    # Sigmoid for a numpy array or a scalar
    return 1 / (1 + np.exp(-x))

    
def center_wh_to_xywh(boxes):
    # Convert cx_cy_w_h format --> xmin_ymin_w_h
    boxes[:, 0] = boxes[:, 0] - 0.5*boxes[:, 2]
    boxes[:, 1] = boxes[:, 1] - 0.5*boxes[:, 3]
    return boxes

def xyxy_wh_to_xywh(boxes):
    # Convert xmin, ymin, xmax, ymax format --> xmin_ymin_w_h
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    return boxes
    
    
def bbox_scale(boxes, shape):
    boxes[:, [0, 2]] *= shape[1]
    boxes[:, [1, 3]] *= shape[0]
    return boxes
    
    

def nms_loacl_maxima(heat, kernel=3):
    # Numpy version of:
        # https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/decode.py
    #pad = (kernel - 1) // 2
    hmax = maximum_filter(heat, size=(1, kernel, kernel), mode='constant')
    keep = (heat == hmax)
    return heat * keep
    
    
def non_max_suppression_fast(boxes, iouth):
    
    """
    Source: https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    Modified to incorporate scores
    
    Arguments:
        boxes: numpy array [N, 6] in labels, score, xmin, ymin, w, h
        conf: Confidence threshold, minimum acceptable score
    Return:
        boxes: filtered bboxes
    """
    
    areas = boxes[:, 4] * boxes[:, 5]
    score_rank = np.argsort(boxes[:, 1])[::-1]
    
    keep_ind = []
    
    while len(score_rank) > 0:
        idx = score_rank[0]
        keep_ind.append(idx)
        
        rest_idx = score_rank[1:]
        if len(rest_idx) == 0:
            break
        
        xx1 = np.maximum(boxes[idx, 2], boxes[rest_idx, 2])
        yy1 = np.maximum(boxes[idx, 3], boxes[rest_idx, 3])
        xx2 = np.minimum(boxes[idx, 2] + boxes[idx, 4], 
                         boxes[rest_idx, 2] + boxes[rest_idx, 4]
                         )
        yy2 = np.minimum(boxes[idx, 3] + boxes[idx, 5], 
                         boxes[rest_idx, 3] + boxes[rest_idx, 5]
                         )
        
        
        w_int = np.maximum(0, xx2 - xx1)
        h_int = np.maximum(0, yy2 - yy1)
        
        area_int = w_int * h_int
        iou = area_int / (areas[idx] + areas[rest_idx] - area_int)
        
        # Keep boxes with IoU below the threshold
        mask = iou <= iouth
        score_rank = rest_idx[mask]
    
    return boxes[keep_ind]
    
    
    
    
def yolo_84_to_6_features(res, conf):
    # Converts 84 box features of yolo to 6 features
    # Confidence filter is applied
    boxes = res[:4 , :]
    scores = np.max(res[4: , :], axis = 0, keepdims=True)
    labels = np.argmax(res[4: , :], axis = 0, keepdims=True)   
    
    res = np.vstack((labels, scores, boxes), dtype = np.float32).T
    res = res[res[:, 1] > conf]
    
    return res
    
    
def frame_preprocessing(frame, model_name):
    frame = np.flip(frame, axis = 1)
    frame = np.expand_dims(frame, axis = 0)
    if "yolo" in model_name:
        frame = (frame/255.0).astype(dtype = np.float32)
        
    return frame


def rgbd_preprocessing(color, depth, depth_scale, model_name):
    color = np.asanyarray(color.get_data(), dtype = np.uint8)
    depth = np.asanyarray(depth.get_data(), dtype = np.float32)
    depth = depth * depth_scale
    if "yolo" in model_name:
        color = color.astype(np.float32) / 255.0
    color = np.expand_dims(color, axis=0)
    return color, depth

def rgbd_init(h, w, fps):
    """
    This function initialize the stream of realsense camera
    """

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)  # RGB stream
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)  # Depth stream
        
    # Start the pipeline
    profile = pipe.start(cfg)
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    
    # Create an align object
    align = rs.align(rs.stream.color)  # Align depth to color
        
    return pipe, align, depth_scale
