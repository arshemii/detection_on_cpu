<div align="center">
    <img src="images/Header.png" alt="Example Image" width="800">
</div>

This repository is the ROS branch of [CPU-based Object Detection Inference](https://github.com/arshemii/detection_on_cpu).



## Specifications:
- &nbsp;&nbsp; Subscribes to the depth and color frame of RGB-D camera (Intel RealSense D435)
- &nbsp;&nbsp; Data sctructure and types are Numpy f32 since the test hardware has Intel® SSE4.2
- &nbsp;&nbsp; Input size flexibility due to dynamic size input layer for all models
- &nbsp;&nbsp; Use OpenVino runtime for inference (Asynchronous inference)
- &nbsp;&nbsp; Publishes captured stream with dynamic frequency
- &nbsp;&nbsp; Publishes top-view map of detected object related to the camera frame
- &nbsp;&nbsp; Logs the X, Y, and Z of the class ID object

### For more information about the actual model, please refer to the master branch:
[CPU-based Object Detection Inference](https://github.com/arshemii/detection_on_cpu)

----------------------------------------------------------------
## How to use:
&nbsp;&nbsp; 1. you should have installed ROS noetic, OpenCV, OpenVino runtime, and numpy. <br>

&nbsp;&nbsp; 2. Create a catkin package called detection, then go the detection <br>
&nbsp;&nbsp; 3. clone this repository (Branch: ROS) <br>
&nbsp;&nbsp; 4. Add permissions of all .py files: (Branch: ROS) <br>
```markdown
chmod +x script_name.py
```
&nbsp;&nbsp; 5. Run inside the terminal:
```markdown
settask -c 0,1 detection inference_main.py

# For settask, select the number of available cores

```

## References:
- https://docs.openvino.ai/2024/index.html
- https://docs.ultralytics.com
- https://github.com/google/automl/tree/master/efficientdet
- https://github.com/xingyizhou/CenterNet
- https://github.com/facebookresearch/detr

## Contact:
- [&nbsp;Email](arshemii1373@gmail.com)
