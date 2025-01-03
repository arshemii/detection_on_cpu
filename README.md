<div align="center">
    <img src="images/Header.png" alt="Example Image" width="800">
</div>

This repository is the second part of **Optimization of Object Detection on LattePanda 3 Delta**. The first part can be found [here](https://github.com/arshemii/detection_quantization).
<div align="center">
    <img src="images/repex.png" alt="Example Image" width="800">
</div>

This repository presents a complete, flexible, and ready-to-use application to infere object detection models on CPU-based edge devices. The inference, preprocessing, and postprocessing is optimized and tested on LattePanda 3 Delta. In details, 6 different Object Detection models (**YOLOv5nu, YOLOv8n, YOLO11n, CenterNet, DETR, and EfficientDet**) are selected, quantized, and validated in [this repository](https://github.com/arshemii/detection_quantization). You can inspect more about these models in:

- [&nbsp;YOLOv5 Documentation](https://docs.ultralytics.com/yolov5/)
- [&nbsp;YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)
- [&nbsp;YOLOv11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [&nbsp;CenterNet Paper](https://arxiv.org/abs/2005.12872)
- [&nbsp;DETR Paper](https://arxiv.org/abs/1904.07850)
- [&nbsp;EfficientDet GitHub Repository](https://github.com/google/automl/tree/master/efficientdet)


## Specifications:
- &nbsp;&nbsp; Capture frames from Webcam and RGB-D device (Intel RealSense D435)
- &nbsp;&nbsp; Data sctructure and types are Numpy f32 since the test hardware has Intel® SSE4.2
- &nbsp;&nbsp; Input size flexibility due to dynamic size input layer for all models
- &nbsp;&nbsp; Use OpenVino runtime for inference (Asynchronous inference)
- &nbsp;&nbsp; 3 different heads for RGB-D: Textual Report, Stream Visualization, and a Top-view B&W Map
- &nbsp;&nbsp; 2 different heads for RGB: Textual Report and Stream Visualization

----------------------------------------------------------------
## How to use:
&nbsp;&nbsp; 1. you should have installed OpenCV, OpenVino runtime, pyrealsense2, scipy, and numpy. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; It is better to install pyrealsense2 on ubunto not newer than 22.04 <br>
&nbsp;&nbsp; 1.1. You can also pull the docker image from using:<br>
'''markdown
docker pull arshemii/drone_od:26nov24 <br>
&nbsp;&nbsp; 2. Clone this repository and cd detection_on_cpu <br>
&nbsp;&nbsp; 3. Open a terminal inside the cloned repository or run: <br>
```markdown
cd path/to/cloned/repo
