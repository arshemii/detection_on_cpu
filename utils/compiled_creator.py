
"""
1. Prepare model input layer for input preprocessing
2. Prepare compiled model and input_layer structure
"""
from pathlib import Path
import openvino as ov
import openvino.properties.hint as hints
from openvino.preprocess import ResizeAlgorithm
import openvino.properties.intel_cpu as intel_cpu

class CompiledModel():
    
    def __init__(self, model_name, root_dir, msg_layout):
        self.model_path_rel = Path(f"/models/{model_name}_quantized_openvino_model/{model_name}_quantized.xml")
        self.model_path = Path(f"{root_dir}/{self.model_path_rel}")
        self.core = ov.Core()
        self.DEVICE = "CPU"
        self.CONFIG = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
        self.model_name = model_name
        self.msg_layout = msg_layout
        # already know model is yolo11n
        self.channels = 'RGB'
        self.layout = "NCHW"

        
        
    def add_ppp(self):
        # By default, stream is BGR (OpenCV is BGR, RealSense is set to BGR8)
        ppp_model = self.core.read_model(self.model_path)
        ppp_model = ov.preprocess.PrePostProcessor(ppp_model)
        ppp_model.input().model().set_layout(ov.Layout(self.layout))
        
                
        # already know model is yolo11n
        ppp_model.input().tensor() \
            .set_element_type(ov.Type.f32) \
                .set_spatial_dynamic_shape() \
                    .set_layout(ov.Layout("NHWC"))
            
        if self.msg_layout == 'BGR':
            ppp_model.input().preprocess() \
                .resize(ResizeAlgorithm.RESIZE_NEAREST) \
                        .reverse_channels()
        else:
            ppp_model.input().preprocess() \
                .resize(ResizeAlgorithm.RESIZE_NEAREST)
                        
                    
        model = ppp_model.build()
        del ppp_model

        return model
    
    def compile_it(self):
        model = self.add_ppp()
        
        self.core.set_property("CPU", intel_cpu.denormals_optimization(True))
        self.core.set_property("CPU", {hints.execution_mode: hints.ExecutionMode.PERFORMANCE})
        
        compiled_model = self.core.compile_model(model, device_name=self.DEVICE, config=self.CONFIG)
        
        input_layer = compiled_model.input(0)
        
        return compiled_model, input_layer