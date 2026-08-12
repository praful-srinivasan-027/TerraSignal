import onnxruntime as onx
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
import cv2
import numpy as np 

ONNX_WEIGHTS_PATH = "weights/plant_leaf_detection.onnx"

session = onx.InferenceSession(ONNX_WEIGHTS_PATH)

input_info = session.get_inputs()[0]

print("Input Size: ", input_info.shape)
print("Input Name: ", input_info.name)
print("Input type: ", input_info.type)

class YOLOCalibrationDatasetReader(CalibrationDataReader):
    def __init__(self, images):
        self.img_paths = images
        self.index = 0
    
    def get_next(self):
        if self.index>=len(self.img_paths):
            return None
        
        image_path = self.img_paths[self.index]
        image = cv2.imread(image_path, -1)
        image = cv2.resize(image, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32')/255.0
        image = image.transpose(2, 0, 1)
        image = image[np.newaxis,:,:,:]
        self.index+=1

        return {
            "images": image
        }

    def rewind(self):
        self.index=0

hi = YOLOCalibrationDatasetReader(["sample_images/leaf_sample.jpg"])   
print(hi.get_next()["images"].shape) 

quantize_static(
    model_input=ONNX_WEIGHTS_PATH,
    model_output="weights/plant_leaf_detection_int8.onnx",
    calibration_data_reader=hi,
    weight_type=QuantType.QInt8
)