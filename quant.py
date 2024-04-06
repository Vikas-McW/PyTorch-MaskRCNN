
import torch
import utils
import pytorch_mask_rcnn as pmr

"""### **Method=MinMax**"""

# Quantization Class ---------------------------------------------------------------------
import onnxruntime as ort
# import timm


use_cuda = True
dataset = "coco"
ckpt_path = "weight/mask_rcnn_pytorch.pth"
data_dir = "dataset/COCO/coco2017/"

class OnnxStaticQuantization:
    def __init__(self) -> None:
        self.enum_data = None
        self.calibration_technique = {
            "MinMax": ort.quantization.calibrate.CalibrationMethod.MinMax,
            "Entropy": ort.quantization.calibrate.CalibrationMethod.Entropy,
            "Percentile": ort.quantization.calibrate.CalibrationMethod.Percentile,
            "Distribution": ort.quantization.calibrate.CalibrationMethod.Distribution
        }

    def get_next(self, EP_list = ['CPUExecutionProvider']):  # EP_list=['CUDAExecutionProvider']    EP_list = ['CPUExecutionProvider']
        if self.enum_data is None:
            session = ort.InferenceSession(self.fp32_onnx_path, providers=EP_list)
            input_name = session.get_inputs()[0].name
            calib_list = []
            count = 0
            for nhwc_data, _ in self.calibration_loader:
                nhwc_data=nhwc_data.cpu()
                calib_list.append({input_name: nhwc_data.numpy()})
                if self.sample == count: break
                count = count + 1
            self.enum_data = iter(calib_list)
        return next(self.enum_data, None)

    def quantization1(self, fp32_onnx_path, future_int8_onnx_path, calib_method, calibration_loader, sample=100):
        self.sample = sample
        self.calibration_loader = calibration_loader
        _ = ort.quantization.quantize_static(
                model_input=fp32_onnx_path,
                model_output=future_int8_onnx_path,
                activation_type=ort.quantization.QuantType.QInt16,
                weight_type=ort.quantization.QuantType.QInt8,
                calibrate_method=self.calibration_technique[calib_method],
                per_channel=True,
                reduce_range=True,
                calibration_data_reader=self
            )
        return self


# --------------------------------------------------------------------------------------
# method=MinMax, calibration_number=1000
# Perform the quantization
val_dataset = pmr.datasets(dataset, data_dir, "val2017", train=False)
val_loader = torch.utils.data.Subset(dataset, 1)
# val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False)
module = OnnxStaticQuantization()
module.fp32_onnx_path = "weight/mask_rcnn_simple_fp32.onnx"
module.quantization1(
    fp32_onnx_path="weight/mask_rcnn_simple_fp32.onnx",
    future_int8_onnx_path="weight/mask_rcnn_simple_fp32_MinMax.onnx",
    calib_method="MinMax",
    calibration_loader=val_loader,
    sample=1   # calibration number
)

print("Quantization Done...!")

