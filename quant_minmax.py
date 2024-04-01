import timm
import timm.data as data
from quantization import OnnxStaticQuantization
import torch


val_dataset = timm.data.ImageDataset("dataset/COCO/coco2017/val2017")
val_loader = timm.data.create_loader(val_dataset, (1,3,800,1200), 1)
module = OnnxStaticQuantization()

# method=MinMax, calibration_number=1000
module.quantization(
    fp32_onnx_path="weight/mask_rcnn_simple_fp32.onnx",
    future_int8_onnx_path="weight/mask_rcnn_simple_fp32_MinMax.onnx",
    calib_method="MinMax",
    calibration_loader=val_loader,
    sample=1000   # calibration number
)

print("Quantization Completed...")
