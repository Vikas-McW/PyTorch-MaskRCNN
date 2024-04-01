
import torch
import onnxruntime
import pytorch_mask_rcnn as pmr
import time
# from .datasets import CocoEvaluator, prepare_for_coco
import re
import sys

import onnxruntime as ort # Import the ONNX Runtime
import torch.nn.functional as F


use_cuda = True
dataset = "coco"

data_dir = "dataset/COCO/coco2017/"

###################################### < COCO EVALUATOR > ######################################

import copy
import torch
import numpy as np
import torch.onnx

import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

class CocoEvaluator:
    def __init__(self, coco_gt, iou_types="bbox"):
        if isinstance(iou_types, str):
            iou_types = [iou_types]
        
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        #self.ann_labels = ann_labels
        self.coco_eval = {iou_type: COCOeval(coco_gt, iouType=iou_type) for iou_type in iou_types}
        self.has_results = False
    
    def accumulate(self, coco_results): # input all predictions
        # sourcery skip: collection-builtin-to-comprehension, comprehension-to-generator
        if len(coco_results) == 0:
            return

        image_ids = list(set([res["image_id"] for res in coco_results]))
        for iou_type in self.iou_types:
            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = self.coco_gt.loadRes(coco_results) # use the method loadRes
            coco_eval.params.imgIds = image_ids # ids of images to be evaluated
            coco_eval.evaluate() # 15.4s
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

            coco_eval.accumulate() # 3s
            
        self.has_results = True
    
    def summarize(self):
        if self.has_results:
            for iou_type in self.iou_types:
                print(f"IoU metric: {iou_type}")
                self.coco_eval[iou_type].summarize()
        else:
            print("evaluation has no results")
            

def prepare_for_coco(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]
        masks = prediction["masks"]

        x1, y1, x2, y2 = boxes.unbind(1)
        boxes = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=1)
        boxes = boxes.tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        masks = masks > 0.5
        rles = [
            mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[i],
                    "bbox": boxes[i],
                    "segmentation": rle,
                    "score": scores[i],
                }
                for i, rle in enumerate(rles)
            ]
        )
    return coco_results    


################################################################################################


device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    pmr.get_gpu_prop(show=True)
print(f"\ndevice: {device}")

class TextArea:
    def __init__(self):
        self.buffer = []

    def write(self, s):
        self.buffer.append(s)

    def __str__(self):
        return "".join(self.buffer)

    def get_AP(self):
        result = {"bbox AP": 0.0, "mask AP": 0.0}
        
        txt = str(self)
        values = re.findall(r"(\d{3})\n", txt)
        if len(values) > 0:
            values = [int(v) / 10 for v in values]
            result = {"bbox AP": values[0], "mask AP": values[12]}
        
        return result

class Meter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:sum={sum:.2f}, avg={avg:.4f}, count={count}"
        return fmtstr.format(**self.__dict__)
    

# Load the dataset
ds = pmr.datasets(dataset, data_dir, "val2017", train=True)
d = torch.utils.data.DataLoader(ds, shuffle=False)

# model = pmr.maskrcnn_resnet50(True, max(ds.classes) + 1).to(device)
# model = torch.onnx.load("weight/mask_rcnn_fp32.onnx")

# Load the model and create an InferenceSession
model = ort.InferenceSession("weight/mask_rcnn_simple_fp32.onnx", providers=['CPUExecutionProvider'])

t_m = Meter("total")
m_m = Meter("model")
coco_results = []

A = time.time()

# ==============================================================================================

for i, (image, target) in enumerate(d): 
    T = time.time()
    
    image = image.to(device)
    size = [800, 1200]
    image = F.interpolate(image[-2:], size=size, mode='bilinear', align_corners=False)[0]
    
    target = {k: v.to(device) for k, v in target.items()}

    S = time.time()
    # torch.cuda.synchronize()
    output = model.run(None, {'image.1': image.cpu().numpy()})
    print(output)   
    m_m.update(time.time() - S)

    prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
    coco_results.extend(prepare_for_coco(prediction))

    t_m.update(time.time() - T)

    # --------------- added script -----------------
    if (i >= 10):
        break
    # ----------------------------------------------

# ==============================================================================================

A = time.time() - A 

# return A / iters, coco_results
dataset = ds #
iou_types = ["bbox", "segm"]
coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

# results = torch.load(args.results, map_location="cpu")

S = time.time()
coco_evaluator.accumulate(coco_results)
print("Accumulate: {:.1f}s".format(time.time() - S))

# collect outputs of build in function print
temp = sys.stdout
sys.stdout = TextArea()

coco_evaluator.summarize()
output = sys.stdout
sys.stdout = temp

print(output.get_AP())


