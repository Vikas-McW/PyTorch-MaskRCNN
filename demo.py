import torch
import pytorch_mask_rcnn as pmr


use_cuda = True
dataset = "coco"
ckpt_path = "weight\\resnet50-0676ba61.pth"
data_dir = "dataset/COCO/coco2017/"

device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    pmr.get_gpu_prop(show=True)
print("\ndevice: {}".format(device))

ds = pmr.datasets(dataset, data_dir, "val2017", train=False)
#indices = torch.randperm(len(ds)).tolist()
#d = torch.utils.data.Subset(ds, indices)
# data_loader = torch.utils.data.DataLoader(ds, shuffle=False)

from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
# load model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

for p in model.parameters():
    p.requires_grad_(False)
    
iters = 5

for i, (image, target) in enumerate(ds):
    image = image.to(device)[0]
    #target = {k: v.to(device) for k, v in target.items()}
    
    with torch.no_grad():
        result = model(image)
    
    pmr.show(image, result, ds.classes, "./image/output{}.jpg".format(i))

    if i >= iters - 1:
        break










