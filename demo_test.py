
import torch  # type: ignore
import pytorch_mask_rcnn as pmr  # type: ignore  


use_cuda = True
dataset = "coco"
# ckpt_path = "../ckpts/maskrcnn_voc-5.pth"
data_dir = "dataset/COCO/coco2017/"

device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    pmr.get_gpu_prop(show=True)
print(f"\ndevice: {device}")

ds = pmr.datasets(dataset, data_dir, "val2017", train=True)

d = torch.utils.data.DataLoader(ds, shuffle=False)

model = pmr.maskrcnn_resnet50(True, max(ds.classes) + 1).to(device)
model.eval()
model.head.score_thresh = 0.3

for p in model.parameters():
    p.requires_grad_(False)


image, target = next(iter(d))

image = image.to(device)[0]
# print(target)
#target = {k: v.to(device) for k, v in target.items()}


result = model(image)
print(result)





