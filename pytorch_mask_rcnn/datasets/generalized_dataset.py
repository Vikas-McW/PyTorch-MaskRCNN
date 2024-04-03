import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torchvision.transforms.functional as tf

# #################################################################################################
import math
import torch.nn.functional as F
from torchvision import transforms

class Transformer:
    def __init__(self, min_size, max_size, image_mean, image_std):
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
    
    def __call__(self, image, target):
        image = self.normalize(image)
        image, target = self.resize(image, target)
        image = self.batched_image(image)
        return image, target
    
    def normalize(self, image):
        image=image.squeeze(0)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        dtype, device = image.dtype, image.device
        mean = torch.tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.tensor(self.image_std, dtype=dtype, device=device)
        image = (image - mean[:, None, None]) / std[:, None, None]
        return image
    
    def resize(self, image, target):
        # ==================================== image resize =======================================
        # print("Original size : ", image.shape) # new line added
        ori_image_shape = image.shape[-2:]
        size = [800, 1200]
        image = F.interpolate(image[None], size=size, mode='bilinear', align_corners=False)[0]
        # print("Resized size : ", image.shape) # new line added

        if target is None:
            return image, target
        

        # ================================= bounding box resize ===================================
        '''
        [x, y, w, h]
            x, y : center of bounding box
            w, h : height and width of bounding box
            actual image : ori_image_shape
            resize image : image.shape
        '''
        box = target['boxes']
        if (ori_image_shape[-1] < image.shape[1]): 
            bbox_width_percent = 100 - ((image.shape[1] * 100) / ori_image_shape[-1])
            bbox_width = bbox_width_percent / 100
            box[:, [0, 2]] = box[:, [0, 2]] + (box[:, [0, 2]] * bbox_width)
        else:
            bbox_width_percent = (image.shape[1] * 100) / ori_image_shape[-1]
            bbox_width = bbox_width_percent / 100
            box[:, [0, 2]] = box[:, [0, 2]] * bbox_width
        
        
        if (ori_image_shape[0] < image.shape[-2]) : 
            bbox_height_percent = 100 - ((image.shape[-2] * 100) / ori_image_shape[0])
            bbox_height = bbox_height_percent / 100
            box[:, [1, 3]] = box[:, [1, 3]] + box[:, [1, 3]] * bbox_height
        else:
            bbox_height = (image.shape[-2] * 100) / ori_image_shape[0]
            bbox_height = bbox_height_percent / 100
            box[:, [1, 3]] = box[:, [1, 3]] * bbox_height
        target['boxes'] = box
        

        # ==================================== mask resize ==========================================
        if 'masks' in target:
            mask = target['masks']
            mask = F.interpolate(mask[None].float(), size=size)[0].byte()
            target['masks'] = mask
        
        return image, target
    
    
    def batched_image(self, image, stride=32):
        size = image.shape[-2:]
        max_size = tuple(math.ceil(s / stride) * stride for s in size)
        batch_shape = (image.shape[-3],) + max_size
        batched_img = image.new_full(batch_shape, 0)
        batched_img[:, :image.shape[-2], :image.shape[-1]] = image
        
        return batched_img[None]


    def postprocess(self, result, image_shape, ori_image_shape):
        box = result['boxes']
        # box[:, [0, 2]] = box[:, [0, 2]] * ori_image_shape[1] / image_shape[1]
        if (ori_image_shape[-1] < image_shape.shape[1]): 
            bbox_width_percent = 100 - ((image_shape.shape[1] * 100) / ori_image_shape[-1])
            bbox_width = bbox_width_percent / 100
            box[:, [0, 2]] = box[:, [0, 2]] + (box[:, [0, 2]] * bbox_width)
        else:
            bbox_width_percent = (image_shape.shape[1] * 100) / ori_image_shape[-1]
            bbox_width = bbox_width_percent / 100
            box[:, [0, 2]] = box[:, [0, 2]] * bbox_width
        
        # box[:, [1, 3]] = box[:, [1, 3]] * ori_image_shape[0] / image_shape[0]
        if (ori_image_shape[0] < image_shape.shape[-2]) : 
            bbox_height_percent = 100 - ((image_shape.shape[-2] * 100) / ori_image_shape[0])
            bbox_height = bbox_height_percent / 100
            box[:, [1, 3]] = box[:, [1, 3]] + box[:, [1, 3]] * bbox_height
        else:
            bbox_height_percent = (image_shape.shape[-2] * 100) / ori_image_shape[0]
            bbox_height = bbox_height_percent / 100
            box[:, [1, 3]] = box[:, [1, 3]] * bbox_height
        
        result['boxes'] = box
        
        if 'masks' in result:
            mask = result['masks']
            mask = paste_masks_in_image(mask, box, 1, ori_image_shape)
            result['masks'] = mask
        
        return result


def expand_detection(mask, box, padding):
    M = mask.shape[-1]
    scale = (M + 2 * padding) / M
    padded_mask = torch.nn.functional.pad(mask, (padding,) * 4)
    
    w_half = (box[:, 2] - box[:, 0]) * 0.5
    h_half = (box[:, 3] - box[:, 1]) * 0.5
    x_c = (box[:, 2] + box[:, 0]) * 0.5
    y_c = (box[:, 3] + box[:, 1]) * 0.5
    
    w_half = w_half * scale
    h_half = h_half * scale
    
    box_exp = torch.zeros_like(box)
    box_exp[:, 0] = x_c - w_half
    box_exp[:, 2] = x_c + w_half
    box_exp[:, 1] = y_c - h_half
    box_exp[:, 3] = y_c + h_half
    return padded_mask, box_exp.to(torch.int64)


def paste_masks_in_image(mask, box, padding, image_shape):
    mask, box = expand_detection(mask, box, padding)
    
    N = mask.shape[0]
    size = (N,) + tuple(image_shape)
    im_mask = torch.zeros(size, dtype=mask.dtype, device=mask.device)
    for m, b, im in zip(mask, box, im_mask):
        b = b.tolist()
        w = max(b[2] - b[0], 1)
        h = max(b[3] - b[1], 1)
        
        m = F.interpolate(m[None, None], size=(h, w), mode='bilinear', align_corners=False)[0][0]
        
        x1 = max(b[0], 0)
        y1 = max(b[1], 0)
        x2 = min(b[2], image_shape[1])
        y2 = min(b[3], image_shape[0])
        
        im[y1:y2, x1:x2] = m[(y1 - b[1]):(y2 - b[1]), (x1 - b[0]):(x2 - b[0])]
    return im_mask


# ##################################################################################################



class GeneralizedDataset:
    """
    Main class for Generalized Dataset.
    """
    
    def __init__(self, max_workers=2, verbose=False):
        self.max_workers = max_workers
        self.verbose = verbose
        
        self.transformer = Transformer(
            min_size=800, max_size=1200, 
            image_mean=[0.485, 0.456, 0.406], 
            image_std=[0.229, 0.224, 0.225])
            
    def __getitem__(self, i):
        img_id = self.ids[i]
        image = self.get_image(img_id)
        image = transforms.ToTensor()(image)
        # target = self.get_target(img_id)["masks"]
        target = self.get_target(img_id)
        image, target = self.transformer(image, target) 
        
        return image, target   
    
    def __len__(self):
        return len(self.ids)
    
    def check_dataset(self, checked_id_file):
        """
        use multithreads to accelerate the process.
        check the dataset to avoid some problems listed in method `_check`.
        """
        
        if os.path.exists(checked_id_file):
            info = [line.strip().split(", ") for line in open(checked_id_file)]
            self.ids, self.aspect_ratios = zip(*info)
            return

        since = time.time()
        print("Checking the dataset...")

        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        seqs = torch.arange(len(self)).chunk(self.max_workers)
        tasks = [executor.submit(self._check, seq.tolist()) for seq in seqs]

        outs = []
        for future in as_completed(tasks):
            outs.extend(future.result())
        if not hasattr(self, "id_compare_fn"):
            self.id_compare_fn = lambda x: int(x)
        outs.sort(key=lambda x: self.id_compare_fn(x[0]))

        with open(checked_id_file, "w") as f:
            for img_id, aspect_ratio in outs:
                f.write("{}, {:.4f}\n".format(img_id, aspect_ratio))

        info = [line.strip().split(", ") for line in open(checked_id_file)]
        self.ids, self.aspect_ratios = zip(*info)
        print(f"checked id file: {checked_id_file}")
        print("{} samples are OK; {:.1f} seconds".format(len(self), time.time() - since))
        
    def _check(self, seq):
        out = []
        for i in seq:
            img_id = self.ids[i]
            target = self.get_target(img_id)
            boxes = target["boxes"]
            labels = target["labels"]
            masks = target["masks"]

            try:
                assert len(boxes) > 0, f"{i}: len(boxes) = 0"
                assert len(boxes) == len(labels), f"{i}: len(boxes) != len(labels)"
                assert len(boxes) == len(masks), f"{i}: len(boxes) != len(masks)"

                out.append((img_id, self._aspect_ratios[i]))
            except AssertionError as e:
                if self.verbose:
                    print(img_id, e)
        return out




