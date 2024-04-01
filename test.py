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
        return (image - mean[:, None, None]) / std[:, None, None]
    
    def resize(self, image, target):
        # ==================================== image resize =======================================
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
        

        # =================================== mask resize ========================================
        if 'masks' in target:
            mask = target['masks']
            mask = F.interpolate(mask[None].float(), size=size)[0].byte()
            target['masks'] = mask
        
        return image, target
    