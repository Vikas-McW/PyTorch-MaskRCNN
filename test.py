
import cv2  # type: ignore
import torch # type: ignore



# #############################################################################################################

import torch # type: ignore
import torchvision.transforms.functional as F  # type: ignore

class Transformer:
    def __init__(self, image, target):
        image = self.normalize(image)
        image, target = self.resize(image, target)
        # image = self.batched_image(image)
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
        
        bbox = image.shape[-1] / ori_image_shape[1]
        box[:, [0, 2]] = torch.round(box[:, [0, 2]] * bbox).int()
        
        bbox = image.shape[-2] / ori_image_shape[0]
        box[:, [1, 3]] = torch.round(box[:, [1, 3]] * bbox).int()
        
        target['boxes'] = box
        
        
        # =================================== mask resize ========================================
        if 'masks' in target:
            mask = target['masks']
            mask = F.interpolate(mask[None].float(), size=size)[0].byte()
            target['masks'] = mask
        
        return image, target

# #############################################################################################################




def main():
    image_path = "image/continental650.jpg"
    ori_image = cv2.imread(image_path)
    # -----------------------------------------------------------
    image = torch.from_numpy(ori_image).permute(2, 0, 1).float()   # Convert the image to the format expected by the resize method
    print(image.shape)
    target = {"boxes": torch.tensor([[100, 100, 200, 150]])} 
    
    
    # # resize logic
    # trf_image, trf_target = Transformer(image, target)
    
    # -----------------------------------------------------------
    cv2.imshow("Original Image", ori_image)
    cv2.waitKey(0)
    
    # cv2.imshow("Transformed Image", trf_image)
    # cv2.waitKey(0)
    # cv2.imwrite('trf_image.jpg', trf_image)
    
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
