import cv2  # type: ignore
import numpy as np # type: ignore



# #############################################################################################################

import torch # type: ignore
import torch.nn.functional as F  # type: ignore
def resize(image, target):
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




# ============================== Draw the bounding box =====================================
def drawBox(boxes, image):
    for i in range(len(boxes)):
        # changed color and width to make it visible
        cv2.rectangle(image, (boxes[i][2], boxes[i][3]), (boxes[i][4], boxes[i][5]), (255, 0, 0), 1)
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image, boxes


def bounding_box():
    image_path = "image/continental650.jpg"
    ori_image = cv2.imread(image_path, 3)
    print("Original Image Size : ", ori_image.shape)

    # Set the scale to resize
    size = [1200, 800]
    x_scale = size[0] / ori_image.shape[1]
    y_scale = size[1] / ori_image.shape[0]
    print(x_scale, y_scale)
    
    # make image static
    image = cv2.resize(ori_image, (size[0], size[1]))
    print("Resized Image Size : ", image.shape)
    
    img = np.array(image) # convert image to numpy

    # original frame as named values
    (x_pos, y_pos, w_pos, h_pos) = (550, 2400, 2910, 4400)

    x = int(np.round(x_pos * x_scale))
    y = int(np.round(y_pos * y_scale))
    w = int(np.round(w_pos * x_scale))
    h = int(np.round(h_pos * y_scale))
    print("target : ", [x, y, w, h])
    
    image, boxes = drawBox([[1, 0, x, y, w, h]], img)
    
    # ===============================================================================
    new_size = [1400, 1000]
    image = cv2.resize(ori_image, (new_size[0], new_size[1]))
    target = []
    


def main():
    bounding_box()


if __name__ == "__main__":
    main()

