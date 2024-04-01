'''
TVM : Tensor Virtual Machine 
'''

# =====================================================================================================================

# Import Library
import torchvision.datasets as datasets
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms
import onnx
import tvm
import tvm.relay as relay
import numpy as np
import time
from tqdm import tqdm


# Test Dataset -------------------------------------------------------------------------------
transform = transforms.Compose([
        transforms.Resize(640),
        transforms.CenterCrop(640),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
data_folder = "/home/ubuntu/Datasets/COCO/coco/images/"
test_dataset = datasets.ImageFolder(root=data_folder, transform=transform)
subset_size=5000

# Create a DataLoader for the test dataset ---------------------------------------------------
subset_indices = list(range(0, 50000, 50))
subset_dataset = Subset(test_dataset, subset_indices)
test_loader = DataLoader(subset_dataset, batch_size=1, shuffle=False)

# Tensro Virtual Machine ---------------------------------------------------------------------
def eval_tvm(data_loader):
    input_shape = (1, 3, 640, 640)
    model_path="/home/ubuntu/workspace/vikas/yolov5/yolov5n_fp32_Distribution.onnx"
    onnx_model=onnx.load(model_path)
    input_name = "images"
    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        executor = relay.build_module.create_executor("graph", mod, tvm.cpu(0), target, params).evaluate()

    print("finished tvm convertion")

    x,y = next(iter(data_loader))
    ndarray = x.numpy()
    input_data = tvm.nd.array(ndarray.astype("float32"))
    start_time = time.time()
    
    # output = executor(input_data).numpy()
    end_time = time.time()
    inference_time = end_time - start_time
    print("Inference Time:", inference_time, "seconds")
    top1_correct = 0
    top5_correct = 0
    total_samples = 0

    #acc check
    for idx, (images, labels) in tqdm(enumerate(data_loader), total=5000, desc="Processing images"):
        # Set the input data
        numpy_images = images.numpy()
        input_data = tvm.nd.array(numpy_images.astype("float32"))
        tvm_output = executor(input_data).numpy()
        predicted_labels = np.argmax(tvm_output, axis=1)
        top1_correct += np.sum(predicted_labels == labels.numpy())

        # Calculate top-5 accuracy
        top5_predicted_labels = np.argsort(tvm_output, axis=1)[:, -5:]
        for i in range(labels.size(0)):
            if labels.numpy()[i] in top5_predicted_labels[i]:
                top5_correct += 1

        total_samples += labels.size(0)
        if idx >= 5000:
            break

    # Calculate accuracy ----------------------------------------------------------------------
    top1_accuracy /= 100
    top1_accuracy = top1_correct / total_samples
    top5_accuracy = top5_correct / total_samples

    print(f"Top-1 Accuracy: {top1_accuracy * 100:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy * 100:.2f}%")


# =====================================================================================================================


eval_tvm(test_loader)


