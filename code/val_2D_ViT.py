import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[224, 224]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_image(image, label, net, classes, patch_size=[512, 512]):
    """
    Test a single 2D image and its corresponding label using the provided network.

    Args:
        image (torch.Tensor): The input image tensor of shape (1, H, W).
        label (torch.Tensor): The ground truth label tensor of shape (1, H, W).
        net (torch.nn.Module): The trained segmentation network.
        classes (int): The number of classes in the segmentation task.
        patch_size (list): The target patch size for resizing.

    Returns:
        list: A list of metrics for each class.
    """
    # Ensure the image and label are on the CPU and detached from the computation graph
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    # Initialize the prediction array
    prediction = np.zeros_like(label)

    # Resize the image to the target patch size
    x, y = image.shape
    resized_image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)

    # Convert the resized image to a tensor and add batch and channel dimensions
    input_tensor = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).float().cuda()

    # Set the network to evaluation mode
    net.eval()

    # Perform inference
    with torch.no_grad():
        output = torch.argmax(torch.softmax(net(input_tensor), dim=1), dim=1).squeeze(0)
        output = output.cpu().detach().numpy()

        # Resize the output back to the original image size
        resized_output = zoom(output, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = resized_output

    # Initialize a list to store metrics for each class
    metric_list = []

    # Calculate metrics for each class
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    return metric_list