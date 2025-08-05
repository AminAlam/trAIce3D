import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

def calculate_ssim(output, desired_output):
    output = torch.nn.functional.sigmoid(output)
    output = output.float()

    ssim_operator = SSIM(data_range=1.0)
    ssim = ssim_operator(output, desired_output)
    return ssim

def calculate_F1_score(output, desired_output):
    output = torch.sigmoid(output)
    output = output.float()
    if torch.max(output) != 0:
        output = output / torch.max(output)
    desired_output = desired_output.float()
    if torch.max(desired_output) != 0:
        desired_output = desired_output / torch.max(desired_output)
    # output = torch.round(output)
    # desired_output = torch.round(desired_output)
    TP = torch.sum((output > 0.5) & (desired_output > 0.5)).float()
    FP = torch.sum((output > 0.5) & (desired_output < 0.5)).float()
    FN = torch.sum((output < 0.5) & (desired_output > 0.5)).float()
    TN = torch.sum((output < 0.5) & (desired_output < 0.5)).float()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * precision * recall / (precision + recall)
    return F1_score

def calculate_accuracy(output, desired_output):
    output = torch.sigmoid(output)
    output = output.float()
    if torch.max(output) != 0:
        output = output / torch.max(output)
    desired_output = desired_output.float()
    if torch.max(desired_output) != 0:
        desired_output = desired_output / torch.max(desired_output)
    output = torch.round(output)
    desired_output = torch.round(desired_output)
    TP = torch.sum((output > 0.5) & (desired_output > 0.5)).float()
    FP = torch.sum((output > 0.5) & (desired_output < 0.5)).float()
    FN = torch.sum((output < 0.5) & (desired_output > 0.5)).float()
    TN = torch.sum((output < 0.5) & (desired_output < 0.5)).float()
    accuracy = TP / (TP + FP + FN)
    return accuracy

def calculate_FP(output, desired_output):
    output = torch.nn.functional.sigmoid(output)
    output = output.float()
    if torch.max(output) != 0:
        output = output / torch.max(output)
    desired_output = desired_output.float()
    if torch.max(desired_output) != 0:
        desired_output = desired_output / torch.max(desired_output)
    output = torch.round(output)
    desired_output = torch.round(desired_output)
    FP = torch.sum((output > 0.5) & (desired_output == 0)).float()
    return FP

def plot_sample_soma(sample, model, device):
    images = np.array(sample['image'], dtype=np.uint8)
    soma_images = np.array(sample['soma'], dtype=np.uint8)
    soma = sample['soma']
    img = sample['image']
    soma = soma[None,:,:,:]
    img = img[None,:,:,:]
    input2model = img
    input2model = input2model.type(torch.float32)
    input2model = input2model.to(device)
    output = model(input2model)
    output = np.squeeze(output.cpu().detach().numpy())
    # output = output>0.5
    output = output/np.max(output)*255
    segmented_image = output
    sample_segmented = np.expand_dims(segmented_image, axis=0)
    sample_img = images
    sample_soma = soma_images
    order = [3, 1, 2, 0] # z x y c
    segmented = np.transpose(sample_segmented, order)
    image = np.transpose(sample_img, order)
    if np.max(segmented) != 0:
        segmented = segmented/np.max(segmented)
    if np.max(image) != 0:
        image = image/np.max(image)

    return_arg = []
    for img_ in [sample_soma]:

        trace = np.transpose(img_, order)
        if np.max(trace) != 0:
            trace = trace/np.max(trace)

        # null_img = np.transpose(null_img, order)
        sample = np.concatenate([image, trace, segmented], axis=3)
        print(f"image ==> Min: {np.min(image)}, Max: {np.max(image)}")
        print(f"trace ==> Min: {np.min(trace)}, Max: {np.max(trace)}")
        print(f"Soma ==> Min: {np.min(segmented)}, Max: {np.max(segmented)}")

        sample = np.transpose(sample, [3, 0, 1, 2]) # c z x y

        # calculate the maximum z intensity projection
        max_z_projection = np.max(sample, axis=1)
        # calculate the maximum x intensity projection
        max_x_projection = np.max(sample, axis=2)
        # calculate the maximum y intensity projection
        max_y_projection = np.max(sample, axis=3)

        max_z_projection = np.transpose(max_z_projection, [1, 2, 0])
        max_x_projection = np.transpose(max_x_projection, [1, 2, 0])
        max_y_projection = np.transpose(max_y_projection, [1, 2, 0])

        # plot them in a subplot
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        ax.imshow(max_z_projection)
        ax.set_title('Maximum Z Intensity Projection')
        plt.tight_layout()

        return_arg.append(fig)
        plt.close(fig)

    return return_arg   