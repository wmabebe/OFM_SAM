import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pycocotools import mask as mask_utils
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
import json
from pycocotools import mask as coco_mask
from monai.losses import DiceLoss, FocalLoss, DiceFocalLoss, DiceCELoss
import tifffile
from patchify import patchify  #Only to handle large images
from datasets import Dataset as DatasetX
from tqdm import tqdm
import argparse
import gc
from statistics import mean
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import copy
import timeit
from torch.utils.data.dataloader import default_collate
import shutil

def get_args():
    parser = argparse.ArgumentParser()
    valid_datasets = ['sa1b', 'mito', 'coco']
    valid_trainables = ['e','m','p', 'em', 'ep','mp','emp']
    valid_submodels = ['lsm','ls','lm','sm','m','s']
    train_prompts = ['p','b','p+b','pob']
    test_prompts = ['p','b','p+b']
    loss_funcs = ['dice', 'dicefocal']
    reorder_interval = ['','once','per_epoch','per_batch']
    reorder_method = ['magnitude','movement','wanda']
    parser.add_argument('--dataset', type=str, choices=valid_datasets, default='sa1b', help='Dataset (choices: {%(choices)s}, default: %(default)s).')
    parser.add_argument('--trainable', type=str, choices=valid_trainables, default='em', help='Choice of SAM modules to train (Encoder, Mask decoder, Both) (choices: {%(choices)s}, default: %(default)s).')
    parser.add_argument('--sandwich', type=str, choices=valid_submodels, default='lsm', help='Sandwich configuration: submodels to train (Largest, Smallest, Medium) (choices: {%(choices)s}, default: %(default)s).')
    parser.add_argument('--train_prompt', type=str, choices=train_prompts, default='pb', help='Prompts used for trainig (Point, Box, Together, Either/Or) (choices: {%(choices)s}, default: %(default)s).')
    parser.add_argument('--test_prompt', type=str, choices=test_prompts, default='b', help='Prompts used for testing (Point, Box, Together) (choices: {%(choices)s}, default: %(default)s).')
    parser.add_argument('--train_subset', type=int, default=0, help='Specify training data size. Leave 0 if for default dataset size.')
    parser.add_argument('--test_subset', type=int, default=0, help='Specify testing data size. Leave 0 if for default dataset size.')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs.')
    parser.add_argument('--save_interval', type=int, default=10, help='Intervals for saving supermodel checkpoint during training.')
    parser.add_argument('--lr',type=float,default=1e-5,help='Learning rate.') #mito lr=1e-5
    parser.add_argument('--weight_decay',type=float,default=0.0001,help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=2, help='Specify dataloader batch size.')
    parser.add_argument('--prompt_batch_size', type=int, default=32, help='Specify prompt batch size.')
    parser.add_argument('--loss', type=str, default='dice', choices=loss_funcs, help='Loss function')
    parser.add_argument('--no_verbose', action='store_true', help='Enable to disable verbose mode.')
    parser.add_argument('--crop', action='store_true', help='Crop SA1B images for training with larger batch sizes.')
    parser.add_argument('--reorder', type=str, default='', choices=reorder_interval, help='How often to reorder the mlp layers space')
    parser.add_argument('--reorder_method', type=str, default='magnitude', choices=reorder_method, help='Reordering method')



    args = parser.parse_args()
    return args

def get_logger():
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger

def init_logs(log_file_name, log_dir=None):
    
    #mkdirs(log_dir)
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    log_path = log_file_name + '.log'

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


    logging.basicConfig(
        filename=os.path.join(log_dir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    return logger

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = ann["color"] if "color" in ann.keys() else np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)
#     del mask
#     gc.collect()

def show_masks_on_image(raw_image, masks, filename=None):
    plt.imshow(np.array(raw_image))
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for mask in masks:
        show_mask(mask, ax=ax, random_color=True)
    plt.axis("off")
    if not filename:
        plt.show()
        del mask
        gc.collect()
    else:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)  # Save the figure to file
        plt.close()

def save_masks(masks,filename):
    numpy_array = masks.squeeze().numpy()
    # Plot the NumPy array using matplotlib
    plt.imshow(numpy_array, cmap='gray')  # Assuming grayscale image
    plt.axis('off')  # Turn off axis
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)  # Save the figure
    plt.close()  # Close the plot to free memory
        
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


def show_box(box, ax, label="",linecolor=[]):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='black', facecolor=(0,0,0,0), lw=5))
    if label:
        # Add label to the box
        label_x = x0 + 0.5 * w  # x-coordinate of the label position
        label_y = y0 + 0.5 * h  # y-coordinate of the label position
        ax.text(label_x, label_y, label, fontsize=20, color='red',
                ha='center', va='center')

def save_preds(map,folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    for name, (img,point_prompt,box_prompt,gts,pred,iou) in map.items():
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        print(f'img {img.shape}')
        print(f'point prompt {point_prompt} shape = {point_prompt.shape}')
        print(f'box prompt {box_prompt} shape = {box_prompt.shape}')
        print(f'gts {gts.shape}')
        print(f'pred {pred.shape}')

        # point_prompt = point_prompt.squeeze(0)
        point_prompt = point_prompt.numpy()

        box_prompt = box_prompt.squeeze(0)
        box_prompt = box_prompt.numpy()



        #Transpose image for plotting
        img = img.transpose(1, 2, 0)

        axes[0].imshow(img)
        show_mask(gts, axes[0]) #plt.gca()
        show_points(point_prompt, np.array([1]), axes[0])
        show_box(box_prompt, axes[0])
        axes[0].set_title(f'Ground truth',fontsize=18)

        axes[1].imshow(img)
        show_mask(pred, axes[1])
        show_points(point_prompt, np.array([1]), axes[1])
        show_box(box_prompt, axes[1])
        axes[1].set_title(f'Prediction iou={iou}',fontsize=18)
        
        plt.axis('off')
        plot_path = os.path.join(folder, f'{name}.png')
        plt.tight_layout()
        plt.savefig(plot_path)

def structured_pruning(model,layers_to_prune,global_attn_indexes):
    """Prune the SAM model's vision encoder layers.

    Args:
        model (nn.module): SAM model.
        layers_to_prune (list[int]): Indices of layers to remove.
        global_attn_indexes (list[int]): Original SAM vision encoder global attention indices.

    Returns:
        (list[int],list[int]): Prunned layers and shifted global indices.
    """
    
    #Assert layer indices are valid
    for l in layers_to_prune:
        assert l in list(range(len(model.vision_encoder.layers)))
    
    #Remove overlapping indices from global_attn_indexes
    global_attn_indexes = [l for l in global_attn_indexes if l not in layers_to_prune]
    
    #Shift prunnable layers to left  Eg. layers 1,6,9 --> 1, 5, 7 
    layers_to_prune = [i-idx for idx,i in enumerate(layers_to_prune)]

    #Remove layers and shift global_attention indices accordingly
    for l in layers_to_prune:
        del model.vision_encoder.layers[l]
        global_attn_indexes = [i-1 if i > l else i for i in global_attn_indexes]


    model.vision_encoder.config.global_attn_indexes = global_attn_indexes
    model.vision_encoder.config.num_hidden_layers = 12 - len(layers_to_prune)

    return layers_to_prune, global_attn_indexes

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mutual_overlap(parent_mask, child_mask):
    intersection_mask = np.logical_and(parent_mask, child_mask)
    inter_inter_child = np.logical_and(intersection_mask, child_mask)
    inter_inter_parent = np.logical_and(intersection_mask, parent_mask)

    intersection_pixels = np.sum(intersection_mask)

    if not intersection_pixels:
        return 0, 0, 0

    child_pixels = np.sum(child_mask)
    parent_pixels = np.sum(parent_mask)
    
    inter_inter_child_pixels = np.sum(inter_inter_child)
    inter_inter_parent_pixels = np.sum(inter_inter_parent)

    child_overlap = inter_inter_child_pixels / child_pixels * 100
    parent_overlap = inter_inter_parent_pixels / parent_pixels * 100
    intersection_overlap = intersection_pixels / parent_pixels * 100
    
    # False, False -> Unrelated masks
    # True, False -> component mask
    # False, True -> reverse component (i.e. parent is the component of child)
    # True, True -> redundant masks

    return child_overlap, parent_overlap, intersection_overlap

def compute_iou(mask_pred, mask_gt):

    if isinstance(mask_pred, list):
        ious = []
        for pred, gt in zip(mask_pred,mask_gt):
            pred, gt = pred.cpu(), gt.cpu()
            intersection = torch.logical_and(pred, gt).sum().float()
            union = torch.logical_or(pred, gt).sum().float()
            iou = intersection / union if union > 0 else 0.0
            ious.append(iou)
        miou = sum(ious) / len(ious)
        return miou.item(), ious

    elif isinstance(mask_pred, torch.Tensor):
        pred, gt = mask_pred.cpu(), mask_gt.cpu()
        intersection = torch.logical_and(pred, gt).sum().float()
        union = torch.logical_or(pred, gt).sum().float()
        iou = intersection / union if union > 0 else 0.0
        return iou, [iou]
    else:
        raise Exception(f'mask_pred is neither a torch tensor nor a list!')

def calculate_metrics(predicted_mask, ground_truth_mask):    
    intersection = np.logical_and(predicted_mask, ground_truth_mask)
    union = np.logical_or(predicted_mask, ground_truth_mask)
    true_positives = np.sum(np.logical_and(predicted_mask, ground_truth_mask))
    #true_negatives = np.sum(np.logical_and(np.logical_not(predicted_mask), np.logical_not(ground_truth_mask)))
    false_positives = np.sum(np.logical_and(predicted_mask, np.logical_not(ground_truth_mask)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predicted_mask), ground_truth_mask))
    
    iou = np.sum(intersection) / np.sum(union)
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    #fpr = false_positives / (false_positives + true_negatives)
    f1 = 2 * (precision * recall) / (precision + recall) if precision * recall != 0 else 0
    
    return iou, recall, precision, f1

def compute_fpr(predicted_masks, ground_truth_mask):
    fpr = 0
    for predicted_mask in predicted_masks:
        intersection = np.logical_and(predicted_mask, ground_truth_mask)
        union = np.logical_or(predicted_mask, ground_truth_mask)
        iou = np.sum(intersection) / np.sum(union)
        fpr += iou
    return fpr

def toBinaryMask(coco, annIds, input_image_size):
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)
        
        #print(f'train_mask: {train_mask.shape}, new_mask: {new_mask.shape}')
        #Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask

def point_grid(image_size, n):
    """
    Generate a list of coordinates representing an nxn grid within the image's dimensions.

    Args:
    - image_size (tuple): Tuple representing the dimensions of the image (x, y).
    - n (int): Number of points along each dimension for the grid.

    Returns:
    - List of coordinates [(x1, y1), (x2, y2), ..., (xn, yn)].
    """
    x_max, y_max = image_size

    # Calculate the step size between points
    step_x = x_max / (n + 1)
    step_y = y_max / (n + 1)

    # Generate the grid of coordinates
    grid = [[int(i * step_x), int(j * step_y)] for i in range(1, n + 1) for j in range(1, n + 1)]

    return grid

def get_image_info(dataset_directory, num_images=1):
    image_mask_pairs = []
    for filename in os.listdir(dataset_directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(dataset_directory, filename)
            mask_filename = filename.replace(".jpg", ".json")
            mask_path = os.path.join(dataset_directory, mask_filename)
            if os.path.exists(mask_path):
                image_mask_pairs.append((image_path, mask_path))
    selected_pairs = random.sample(image_mask_pairs, min(num_images, len(image_mask_pairs)))
    return selected_pairs



def get_ground_truth_masks(mask_path):
    binary_masks = []
    with open(mask_path, 'r') as json_file:
        mask_data = json.load(json_file)
    for annotation in mask_data['annotations']:
        rle_mask = annotation['segmentation']
        binary_mask = coco_mask.decode(rle_mask)
        binary_masks.append(binary_mask)
    return binary_masks

# def calculate_metrics(pred_mask, gt_mask):
#     intersection = np.logical_and(pred_mask, gt_mask).sum()
#     union = np.logical_or(pred_mask, gt_mask).sum()
#     iou = intersection / union if union != 0 else 0
#     return iou

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
    F_loss = alpha * (1-pt)**gamma * BCE_loss
    return F_loss.mean() if reduction == 'mean' else F_loss.sum()

def dice_loss(inputs, targets, smooth=1e-6):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice

def valid_points_from_masks(gt_masks):
    points = []
    for mask in gt_masks:
        ys, xs = np.where(mask > 0)
        points += [(x, y) for x, y in zip(xs, ys)]
    return points

# Load tiff stack images and masks
def load_dataset(data_path, label_path, patch_size=256, step=256):
    
    #165 large images as tiff image stack
    large_images = tifffile.imread(data_path)
    large_masks = tifffile.imread(label_path)

    large_images.shape

    """Now. let us divide these large images into smaller patches for training. We can use patchify or write custom code."""
    all_img_patches = []
    for img in range(large_images.shape[0]):
        large_image = large_images[img]
        patches_img = patchify(large_image, (patch_size, patch_size), step=step)  #Step=256 for 256 patches means no overlap

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):

                single_patch_img = patches_img[i,j,:,:]
                all_img_patches.append(single_patch_img)

    images = np.array(all_img_patches)

    #Let us do the same for masks
    all_mask_patches = []
    for img in range(large_masks.shape[0]):
        large_mask = large_masks[img]
        patches_mask = patchify(large_mask, (patch_size, patch_size), step=step)  #Step=256 for 256 patches means no overlap

        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):

                single_patch_mask = patches_mask[i,j,:,:]
                single_patch_mask = (single_patch_mask / 255.).astype(np.uint8)
                all_mask_patches.append(single_patch_mask)

    masks = np.array(all_mask_patches)

    print("images.shape:",images.shape)
    print("masks.shape:",masks.shape)

    """Now, let us delete empty masks as they may cause issues later on during training. If a batch contains empty masks then the loss function will throw an error as it may not know how to handle empty tensors."""

    # Create a list to store the indices of non-empty masks
    valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
    # Filter the image and mask arrays to keep only the non-empty pairs
    filtered_images = images[valid_indices]
    filtered_masks = masks[valid_indices]
    print("Image shape:", filtered_images.shape)  # e.g., (num_frames, height, width, num_channels)
    print("Mask shape:", filtered_masks.shape)

    """Let us create a 'dataset' that serves us input images and masks for the rest of our journey."""



    # Convert the NumPy arrays to Pillow images and store them in a dictionary
    dataset_dict = {
        "image": [Image.fromarray(img) for img in filtered_images],
        "label": [Image.fromarray(mask) for mask in filtered_masks],
    }

    # Create the dataset using the datasets.Dataset class
    dataset = DatasetX.from_dict(dataset_dict)

    return dataset
  
def get_prompt_grid(array_size=256, grid_size=10, batch_size=1):

    # Generate the grid points
   x = np.linspace(0, array_size-1, grid_size)
   y = np.linspace(0, array_size-1, grid_size)

    # Generate a grid of coordinates
   xv, yv = np.meshgrid(x, y)

    # Convert the numpy arrays to lists
   xv_list = xv.tolist()
   yv_list = yv.tolist()

    # Combine the x and y coordinates into a list of list of lists
   input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)]
   input_points = torch.tensor(input_points).view(1, 1, grid_size*grid_size, 2)

   input_points = input_points.repeat(batch_size, 1, 1, 1)
   
   return input_points

def get_optimizer_and_scheduler(params,lr=1e-5,weight_decay=0,scheduler=None):
    optimizer = AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
    )
    if not scheduler:
        lower_bound_lr = lr / 1000
        scheduler = LambdaLR(optimizer, lr_lambda=lambda x: max(lower_bound_lr, 0.975**x))
    else:
        scheduler.optimizer = optimizer
    return optimizer, scheduler

def get_trainable_parameters(model,trainable):
    # Freeze modules that are not set as trainable
    if 'p' not in trainable:
        for name, param in model.named_parameters():
            if name.startswith("prompt_encoder"):
                param.requires_grad_(False)
    if 'e' not in trainable:
         for name, param in model.named_parameters():
            if name.startswith("vision_encoder"):
                param.requires_grad_(False)
    if 'm' not in trainable:
         for name, param in model.named_parameters():
            if name.startswith("mask_decoder"):
                param.requires_grad_(False)

    #Select the parameters to train
    trainable_params = []
    if 'e' in trainable:
        trainable_params += list(model.vision_encoder.parameters())
    if 'm' in trainable:
        trainable_params += list(model.mask_decoder.parameters())
    if 'p' in trainable:
        trainable_params += list(model.prompt_encoder.parameters())
    
    return trainable_params

# No need to define a new class
# Suppose you know the order of your customized Dataset
def sa1b_collate_fn(batch):
    return list(map(list, zip(*batch)))  # transpose list of list

def plot_dist(score_dist,filename='score-dist.png',importance='Magnitude'):
    # Create a figure with 3 rows and 4 columns of subplots
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))

    # Flatten the axes array for easy iteration
    axs = axs.flatten()

    # Iterate through the dictionary and the axes
    for i, (key, values) in enumerate(score_dist.items()):
        # Calculate the probability distribution
        count, bins, ignored = axs[i].hist(values, bins=100, density=True, alpha=0.6, color='g')
        
        # Plot the probability density function
        mu, sigma = np.mean(values), np.std(values)
        print(f'Head {key} min, mean, max ({min(values)},{mu},{max(values)})')
        best_fit_line = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2)
        axs[i].plot(bins, best_fit_line, '--', linewidth=2)
        
        # Set title and labels
        axs[i].set_title(f'Head {key}')
        axs[i].set_xlabel(importance)
        axs[i].set_ylabel('Probability')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    plt.savefig(filename)

#Collate for skipping blank masks in coco 
def none_skipper_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter(lambda x : x is not None, batch))
    return default_collate(batch)

def euclidean_distance(tensor1, tensor2):
    return torch.sqrt(torch.sum((tensor1 - tensor2) ** 2))

def embedding_distance(model_1, model_2, dataloader, device='cuda',disable_verbose=True):
    
    model_1 = model_1.vision_encoder.to(device)
    model_2 = model_2.vision_encoder.to(device)

    dists = []
    with torch.no_grad():
        for idx,(inputs, images, labels, boxes, points) in enumerate(tqdm(dataloader, disable=disable_verbose)):

            data = {'pixel_values': torch.stack([d['pixel_values'].squeeze(0) for d in inputs]),
                'original_sizes' : torch.stack([d['original_sizes'].squeeze(0) for d in inputs]),
                'reshaped_input_sizes' : torch.stack([d['reshaped_input_sizes'].squeeze(0) for d in inputs])}

            output_1 = model_1(data["pixel_values"].to(device))
            output_2 = model_2(data["pixel_values"].to(device))
            
            embeddings_1 = output_1[0]
            embeddings_2 = output_2[0]

            for emb_1, emb_2 in zip(embeddings_1, embeddings_2):
                dists.append(euclidean_distance(emb_1, emb_2))


    # print(f'embeddings_1 : {len(embeddings_1)}')
    # print(embeddings_1)
    # print(f'embeddings_2 : {len(embeddings_2)}')
    # print(embeddings_2)

    return sum(dists) / len(dists)



def eval(model,dataloader,device='cuda',disable_verbose=False,processor=None, prompt='p',num_objects=0,pretrained=None):
    map = {}
    mIoU = []
    model.eval()
    model = nn.DataParallel(model).to(device)
    if pretrained:
        pretrained = pretrained.to(device)
    for idx,(inputs, images, labels, boxes, points) in enumerate(tqdm(dataloader, disable=disable_verbose)): 

        #If batch_size=1, set num_objects to all objects in the current data point
        if len(points) == 1:
            num_objects = len(points[0])
        #If num_objects specified, set as upper limit for num_objects 
        elif num_objects:
            num_objects = min(num_objects, min([len(pts) for pts in points]))
        #Set num_objects to number of labels of the image with fewest labels
        else:
            num_objects = min([len(pts) for pts in points])
        

        # Filter num_objects prompt indices from points and boxes
        input_boxes, input_points = [], []
        for i,d in enumerate(inputs):
            indices = random.sample(range(d['input_boxes'].size(1)), num_objects)
            input_boxes.append(d['input_boxes'][:,indices, :])
            input_points.append(d['input_points'][indices, :, :])
            labels[i] = [labels[i][j] for j in indices]
        input_boxes = torch.stack(input_boxes)
        input_points = torch.stack(input_points)

        
        ignore = []
        # for i in range(len(points)):
        #     # print(f'points[{i}]: {len(points[i])}')
        #     if len(points[i]) < num_objects:
        #         ignore.append(i)
        
        # images = [image for idx, image in enumerate(images) if idx not in ignore]
        labels = [label for idx, label in enumerate(labels) if idx not in ignore]
        # points = [point for idx, point in enumerate(points) if idx not in ignore]
        # boxes = [box for idx, box in enumerate(boxes) if idx not in ignore]
            
        # # print(f'ignore : {ignore}')

        # # print(f'boxes : {torch.tensor(boxes).shape}') 
        # # print(f'points: {torch.tensor([points]).shape}')

        data = {'pixel_values': torch.stack([d['pixel_values'] for i,d in enumerate(inputs) if i not in ignore]),
                 'original_sizes' : torch.stack([d['original_sizes'] for i,d in enumerate(inputs) if i not in ignore]),
                 'reshaped_input_sizes' : torch.stack([d['reshaped_input_sizes'] for i,d in enumerate(inputs) if i not in ignore]),
                 'input_boxes' : input_boxes, #torch.stack([d['input_boxes'][:, :num_objects, :] for i,d in enumerate(inputs) if i not in ignore]),
                 'input_points' : input_points }#torch.stack([d['input_points'][:num_objects, :, :] for i,d in enumerate(inputs) if i not in ignore]) }
        
        data["pixel_values"] = data["pixel_values"].squeeze(1)
        data['original_sizes'] = data['original_sizes'].squeeze(1)
        data['reshaped_input_sizes'] = data['reshaped_input_sizes'].squeeze(1)
        data['input_points'] = data['input_points'].squeeze(1)
        data['input_boxes'] = data['input_boxes'].squeeze(1)
        

        # print(f'pixel_values.shape : {data["pixel_values"].shape}')
        # print(f'original_sizes.shape : {data["original_sizes"].shape}')
        # print(f'reshaped_input_sizes.shape : {data["reshaped_input_sizes"].shape}')
        # print(f'input_points.shape : {data["input_points"].shape}')
        # print(f'input_boxes.shape : {data["input_boxes"].shape}')
        # print(f'labels size: {len(labels)}')

        with torch.no_grad():
            if prompt == 'p':
                outputs = model(pixel_values=data["pixel_values"].to(device),
                                input_points=data["input_points"].to(device),
                                multimask_output=True)
                if pretrained:
                    outputs_pretrained = pretrained(pixel_values=data["pixel_values"].to(device),
                                input_points=data["input_points"].to(device),
                                multimask_output=False)
            elif prompt == 'b': 
                outputs = model(pixel_values=data["pixel_values"].to(device),
                                input_boxes=data["input_boxes"].to(device),
                                multimask_output=True)
                if pretrained:
                    outputs_pretrained = pretrained(pixel_values=data["pixel_values"].to(device),
                                input_boxes=data["input_boxes"].to(device),
                                multimask_output=False)
            
            masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), 
                                                                     data["original_sizes"].cpu(), 
                                                                     data["reshaped_input_sizes"].cpu())
            scores = outputs.iou_scores

            if pretrained:
                golden = processor.image_processor.post_process_masks(outputs_pretrained.pred_masks.cpu(), 
                                                                     data["original_sizes"].cpu(), 
                                                                     data["reshaped_input_sizes"].cpu())


            #print(f'masks : {len(masks)}')

            #loop through batch (images)
            for i, mask in enumerate(masks):
                pts, bxs = points[i], boxes[i]
                #loop through objects
                for j,(pred_1, pred_2, pred_3) in enumerate(mask):
                    if pretrained:
                        print(f'\tpred_1.shape : {pred_1.shape}')
                        print(f'\tgolden[{i}].shape : {golden[i].shape}')
                        print(f'\tgolden[{i}][{j}].shape : {golden[i][j].shape}')
                        
                        _, mious = compute_iou([pred_1, pred_2, pred_3],[golden[i][j],golden[i][j],golden[i][j]])
                        mIoU.append(max(mious))

                    else:
                        gt = mask_utils.decode(labels[i][j]['segmentation'])
                        gt = torch.from_numpy(gt)
                        _, mious = compute_iou([pred_1, pred_2, pred_3],[gt,gt,gt])

                        mIoU.append(max(mious))

                    # pt, bx = [pts[j]], bxs[j]
                    # print(f'pt : {pt} \t bx : {bx}')
                    # map[f'img-{i}-obj-{j}-pred-0'] = (images[i],torch.tensor(pt),torch.tensor(bx),gt,pred_1,mious[0])
                    # map[f'img-{i}-obj-{j}-pred-1'] = (images[i],torch.tensor(pt),torch.tensor(bx),gt,pred_2,mious[1])
                    # map[f'img-{i}-obj-{j}-pred-2'] = (images[i],torch.tensor(pt),torch.tensor(bx),gt,pred_3,mious[2])

    model = model.module
    model = model.to('cpu')
    mIoU = torch.tensor(mIoU)
    average_mIoU = torch.mean(mIoU)
    return average_mIoU, mIoU, map

#Test model performance on given dataset
def test(model,dataloader,size=10000,device='cuda',disable_verbose=False,processor=None, prompt='b'):
    ragged_tensors = False
    map = {}
    mIoU = []
    count = 0
    model.eval()
    model = nn.DataParallel(model)
    model = model.to(device)

    for (data, gts) in tqdm(dataloader, disable=disable_verbose):
    # For plotting gts, preds side by side
    #for (data, gts, imgs, bboxes, points) in tqdm(dataloader, disable=disable_verbose):

        # print(f'data : {data}')
        # print(f'gts : {gts}')
        # exit()

        # Convert torch tensor to NumPy array
        # idx = random.randint(0, 31)
        # numpy_array = data[idx].squeeze().numpy()  # Squeeze the tensor to remove the batch dimension
        # plt.imshow(numpy_array)
        # plt.axis('off')  # Turn off axis
        # plt.title('Image')  # Set title
        # plt.savefig('coco_image.jpg', bbox_inches='tight', pad_inches=0)

        # numpy_array = gts[idx].squeeze().numpy()  # Squeeze the tensor to remove the batch dimension
        # plt.imshow(numpy_array)
        # plt.axis('off')  # Turn off axis
        # plt.title('Mask')  # Set title
        # plt.savefig('coco_mask.jpg', bbox_inches='tight', pad_inches=0)

        #SA1B case rework data returned from the dataloader
        if type(data) == list:
            ragged_tensors = True
            data = {'pixel_values': torch.stack([d['pixel_values'] for d in data]),
                 'original_sizes' : torch.stack([d['original_sizes'] for d in data]),
                 'reshaped_input_sizes' : torch.stack([d['reshaped_input_sizes'] for d in data]),
                 'input_boxes' : torch.stack([d['input_boxes'] for d in data]),
                 'input_points' : torch.stack([d['input_points'] for d in data])}

        # forward pass
        with torch.no_grad():

            #forward with point only
            if prompt == 'p':
                outputs = model(pixel_values=data["pixel_values"].to(device),
                            input_points=data["input_points"].to(device),
                            multimask_output=False)
            
            #forward with box only
            elif prompt == 'b':
                outputs = model(pixel_values=data["pixel_values"].to(device),
                                input_boxes=data["input_boxes"].to(device),
                                multimask_output=False)

            # outputs = my_mito_model(**inputs, multimask_output=False)
        
        pred_masks = outputs.pred_masks.to('cpu')

        masks = processor.image_processor.post_process_masks(
            pred_masks, data["original_sizes"], data["reshaped_input_sizes"]
        )

        #If images have varying shapes pass (list,list) to compute_miou
        if ragged_tensors:
            predicted_masks_binary = []
            for mask in [m.squeeze(0) for m in masks]:
                single_patch_prob = torch.sigmoid(mask)
                # convert soft mask to hard mask
                single_patch_prob = single_patch_prob.squeeze()
                bin_mask = (single_patch_prob > 0.5)
                predicted_masks_binary.append(bin_mask)

            ground_truth_masks = gts
            miou, mious = compute_iou(predicted_masks_binary,ground_truth_masks)
            mIoU.append(miou)

            # print(f'imgs : {len(imgs)}')
            # print(f'gts : {len(gts)}')
            # print(f'preds : {len(predicted_masks_binary)}')
            # print(f'mIoU : {len(mIoU)}')

            #Mapping plots
            # for idx,iou in enumerate(mious):
            #     map[iou] = (imgs[idx],points[idx],bboxes[idx],gts[idx],predicted_masks_binary[idx])

        else:
            #print(f'masks[0] : {masks[0].shape}')
            predicted_masks = torch.stack([m.squeeze(0) for m in masks])

            # compute iou
            #print(f'outputs.pred_masks shape : {outputs.pred_masks.shape}')
            #predicted_masks = outputs.pred_masks.squeeze(1)
            #predicted_masks = predicted_masks.squeeze(1)
            #ground_truth_masks = data["ground_truth_mask"].float().to(device)
            ground_truth_masks = gts
            #ground_truth_masks = torch.stack([gt for gt in gts])

            # print(f'predicted_masks.shape : {predicted_masks.shape}')
            # print(f'ground_truth_masks.shape : {ground_truth_masks.shape}')

            # apply sigmoid
            single_patch_prob = torch.sigmoid(predicted_masks)
            # convert soft mask to hard mask
            single_patch_prob = single_patch_prob.squeeze()
            predicted_masks_binary = (single_patch_prob > 0.5)

            #print(f'predicted_masks_binary.shape : {predicted_masks_binary.shape}')

            #predicted_masks_binary = masks[0]

            # print(f'predicted_masks_binary shape : {predicted_masks_binary.shape}')
            # print(f'ground_truth_masks shape : {ground_truth_masks.shape}')
            miou, mious = compute_iou(predicted_masks_binary,ground_truth_masks)
            mIoU.append(miou)

        # Visualize masks
        # to_pil = transforms.ToPILImage()
        # raw_image = batch["pixel_values"].squeeze(0).cpu()
        # masks = [masks[0].cpu()]
        # ground_truth_masks = ground_truth_masks.cpu()
        # raw_image = to_pil(raw_image)
        # save_masks(masks[0], f'logs/{count}predicted.png')
        # save_masks(ground_truth_masks, f'logs/{count}ground_truth.png')

        if count == size:
          break

        count += 1

    model = model.module
    model = model.to('cpu')
    return round(sum(mIoU) / len(mIoU), 4), mIoU, map

def train(submodel, optimizer, epochs, train_dataloader, processor, device, no_verbose=True):
    #Training loop
    # Initialize the optimizer and the loss function
    seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    dice_focal = DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean',lambda_focal=0.95,lambda_dice=0.05)

    ragged_tensors = False
    submodel.to(device)
    submodel.train()
    for epoch in range(epochs):
        epoch_losses = []
        for (data, gts) in tqdm(train_dataloader, disable=no_verbose):

            #SA1B case rework data returned from the dataloader 
            if type(data) == list:
                ragged_tensors = True
                data = {'pixel_values': torch.stack([d['pixel_values'] for d in data]),
                 'original_sizes' : torch.stack([d['original_sizes'] for d in data]),
                 'reshaped_input_sizes' : torch.stack([d['reshaped_input_sizes'] for d in data]),
                 'input_boxes' : torch.stack([d['input_boxes'] for d in data])}

            # forward pass
            outputs = submodel(pixel_values=data["pixel_values"].to(device),
                        input_boxes=data["input_boxes"].to(device),
                        multimask_output=False)

            
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks, data["original_sizes"], data["reshaped_input_sizes"], binarize=False
            )
            
            #data images have varying shapes (not stackable)
            if ragged_tensors:
                predicted_masks = [m.squeeze(0) for m in masks]
                ground_truth_masks = gts
                
                loss = torch.zeros(1, dtype=predicted_masks[0].dtype, device=device)

                for pred, gt in zip(predicted_masks, gts):
                    gt = gt.to(device)
                    loss += dice_focal(pred.squeeze(0), gt)

                loss /= len(gts)

            #data images have identical shapes (hence stackable)
            else:
                predicted_masks = torch.stack([m.squeeze(0) for m in masks])

                # compute loss
                #predicted_masks = masks[0] #outputs.pred_masks.squeeze(1)
                #ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                ground_truth_masks = gts.float().to(device)

                loss = dice_focal(predicted_masks, ground_truth_masks.unsqueeze(1))
                
                #Updated Loss
                # Calculate focal loss and dice loss
                # loss_focal = focal_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
                # loss_dice = dice_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
                
                # Combine the losses
                # loss = 20 * loss_focal + loss_dice

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')

def training_step_encoder(pretrained, supermodel, model, data, gts, optimizer, scheduler, device):
    
    mse_loss = nn.MSELoss(reduction='mean')
    local_grad = {k: v.cpu() for k, v in model.state_dict().items()}

    with torch.no_grad():
        pretrained.to(device)
        pretrained = nn.DataParallel(pretrained)
        targets = pretrained.module.vision_encoder(pixel_values=data["pixel_values"].to(device))
        targets = pretrained.module.get_image_embeddings(data["pixel_values"].to(device))
        pretrained = pretrained.module
        pretrained = pretrained.to('cpu')


    model.to(device)
    model = nn.DataParallel(model)
    model.train()

    optimizer.zero_grad()

    # forward pass
    outputs = model.module.vision_encoder(data["pixel_values"].to(device))

    loss = mse_loss(targets,outputs.last_hidden_state)

    # backward pass (compute gradients of parameters w.r.t. loss)
    optimizer.zero_grad()
    loss.backward()
    # loss.sum().backward()
    optimizer.step()
    scheduler.step()

    model = model.module
    model = model.to('cpu')

    with torch.no_grad():
        for k, v in model.state_dict().items():
            local_grad[k] = local_grad[k] - v.cpu()

    supermodel.apply_grad(local_grad)

    train_metrics = {
        "train_loss": loss.sum().item(),
        "params": model.config.num_parameters,
    }
    return train_metrics

def interactive_training_step(supermodel, model, data, gts, optimizer, scheduler, processor, device, ragged_tensors, soft_labels=None, steps=4):
    dice_focal = DiceFocalLoss(lambda_focal=0.95,lambda_dice=0.05)

    local_grad = {k: v.cpu() for k, v in model.state_dict().items()}

    model.to(device)
    model = nn.DataParallel(model)

    model.train()
    optimizer.zero_grad()

    #Generate embeddings once
    image_embeddings = model.module.vision_encoder(pixel_values=data["pixel_values"].to(device))

    sparse_embeddings, dense_embeddings = model.module.prompt_encoder(input_points=data["input_points"].to(device),
                                                                      input_labels=torch.ones(len(data["input_points"])),
                                                                      input_boxes=None,input_masks=None)
    image_positional_embeddings = model.module.get_image_wide_positional_embeddings()
        

    # forward pass uniformly selects either box or point prompts for mask generation
    outputs = model.module.mask_decoder(image_embeddings, image_positional_embeddings,
                                             sparse_embeddings, dense_embeddings, 
                                             multimask_output=False)

    #post process outputs and retreive masks    
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks, data["original_sizes"], data["reshaped_input_sizes"], binarize=False
    )
    
    #data images have varying shapes (not stackable)
    if ragged_tensors:
        predicted_masks = [m.squeeze(0) for m in masks]
        loss = torch.zeros(1, dtype=predicted_masks[0].dtype, device=device)

        for pred, gt in zip(predicted_masks, gts):
            gt = gt.to(device)
            # focal_loss = focal(pred.squeeze(0), soft_labels if soft_labels else gt)
            # dice_loss = dice(pred.squeeze(0), soft_labels if soft_labels else gt)
            # loss += (0.95 * focal_loss) + (0.05 * dice_loss)

            loss += dice_focal(pred.squeeze(0), soft_labels if soft_labels else gt)


        loss /= len(gts)

    #data images have identical shapes (hence stackable)
    else:
        predicted_masks = torch.stack([m.squeeze(0) for m in masks])
        gts = gts.float().to(device)
        #loss = seg_loss(predicted_masks, soft_labels if soft_labels else gts.unsqueeze(1))
        
        # focal_loss = focal(predicted_masks, soft_labels if soft_labels else gts.unsqueeze(1))
        # dice_loss = dice(predicted_masks, soft_labels if soft_labels else gts.unsqueeze(1))
        # loss = (0.95 * focal_loss) + (0.05 * dice_loss)

        loss = dice_focal(predicted_masks, soft_labels if soft_labels else gts.unsqueeze(1))

    # backward pass (compute gradients of parameters w.r.t. loss)
    optimizer.zero_grad()
    loss.backward()
    # loss.sum().backward()
    optimizer.step()
    scheduler.step()

    model = model.module
    model = model.to('cpu')

    with torch.no_grad():
        for k, v in model.state_dict().items():
            local_grad[k] = local_grad[k] - v.cpu()

    supermodel.apply_grad(local_grad)

    train_metrics = {
        "train_loss": loss.sum().item(),
        "params": model.config.num_parameters,
    }
    return train_metrics

        
def training_step(supermodel, model, data, gts, optimizer, scheduler, processor, device, ragged_tensors, soft_labels=None, prompt='pb',loss_func='dice'):
    
    #seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    if loss_func == 'dice':
        loss_func = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    elif loss_func == 'dicefocal':
        loss_func = DiceFocalLoss(lambda_focal=0.95,lambda_dice=0.05,sigmoid=True, squared_pred=True, reduction='mean')

        
    local_grad = {k: v.cpu() for k, v in model.state_dict().items()}

    model.to(device)
    model = nn.DataParallel(model)

    model.train()
    optimizer.zero_grad()

    #forward pass randomly selects either box or point prompts for mask generation
    if prompt == 'pob':
        if random.choice(['point', 'box']) == 'point':
            outputs = model(pixel_values=data["pixel_values"].to(device),
                        input_points=data["input_points"].to(device),
                        multimask_output=False)
        else:
            outputs = model(pixel_values=data["pixel_values"].to(device),
                        input_boxes=data["input_boxes"].to(device),
                        multimask_output=False)
    #forward with point and box
    elif prompt == 'p+b':
        outputs = model(pixel_values=data["pixel_values"].to(device),
                        input_points=data["input_points"].to(device),
                        input_boxes=data["input_boxes"].to(device),
                        multimask_output=False)
    #forward with point only
    elif prompt == 'p':
        outputs = model(pixel_values=data["pixel_values"].to(device),
                        input_points=data["input_points"].to(device),
                        multimask_output=False)
     #forward with box only
    elif prompt == 'b':
        outputs = model(pixel_values=data["pixel_values"].to(device),
                        input_boxes=data["input_boxes"].to(device),
                        multimask_output=False)

    #post process outputs and retreive masks    
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks, data["original_sizes"], data["reshaped_input_sizes"], binarize=False
    )
    
    #data images have varying shapes (not stackable)
    if ragged_tensors:
        predicted_masks = [m.squeeze(0) for m in masks]
        loss = torch.zeros(1, dtype=predicted_masks[0].dtype, device=device)

        for pred, gt in zip(predicted_masks, gts):
            gt = gt.to(device)
            loss += loss_func(pred.squeeze(0), soft_labels.squeeze(0) if soft_labels else gt.squeeze(0))

        loss /= len(gts)

    #data images have identical shapes (hence stackable)
    else:
        predicted_masks = torch.stack([m.squeeze(0) for m in masks])
        gts = gts.float().to(device)
        loss = loss_func(predicted_masks, soft_labels if soft_labels else gts.unsqueeze(1))

    # backward pass (compute gradients of parameters w.r.t. loss)
    optimizer.zero_grad()
    loss.backward()
    # loss.sum().backward()
    optimizer.step()
    scheduler.step()

    model = model.module
    model = model.to('cpu')

    with torch.no_grad():
        for k, v in model.state_dict().items():
            local_grad[k] = local_grad[k] - v.cpu()

    supermodel.apply_grad(local_grad)

    train_metrics = {
        "train_loss": loss.sum().item(),
        "params": model.config.num_parameters,
    }
    return predicted_masks, train_metrics

def single_step_encoder(args,submodel,data, gts, model_size, do_test=False):
    trainable_params = get_trainable_parameters(submodel, 'e')
    optimizer, scheduler = get_optimizer_and_scheduler(trainable_params,args.lr)
    start_train = timeit.default_timer()
    metrics = training_step_encoder(args.pretrained, args.supermodel, submodel, data, gts, optimizer, scheduler, args.device)
    end_train = timeit.default_timer()
    if do_test:
        start_test = timeit.default_timer()
        miou = test(submodel,args.test_dataloader,disable_verbose=args.no_verbose,processor=args.processor,prompt=args.test_prompt)
        end_test = timeit.default_timer()
        metrics['test_miou'] = miou
        args.logger.info(f'\t{model_size} submodel train time : {round(end_train-start_train,4)}, test time {round(end_test-start_test,4)}, metrics {metrics}')
    else:
        args.logger.info(f'\t{model_size} submodel train time : {round(end_train-start_train,4)}, metrics {metrics}')


def single_step(args,submodel,data,gts,ragged_tensors, model_size, do_test=False):
    trainable_params = get_trainable_parameters(submodel, args.trainable)
    optimizer, args.scheduler = get_optimizer_and_scheduler(trainable_params,args.lr,args.weight_decay, args.scheduler)
    start_train = timeit.default_timer()
    preds, metrics = training_step(args.supermodel, submodel, data, gts, optimizer, args.scheduler, 
                                    args.processor, args.device, ragged_tensors, soft_labels=None, 
                                    prompt=args.train_prompt, loss_func=args.loss)
    end_train = timeit.default_timer()
    if do_test:
        start_test = timeit.default_timer()
        miou = test(submodel,args.test_dataloader,disable_verbose=args.no_verbose,processor=args.processor,prompt=args.test_prompt)
        end_test = timeit.default_timer()
        metrics['test_miou'] = miou
        args.logger.info(f'\t{model_size} submodel train time : {round(end_train-start_train,4)}, test time {round(end_test-start_test,4)}, metrics {metrics}')
    else:
        args.logger.info(f'\t{model_size} submodel train time : {round(end_train-start_train,4)}, metrics {metrics}')

    return preds

def train_nas(args):
    #Training loop
    trainable_params = get_trainable_parameters(args.supermodel.model, args.trainable)
    _, args.scheduler = get_optimizer_and_scheduler(trainable_params,args.lr,args.weight_decay)
    ragged_tensors = False
    for epoch in range(args.epochs):
        args.logger.info(f'EPOCH {epoch}: starts')
        #Train encoder only halfway, then finetune end-2-end
        #args.trainable = 'e' if epoch < args.epochs / 2 else 'em' 
        start_epoch = timeit.default_timer()
        for idx, (data, gts) in enumerate(tqdm(args.train_dataloader, disable=args.no_verbose)):

            #SA1B case rework data returned from the dataloader 
            if type(data) == list:
                ragged_tensors = True
                data = {'pixel_values': torch.stack([d['pixel_values'] for d in data]),
                 'original_sizes' : torch.stack([d['original_sizes'] for d in data]),
                 'reshaped_input_sizes' : torch.stack([d['reshaped_input_sizes'] for d in data]),
                 'input_boxes' : torch.stack([d['input_boxes'] for d in data]),
                 'input_points' : torch.stack([d['input_points'] for d in data])}
            
            #set to True to test the submodels after training step
            do_test = (idx == len(args.train_dataloader) - 1)
            
            #Intervals at which to save the model checkpoint
            save_interval = len(args.train_dataloader) // args.save_interval if len(args.train_dataloader) > 10 else 1
            do_save = ((idx + 1) % save_interval == 0)
            
            soft_preds = None
            
            #Train largest submodel (= supernet)
            if 'l' in args.sandwich:
                submodel, submodel.config.num_parameters, submodel.config.arch = (
                        copy.deepcopy(args.supermodel.model),
                        args.supermodel.total_params,
                        {},
                    )
                
                if 'e' in args.trainable and 'm' in args.trainable:
                    soft_preds = single_step(args,submodel,data,gts,ragged_tensors,model_size='Largest',do_test=do_test)
                    # with torch.no_grad():
                    #     if type(soft_preds) == list:
                    #          soft_preds = [sp.clone().detach() for sp in soft_preds]
                    #     else:
                    #         soft_preds = soft_preds.clone().detach()
                else:
                    single_step_encoder(args,submodel,data, gts, model_size='Largest', do_test=do_test)

            #Train smallest model
            if 's' in args.sandwich:
                submodel, submodel.config.num_parameters, submodel.config.arch = args.supermodel.smallest_model()
                if 'e' in args.trainable and 'm' in args.trainable:
                    single_step(args,submodel,data,gts,ragged_tensors,model_size='Smallest',do_test=do_test)
                else:
                    single_step_encoder(args,submodel,data, gts, model_size='Smallest', do_test=do_test)

            #Train arbitrary medium sized model
            if 'm' in args.sandwich:
                submodel, submodel.config.num_parameters, submodel.config.arch = args.supermodel.random_resource_aware_model()
                if 'e' in args.trainable and 'm' in args.trainable:
                    single_step(args,submodel,data,gts,ragged_tensors,model_size='Medium',do_test=do_test)
                else:
                    single_step_encoder(args,submodel,data, gts, model_size='Medium', do_test=do_test)

            # Save the supermodel state dictionary to a file
            if do_save:
                args.supermodel.save_ckpt(f'{args.log_dir}')
                args.logger.info(f'\tInterval {(idx + 1) // save_interval}: Model checkpoint saved.')

        end_epoch = timeit.default_timer()

        # Save the supermodel state dictionary to a file
        args.supermodel.save_ckpt(f'{args.log_dir}')
        args.logger.info(f'EPOCH {epoch}: ends {round(end_epoch-start_epoch,4)} seconds. Model checkpoint saved.')   
        