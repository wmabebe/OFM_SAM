import numpy as np
import torch
from PIL import Image
import torch
import os
from PIL import Image
import json
from pycocotools import mask as mask_utils
import random
from PIL import Image
from pycocotools.coco import COCO
from tqdm import trange
import cv2
import glob

class SA1BDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_directory, processor, encoder=None, max_labels=64, split='train'):
        self.datadir = dataset_directory
        self.processor = processor
        self.imgs, self.labels = SA1BDataset.get_image_json_pairs(self.datadir)
        self.encoder = encoder
        self.max_labels = max_labels
        self.split = split
        

    def __len__(self):
        return len(self.imgs)
    
    def loader(self,file_path):
        #image = Image.open(file_path)
        image = cv2.imread(file_path)
        #image = np.array(image)
        #image = np.moveaxis(image, -1, 0)
        return image

    @staticmethod
    #Crop region around masked object
    def cropper(image, ground_truth_map,padding=50):
        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # Pad the crop
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - padding)
        x_max = min(W, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(H, y_max + padding)

        #For color image
        if len(image.shape) > 2:
            cropped_image = image[:, y_min:y_max, x_min:x_max]
        #For grayscale mask
        else:
            cropped_image = image[y_min:y_max, x_min:x_max]

        return cropped_image
    
    @staticmethod
    #Get bounding boxes from mask.
    def get_bounding_box(ground_truth_map):
        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]

        return bbox
    
    @staticmethod
    def get_random_prompt(ground_truth_map,bbox):
        x_min, y_min, x_max, y_max = bbox
        while True:
            # Generate random point within the bounding box
            x = np.random.randint(x_min, x_max)
            y = np.random.randint(y_min, y_max)
            # Check if the point lies inside the mask
            if ground_truth_map[y, x] == 1:
                return x, y
    
    @staticmethod
    def get_image_json_pairs(directory):
        jpg_files = []
        json_files = []

        for filename in os.listdir(directory):
            if filename.endswith(".jpg"):
                jpg_files.append(filename)
                json_file = filename[:-4] + '.json'
                json_files.append(json_file)

        return jpg_files, json_files
    
    @staticmethod
    def get_embedding(filepath):
        return torch.load(filepath)
    
    def filter_n_masks(self,masks):
        #Random trimming for train data points
        if self.split == 'train':
            if len(masks) < self.max_labels:
                while len(masks) < self.max_labels:
                    masks.append(random.choice(masks))
            elif len(masks) > self.max_labels:
                while len(masks) > self.max_labels:
                    masks.pop(random.randint(0, len(masks) - 1))
        #Fixed trimming for val data points
        if self.split == 'val':
            if len(masks) < self.max_labels:
                while len(masks) < self.max_labels:
                    masks.append(masks[0])
            elif len(masks) > self.max_labels:
                masks = masks[:self.max_labels]
        return masks
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_path =  os.path.join(self.datadir, self.imgs[index]) # discard automatic subfolder labels 
        label_path = os.path.join(self.datadir, self.labels[index])
        image = self.loader(img_path)
        masks = json.load(open(label_path))['annotations'] # load json masks

        if self.split == 'to_embedding':
            inputs = self.processor(image, return_tensors="pt")
            embed_file = self.imgs[index][:-4] + '.pt'
            embed_path = os.path.join(self.datadir,embed_file)
            return inputs, embed_path
        elif self.split == 'from_embedding':
            inputs = self.processor(image, return_tensors="pt")
            embed_file = self.imgs[index][:-4] + '.pt'
            embed_path = os.path.join(self.datadir,embed_file)
            embedding = SA1BDataset.get_embedding(embed_path)

            # remove batch dimension which the processor adds by default
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}

            return inputs, embedding

        elif self.split == 'val':
            bin_masks, points, boxes = [], [], []
            for mask in masks:
                bin_masks.append(mask_utils.decode(mask['segmentation']))
            
            image, bin_masks = resize_image_and_mask(image, bin_masks)

            bin_masks = [(mask > 0).astype(float) for mask in bin_masks]
            bin_masks = [torch.tensor(mask) for mask in bin_masks if np.sum(mask) > 100]
            
            if 0 == len(bin_masks):
                return None

            bin_masks = self.filter_n_masks(bin_masks)
            
            for bin_mask in bin_masks:
                bbox_prompt = SA1BDataset.get_bounding_box(bin_mask)
                point_prompt = SA1BDataset.get_random_prompt(bin_mask,bbox_prompt)
                points.append([point_prompt])
                boxes.append(bbox_prompt)
        
            inputs = self.processor(image, input_points=points, input_boxes=[[boxes]], return_tensors="pt")
            
            # remove batch dimension which the processor adds by default
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}

            #attach stuff to inputs
            inputs["ground_truth_masks"] = torch.stack([torch.tensor(bin_mask) for bin_mask in bin_masks])
            inputs["boxes"] = torch.stack([torch.tensor(box) for box in boxes])
            inputs["points"] = torch.stack([torch.tensor(point) for point in points])
            inputs["img_id"] = index

            image = np.array(image)
            image = np.moveaxis(image, -1, 0)

            inputs["image"] = image
            
            return inputs  #, image, masks, boxes,  points
                
        elif self.split == 'train':
            bin_masks, points, boxes = [], [], []
            for mask in masks:
                bin_masks.append(mask_utils.decode(mask['segmentation']))
            
            image, bin_masks = resize_image_and_mask(image, bin_masks)

            bin_masks = [(mask > 0).astype(float) for mask in bin_masks]
            bin_masks = [torch.tensor(mask) for mask in bin_masks if np.sum(mask) > 100]

            if 0 == len(bin_masks):
                return None

            bin_masks = self.filter_n_masks(bin_masks)

            for bin_mask in bin_masks:
                bbox_prompt = SA1BDataset.get_bounding_box(bin_mask)
                point_prompt = SA1BDataset.get_random_prompt(bin_mask,bbox_prompt)
                points.append([point_prompt])
                boxes.append(bbox_prompt)
                
            inputs = self.processor(image, input_points=[points], input_boxes=[[boxes]], return_tensors="pt")

            # remove batch dimension which the processor adds by default
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}

             #attach stuff to inputs
            inputs["ground_truth_masks"] = torch.stack([torch.tensor(bin_mask) for bin_mask in bin_masks])
            inputs["boxes"] = torch.stack([torch.tensor(box) for box in boxes])
            inputs["points"] = torch.stack([torch.tensor(point) for point in points])
            inputs["img_id"] = index

            image = np.array(image)
            image = np.moveaxis(image, -1, 0)
            
            return inputs  #, image, masks, boxes,  points

    
class MitoDataset(torch.utils.data.Dataset):
    """
    This class is used to create a dataset that serves input images and masks.
    It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
    """
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)
  
    @staticmethod
    #Get bounding boxes from mask.
    def get_bounding_box(ground_truth_map):
        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        ready = False
        while not ready:
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            bbox = [x_min, y_min, x_max, y_max]
            ready = y_min < y_max and x_min < x_max

        return bbox
    
    @staticmethod
    def get_random_prompt(ground_truth_map,bbox):
        x_min, y_min, x_max, y_max = bbox
        while True:
            # Generate random point within the bounding box
            x = np.random.randint(x_min, x_max)
            y = np.random.randint(y_min, y_max)
            # Check if the point lies inside the mask
            if ground_truth_map[y, x] == 1:
                return x, y

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])

        ground_truth_mask = torch.tensor(ground_truth_mask)

        # get bounding box prompt
        bbox_prompt = MitoDataset.get_bounding_box(ground_truth_mask)
        point_prompt = MitoDataset.get_random_prompt(ground_truth_mask,bbox_prompt)


        # Convert the image to grayscale (if it's not already in grayscale)
        image = image.convert("L")

        # Add a channel dimension
        image = image.convert("RGB")

        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[bbox_prompt]],input_points=[[point_prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        inputs["boxes"] = torch.tensor([bbox_prompt])
        inputs["points"] =  torch.tensor([point_prompt])
        inputs["img_id"] = idx
        inputs["ground_truth_mask"] = ground_truth_mask.squeeze()

        image = np.array(image)
        image = np.moveaxis(image, -1, 0)
        inputs["image"] = image

        return inputs


class COCOSegmentation(torch.utils.data.Dataset):
    NUM_CLASSES = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
        1, 64, 20, 63, 7, 72]

    def __init__(self,
                 args,
                 base_dir='datasets/coco',
                 split='train',
                 year='2017',
                 max_labels=8,
                 processor=None):
        super().__init__()
        ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split, year))
        ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(base_dir, '{}{}'.format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask_utils
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.args = args
        self.processor = processor
        self.max_labels = max_labels
        self.mask_idx = 0 if split == 'val' else 1
    
    @staticmethod
    def get_bounding_box(ground_truth_map):
        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]

        return bbox

    @staticmethod
    def get_random_prompt(ground_truth_map,bbox):
        x_min, y_min, x_max, y_max = bbox
        while True:
            # Generate random point within the bounding box
            x = np.random.randint(x_min, x_max)
            y = np.random.randint(y_min, y_max)
            # Check if the point lies inside the mask
            if ground_truth_map[y, x] > 0:
                return x, y


    def __getitem__(self, index):
        #Grab image and associated masks
        _img, _target = self._make_img_gts_pair(index)

        #Resize image, masks to size 256x256
        _img, _target = resize_image_and_mask(_img,_target)
        
        image, masks = np.array(_img), [np.array(t) for t in _target]
        image = np.moveaxis(image, -1, 0)

        #binarize mask
        masks = [(mask > 0).astype(float) for mask in masks]

        masks = [torch.tensor(mask) for mask in masks if np.sum(mask) > 25]

        if 0 == len(masks):
            return None

        masks = self.filter_n_masks(masks)

        if self.split == 'val':
            points, boxes = [], []
            for mask in masks:
                bbox_prompt = COCOSegmentation.get_bounding_box(mask)
                point_prompt = COCOSegmentation.get_random_prompt(mask,bbox_prompt)
                points.append([point_prompt])
                boxes.append(bbox_prompt)
        
            inputs = self.processor(image, input_points=points, input_boxes=[[boxes]], return_tensors="pt")
            # remove batch dimension which the processor adds by default
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}

            #attach stuff to inputs
            inputs["ground_truth_masks"] = torch.stack([torch.tensor(mask) for mask in masks])
            inputs["boxes"] = torch.stack([torch.tensor(box) for box in boxes])
            inputs["points"] = torch.stack([torch.tensor(point) for point in points])
            inputs["img_id"] = index
            inputs["image"] = image
            
            return inputs  #, image, masks, boxes,  points
                
        elif self.split == 'train':
            points, boxes = [], []
            for mask in masks:
                bbox_prompt = COCOSegmentation.get_bounding_box(mask)
                point_prompt = COCOSegmentation.get_random_prompt(mask,bbox_prompt)
                points.append([point_prompt])
                boxes.append(bbox_prompt)
            
            inputs = self.processor(image, input_points=[points], input_boxes=[[boxes]], return_tensors="pt") #Add input_boxes=[[boxes]]

            # remove batch dimension which the processor adds by default
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}

            #attach stuff to inputs
            inputs["ground_truth_masks"] = torch.stack([torch.tensor(mask) for mask in masks])
            inputs["boxes"] = torch.stack([torch.tensor(box) for box in boxes])
            inputs["points"] = torch.stack([torch.tensor(point) for point in points])
            inputs["img_id"] = index
            inputs["image"] = image
            
            return inputs  #, image, masks, boxes,  points

    def filter_n_masks(self,masks):
        #Random trimming for train data points
        if self.split == 'train':
            if len(masks) < self.max_labels:
                while len(masks) < self.max_labels:
                    masks.append(random.choice(masks))
            elif len(masks) > self.max_labels:
                while len(masks) > self.max_labels:
                    masks.pop(random.randint(0, len(masks) - 1))
        #Fixed trimming for val data points
        if self.split == 'val':
            if len(masks) < self.max_labels:
                while len(masks) < self.max_labels:
                    masks.append(masks[0])
            elif len(masks) > self.max_labels:
                masks = masks[:self.max_labels]
        return masks


    def _make_img_gt_point_pair(self, index, mask_idx=0):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        annids = coco.getAnnIds(imgIds=img_id)
        if mask_idx == 0:
            annid = annids[0]
        else:
            idx = random.randint(0, len(annids) - 1)
            annid = annids[idx] 
        cocotarget = coco.loadAnns(annid)
        _target = Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']))

        return _img, _target

    def _make_img_gts_pair(self,index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        annids = coco.getAnnIds(imgIds=img_id)
        _targets = []
        for annid in annids:
            cocotarget = coco.loadAnns(annid)
            _targets.append(Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width'])))
        
        return _img, _targets

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def __len__(self):
        return len(self.ids)

def resize_image_and_mask(image, mask, target_size=(256, 256)):
    """Resize (image, masks) pair

    Args:
        image (PIL.Image, np.ndarray): PIL image or 
        mask ([PIL.Image], [np.ndarray]): _description_
        target_size (tuple, optional): _description_. Defaults to (256, 256).

    Returns:
        (PIL.Image,[PIL.Image]),(): resized (image, masks) pair
    """
    if isinstance(image, Image.Image):
        resized_image = image.resize(target_size, Image.BILINEAR)
    elif isinstance(image, np.ndarray):
        resized_image = cv2.resize(image, dsize=target_size, interpolation=cv2.INTER_LINEAR)
    
    # Resizing for PIL mask
    if isinstance(mask, Image.Image):
        resized_mask = mask.resize(target_size, Image.NEAREST)  # Use NEAREST for mask to preserve label values
    # Resizing for numpy mask
    elif isinstance(mask, np.ndarray):
        resized_mask = cv2.resize(mask, dsize=target_size, interpolation=cv2.INTER_NEAREST)
    
    elif isinstance(mask,list):
        # Resize the PIL masks
        if isinstance(mask[0],Image.Image):
            resized_mask = [m.resize(target_size, Image.NEAREST) for m in mask]
        # Resizing for numpy masks
        elif isinstance(mask[0],np.ndarray):
            resized_mask = [cv2.resize(m, dsize=target_size, interpolation=cv2.INTER_NEAREST) for m in mask]
    
    return resized_image, resized_mask



class ADE20KDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, split='training', processor=None, max_labels=8):
        super().__init__()
        self.base_dir = base_dir
        self.split = split
        self.processor = processor
        self.max_labels = max_labels
        # Set image and annotation paths based on the split
        if split == 'training':
            self.image_dir = os.path.join(base_dir, 'images/training')
            self.annotation_dir = os.path.join(base_dir, 'annotations/training')
        elif split == 'validation':
            self.image_dir = os.path.join(base_dir, 'images/validation')
            self.annotation_dir = os.path.join(base_dir, 'annotations/validation')
        elif split == 'testing':
            self.image_dir = os.path.join(base_dir.replace("ADEChallengeData2016", ""), 'release_test/testing')
            self.annotation_dir = None  # No annotations available for the test set
 
        # Collect image paths
        self.images = sorted(glob.glob(os.path.join(self.image_dir, '*.jpg')))
        # Collect annotation paths only if they exist (for training and validation splits)
        if self.annotation_dir:
            self.annotations = sorted(glob.glob(os.path.join(self.annotation_dir, '*.png')))
            assert len(self.images) == len(self.annotations), "Mismatch between image and annotation counts"
        else:
            self.annotations = None
 
    def __len__(self):
        return len(self.images)
 
    @staticmethod
    def get_bounding_box(ground_truth_map):
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        return [x_min, y_min, x_max, y_max]
 
    @staticmethod
    def get_random_prompt(ground_truth_map, bbox):
        x_min, y_min, x_max, y_max = bbox
        while True:
            x = np.random.randint(x_min, x_max)
            y = np.random.randint(y_min, y_max)
            if ground_truth_map[y, x] > 0:
                return x, y
 
    def filter_n_masks(self, masks):
        if len(masks) < self.max_labels:
            while len(masks) < self.max_labels:
                masks.append(random.choice(masks))
        elif len(masks) > self.max_labels:
            masks = masks[:self.max_labels]
        return masks
 
    def __getitem__(self, index):
        img_path = self.images[index]
 
        # Load the image
        image = Image.open(img_path).convert('RGB')
 
        if self.annotations:
            mask_path = self.annotations[index]
            mask = np.array(Image.open(mask_path))  # Load mask as a numpy array
 
            # Since the mask is black and white, create binary masks
            unique_labels = np.unique(mask)
            unique_labels = unique_labels[unique_labels != 0]  # Exclude background
 
            bin_masks = [(mask == label).astype(float) for label in unique_labels]
            bin_masks = [np.where(mask != 0, 1, 0) for mask in bin_masks]
            
 
            if len(bin_masks) == 0:
                return None

            image, bin_masks = resize_image_and_mask(image, bin_masks)

            #bin_masks = [np.array(mask) for mask in bin_masks if np.sum(mask) > 100]
            bin_masks = [(mask > 0).astype(float) for mask in bin_masks]
            bin_masks = [torch.tensor(mask) for mask in bin_masks if np.sum(mask) > 100]

            if 0 == len(bin_masks):
                return None

            bin_masks = self.filter_n_masks(bin_masks)
 
            points, boxes = [], []
            for bin_mask in bin_masks:
                bbox = ADE20KDataset.get_bounding_box(bin_mask)
                point = ADE20KDataset.get_random_prompt(bin_mask, bbox)
                points.append([point])
                boxes.append(bbox)
 
            if self.processor:
                inputs = self.processor(image, input_points=points, input_boxes=[[boxes]], return_tensors="pt")
            else:
                inputs = {"image": np.array(image), "points": points, "boxes": boxes}
            
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
 
            # remove batch dimension which the processor adds by default
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}

            #attach stuff to inputs
            inputs["ground_truth_masks"] = torch.stack([torch.tensor(bin_mask) for bin_mask in bin_masks])
            inputs["boxes"] = torch.stack([torch.tensor(box) for box in boxes])
            inputs["points"] = torch.stack([torch.tensor(point) for point in points])
            inputs["img_id"] = index

            image = np.array(image)
            image = np.moveaxis(image, -1, 0)

            inputs["image"] = image
            
            return inputs
        else:
            if self.processor:
                inputs = self.processor(image, return_tensors="pt")
                inputs["img_id"] = index
                return inputs
            else:
                return {"image": np.array(image), "img_id": index}