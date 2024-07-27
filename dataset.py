import numpy as np
import torch
from PIL import Image
import torch
import os
from PIL import Image
import json
from pycocotools import mask as mask_utils
import random
from PIL import Image, ImageOps, ImageFilter
from pycocotools.coco import COCO
from torchvision import transforms
import custom_transforms as tr
from tqdm import trange
import pickle


class SA1BDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_directory, processor, do_crop=False, encoder=None, label='first'):
        self.datadir = dataset_directory
        self.processor = processor
        self.imgs, self.labels = SA1BDataset.get_image_json_pairs(self.datadir)
        self.do_crop = do_crop
        self.encoder = encoder
        self.label = label
        

    def __len__(self):
        return len(self.imgs)
    
    def loader(self,file_path):
        image = Image.open(file_path)
        image = np.array(image)
        image = np.moveaxis(image, -1, 0)
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
                return [x, y]
    
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

        #Just pick the first mask
        if self.label == 'first':
            mask = masks[0]
            bin_ground_truth_mask = mask_utils.decode(mask['segmentation'])
        elif self.label == 'rand':
            idx = random.randint(0, len(masks) - 1)
            mask = masks[idx]
            bin_ground_truth_mask = mask_utils.decode(mask['segmentation'])
        elif self.label == 'to_embedding':
            inputs = self.processor(image, return_tensors="pt")
            embed_file = self.imgs[index][:-4] + '.pt'
            embed_path = os.path.join(self.datadir,embed_file)
            return inputs, embed_path
        elif self.label == 'from_embedding':
            inputs = self.processor(image, return_tensors="pt")
            embed_file = self.imgs[index][:-4] + '.pt'
            embed_path = os.path.join(self.datadir,embed_file)
            embedding = SA1BDataset.get_embedding(embed_path)

            return inputs, embedding


        elif self.label == 'all_test':
            points, boxes = [], []
            for mask in masks:
                bin_ground_truth_mask = mask_utils.decode(mask['segmentation'])
                bbox_prompt = SA1BDataset.get_bounding_box(bin_ground_truth_mask)
                point_prompt = SA1BDataset.get_random_prompt(bin_ground_truth_mask,bbox_prompt)
                points.append(point_prompt)
                boxes.append(bbox_prompt)
        
            inputs = self.processor(image, input_points=points, input_boxes=[[boxes]], return_tensors="pt")
            return inputs, image, masks, boxes,  points
                
        elif self.label == 'all_train':
            points, boxes = [], []
            for mask in masks:
                bin_ground_truth_mask = mask_utils.decode(mask['segmentation'])
                bbox_prompt = SA1BDataset.get_bounding_box(bin_ground_truth_mask)
                point_prompt = SA1BDataset.get_random_prompt(bin_ground_truth_mask,bbox_prompt)
                points.append([point_prompt])
                boxes.append(bbox_prompt)
                
            inputs = self.processor(image, input_points=[points], input_boxes=[[boxes]], return_tensors="pt")
            return inputs, image, masks, boxes,  points


        #Crop image and mask
        if self.do_crop:
            image = SA1BDataset.cropper(image,bin_ground_truth_mask)
            bin_ground_truth_mask = SA1BDataset.cropper(bin_ground_truth_mask,bin_ground_truth_mask)

        bbox_prompt = SA1BDataset.get_bounding_box(bin_ground_truth_mask)
        point_prompt = SA1BDataset.get_random_prompt(bin_ground_truth_mask,bbox_prompt)
        bin_ground_truth_mask = torch.tensor(bin_ground_truth_mask)
        
        # prepare image and prompt for the model
        inputs = self.processor(image, input_points=[[point_prompt]], input_boxes=[[bbox_prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        if self.encoder:
            with torch.no_grad():
                encoding = self.encoder(inputs["pixel_values"])
                return inputs, encoding
        
        return inputs, bin_ground_truth_mask #, image, torch.tensor(bbox_prompt),  torch.tensor([point_prompt])
    
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

        return inputs, ground_truth_mask
    
class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


# ImageFile.LOAD_TRUNCATED_IMAGES = True


class COCOSegmentation(torch.utils.data.Dataset):
    NUM_CLASSES = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
        1, 64, 20, 63, 7, 72]

    def __init__(self,
                 args,
                 base_dir='datasets/coco',
                 split='train',
                 year='2017',
                 processor=None):
        super().__init__()
        ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split, year))
        ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(base_dir, '{}{}'.format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.args = args
        self.processor = processor
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
            if ground_truth_map[y, x] == 1:
                return x, y

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index,self.mask_idx)
        sample = {'image': _img, 'label': _target}

        if self.split == "train":
            sample =  self.transform_tr(sample)
        elif self.split == 'val':
            sample = self.transform_val(sample)
        
        if self.processor:
            image, mask = sample['image'], sample['label']

            if not torch.any(mask):
                return None

            img = (image - image.min()) / (image.max() - image.min())

            mask = torch.tensor(mask)
            #binarize mask
            mask = (mask > 0).float()

            bbox_prompt = COCOSegmentation.get_bounding_box(mask)
            point_prompt = COCOSegmentation.get_random_prompt(mask,bbox_prompt)

            

            # prepare image and prompt for the model
            inputs = self.processor(img, input_points=[[point_prompt]], input_boxes=[[bbox_prompt]], return_tensors="pt")

            # remove batch dimension which the processor adds by default
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}

            return inputs, mask
        else:
            return sample

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

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        
        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


    def __len__(self):
        return len(self.ids)



# """MSCOCO Semantic Segmentation pretraining for VOC."""

# class COCOSegmentation(SegmentationDataset):
#     """COCO Semantic Segmentation Dataset for VOC Pre-training.

#     Parameters
#     ----------
#     root : string
#         Path to ADE20K folder. Default is './datasets/coco'
#     split: string
#         'train', 'val' or 'test'
#     transform : callable, optional
#         A function that transforms the image
#     Examples
#     --------
#     >>> from torchvision import transforms
#     >>> import torch.utils.data as data
#     >>> # Transforms for Normalization
#     >>> input_transform = transforms.Compose([
#     >>>     transforms.ToTensor(),
#     >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
#     >>> ])
#     >>> # Create Dataset
#     >>> trainset = COCOSegmentation(split='train', transform=input_transform)
#     >>> # Create Training Loader
#     >>> train_data = data.DataLoader(
#     >>>     trainset, 4, shuffle=True,
#     >>>     num_workers=4)
#     """
#     CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
#                 1, 64, 20, 63, 7, 72]
#     NUM_CLASS = 21

#     def __init__(self, root='./datasets/coco', split='train', mode=None, transform=None, processor=None, **kwargs):
#         super(COCOSegmentation, self).__init__(root, split, mode, transform, **kwargs)
#         # lazy import pycocotools

#         if split == 'train':
#             print('train set')
#             ann_file = os.path.join(root, 'annotations/instances_train2017.json')
#             ids_file = os.path.join(root, 'annotations/train_ids.mx')
#             self.root = os.path.join(root, 'train2017')
#         else:
#             print('val set')
#             ann_file = os.path.join(root, 'annotations/instances_val2017.json')
#             ids_file = os.path.join(root, 'annotations/val_ids.mx')
#             self.root = os.path.join(root, 'val2017')
#         self.coco = COCO(ann_file)
#         self.coco_mask = mask
#         if os.path.exists(ids_file):
#             with open(ids_file, 'rb') as f:
#                 self.ids = pickle.load(f)
#         else:
#             ids = list(self.coco.imgs.keys())
#             self.ids = self._preprocess(ids, ids_file)
#         self.transform = transform
#         self.processor = processor
    
#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, index):
#         coco = self.coco
#         img_id = self.ids[index]
#         img_metadata = coco.loadImgs(img_id)[0]
#         path = img_metadata['file_name']
#         img = Image.open(os.path.join(self.root, path)).convert('RGB')
#         cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
#         mask = Image.fromarray(self._gen_seg_mask(
#             cocotarget, img_metadata['height'], img_metadata['width']))
#         # synchrosized transform
#         if self.mode == 'train':
#             img, mask = self._sync_transform(img, mask)
#         elif self.mode == 'val':
#             img, mask = self._val_sync_transform(img, mask)
#         else:
#             assert self.mode == 'testval'
#             img, mask = self._img_transform(img), self._mask_transform(mask)
#         # general resize, normalize and toTensor
#         if self.transform is not None:
#             img = self.transform(img)
        
#         bbox_prompt = COCOSegmentation.get_bounding_box(mask)
#         point_prompt = COCOSegmentation.get_random_prompt(mask,bbox_prompt)

#         mask = torch.tensor(mask)

#         # prepare image and prompt for the model
#         inputs = self.processor(img, input_points=[[point_prompt]], input_boxes=[[bbox_prompt]], return_tensors="pt")

#         # remove batch dimension which the processor adds by default
#         inputs = {k:v.squeeze(0) for k,v in inputs.items()}

#         return inputs, mask        
        
#         #return img, mask #, os.path.basename(self.ids[index])

#     def _mask_transform(self, mask):
#         return torch.LongTensor(np.array(mask).astype('int32'))

#     def _gen_seg_mask(self, target, h, w):
#         mask = np.zeros((h, w), dtype=np.uint8)
#         coco_mask = self.coco_mask
#         for instance in target:
#             rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
#             m = coco_mask.decode(rle)
#             cat = instance['category_id']
#             if cat in self.CAT_LIST:
#                 c = self.CAT_LIST.index(cat)
#             else:
#                 continue
#             if len(m.shape) < 3:
#                 mask[:, :] += (mask == 0) * (m * c)
#             else:
#                 mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
#         return mask

#     def _preprocess(self, ids, ids_file):
#         print("Preprocessing mask, this will take a while." + \
#               "But don't worry, it only run once for each split.")
#         tbar = trange(len(ids))
#         new_ids = []
#         for i in tbar:
#             img_id = ids[i]
#             cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
#             img_metadata = self.coco.loadImgs(img_id)[0]
#             mask = self._gen_seg_mask(cocotarget, img_metadata['height'], img_metadata['width'])
#             # more than 1k pixels
#             if (mask > 0).sum() > 1000:
#                 new_ids.append(img_id)
#             tbar.set_description('Doing: {}/{}, got {} qualified images'. \
#                                  format(i, len(ids), len(new_ids)))
#         print('Found number of qualified images: ', len(new_ids))
#         with open(ids_file, 'wb') as f:
#             pickle.dump(new_ids, f)
#         return new_ids
    

#     @staticmethod
#     def get_bounding_box(ground_truth_map):
#         # get bounding box from mask
#         y_indices, x_indices = np.where(ground_truth_map > 0)
#         x_min, x_max = np.min(x_indices), np.max(x_indices)
#         y_min, y_max = np.min(y_indices), np.max(y_indices)
#         # add perturbation to bounding box coordinates
#         H, W = ground_truth_map.shape
#         x_min = max(0, x_min - np.random.randint(0, 20))
#         x_max = min(W, x_max + np.random.randint(0, 20))
#         y_min = max(0, y_min - np.random.randint(0, 20))
#         y_max = min(H, y_max + np.random.randint(0, 20))
#         bbox = [x_min, y_min, x_max, y_max]

#         return bbox
    
#     @staticmethod
#     def get_random_prompt(ground_truth_map,bbox):
#         x_min, y_min, x_max, y_max = bbox
#         while True:
#             # Generate random point within the bounding box
#             x = np.random.randint(x_min, x_max)
#             y = np.random.randint(y_min, y_max)
#             # Check if the point lies inside the mask
#             if ground_truth_map[y, x] == 1:
#                 return x, y

#     @property
#     def classes(self):
#         """Category names."""
#         return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
#                 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#                 'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
#                 'tv')