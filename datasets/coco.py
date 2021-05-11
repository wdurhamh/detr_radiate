# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import json
import os
import numpy as np

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def radiate_to_coco(root_dir, folders, rrpn=False):

    license_dicts = [{'url':'https://creativecommons.org/licenses/by-nc-sa/4.0/', 'id':1,
                     'name':'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License'}]
    image_dicts = []
    #For vehcile detection, just a single category
    category_dicts = [{'supercategory':'vehicle', 'id':0, 'name':'vehicle'}]
    annotation_dicts = []

    idd = 0
    an_id = 0
    folder_size = len(folders)

    for folder in folders:
        radar_folder = os.path.join(root_dir, folder, 'Navtech_Cartesian')
        annotation_path = os.path.join(root_dir,
                                       folder, 'annotations', 'annotations.json')
        with open(annotation_path, 'r') as f_annotation:
            annotation = json.load(f_annotation)

        radar_files = os.listdir(radar_folder)
        radar_files.sort()

        for frame_number in range(len(radar_files)):
            record = {}
            objs = []
            bb_created = False
            idd += 1
            filename = os.path.join(
                radar_folder, radar_files[frame_number])

            if (not os.path.isfile(filename)):
                print(filename)
                continue
            record["license"] = 1
            record["file_name"] = filename
            record["id"] = idd
            record["height"] = 1152
            record["width"] = 1152

            image_dicts.append(record)

            for object in annotation:
                if (object['bboxes'][frame_number]):
                    class_obj = object['class_name']
                    if (class_obj != 'pedestrian' and class_obj != 'group_of_pedestrians'):
                        bbox = object['bboxes'][frame_number]['position']
                        angle = object['bboxes'][frame_number]['rotation']
                        bb_created = True
                        if rrpn:
                            cx = bbox[0] + bbox[2] / 2
                            cy = bbox[1] + bbox[3] / 2
                            wid = bbox[2]
                            hei = bbox[3]
                            obj = {
                                "bbox": [cx, cy, wid, hei, angle],
                                #"bbox_mode": BoxMode.XYWHA_ABS,
                                "category_id": 0,
                                "iscrowd": 0,
                                "area" : wid*hei
                            }
                        else:
                            xmin, ymin, xmax, ymax = gen_boundingbox(
                                bbox, angle)
                            obj = {
                                "bbox": [xmin, ymin, xmax, ymax],
                                #"bbox_mode": BoxMode.XYXY_ABS,
                                "category_id": 0,
                                "iscrowd": 0,
                                "area": (xmax-xmin)*(ymax-ymin)
                            }
                        obj["image_id"] = idd
                        obj["id"] = an_id
                        an_id += 1 

                        annotation_dicts.append(obj)
    return {"licenses":license_dicts, "images":image_dicts, "annotations":annotation_dicts,
            "categories":category_dicts}

def gen_boundingbox(bbox, angle):
    theta = np.deg2rad(-angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    points = np.array([[bbox[0], bbox[1]],
                       [bbox[0] + bbox[2], bbox[1]],
                       [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                       [bbox[0], bbox[1] + bbox[3]]]).T

    cx = bbox[0] + bbox[2] / 2
    cy = bbox[1] + bbox[3] / 2
    T = np.array([[cx], [cy]])

    points = points - T
    points = np.matmul(R, points) + T
    points = points.astype(int)

    min_x = np.min(points[0, :])
    min_y = np.min(points[1, :])
    max_x = np.max(points[0, :])
    max_y = np.max(points[1, :])

    #cast to standard ints to allow for json serialization
    return int(min_x), int(min_y), int(max_x), int(max_y)

def build(image_set, args):
    root = Path(args.coco_path).absolute()
    assert root.exists(), f'provided COCO path {root} does not exist'

    if args.dataset_file == 'radiate':
        img_folder = root / image_set
        folders=[]
        #TODO - don't hard code this
        if image_set == 'train':
            folders = ['city_1_0', 'city_1_1']
        else:
            folders = ['city_1_3']
        #TODO - use distinct test/val sets
        folders=['tiny_foggy']
        json_dict = radiate_to_coco(img_folder, folders)
        #save as a file so we can then read it in 
        ann_file = img_folder / 'coco_annotations.json'
        with open(ann_file, 'w') as outfile:
            json.dump(json_dict, outfile)
    else:
        mode = 'instances'
        PATHS = {
            "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        }

        img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
