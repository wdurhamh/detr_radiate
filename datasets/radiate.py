import os
import json
from pathlib import Path

import torch
import torch.utils.data
from torchvision.datasets import VisionDataset
from PIL import Image
from pycocotools.coco import COCO
import numpy as np

from .coco import ConvertCocoPolysToMask
from .coco import make_coco_transforms

#from detectron2.structures import BoxMode

"""
Coppied from CocoDectection class definition (https://pytorch.org/vision/stable/_modules/torchvision/datasets/coco.html).
Modified constructor to allow building dataset from dictionary rather than json file.
"""
class Radiate(VisionDataset):

    def __init__(self, root, radar_dicts, transform = None, target_transform = None, transforms = None):
        #we're storing the complete path to each image (for now), so let root be empty
        self.root = root
        self._transforms = transforms
        super().__init__(self.root, transforms, transform, target_transform)
        self.dataset=radar_dicts
        self.prepare = ConvertCocoPolysToMask(False)

    def _load_image(self, index):
        path = self.dataset[index]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def __getitem__(self, index):
        img = self._load_image(index)
        target = self.dataset[index]
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


    def __len__(self):
        return len(self.dataset)

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

    return min_x, min_y, max_x, max_y

def get_radar_dicts(root_dir, folders, rrpn=False):
    idd = 0
    folder_size = len(folders)
    dataset_dicts = []
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
            record["file_name"] = filename
            record["image_id"] = idd
            record["height"] = 1152
            record["width"] = 1152

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

                        objs.append(obj)
            if bb_created:
                record["annotations"] = objs
                dataset_dicts.append(record)
    return dataset_dicts

def radiate_to_coco(root_dir, folders, rrpn=False):

    license_dicts = [{'url':'https://creativecommons.org/licenses/by-nc-sa/4.0/', 'id':1,
                     'name':'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License'}]
    image_dicts = []
    #For vehcile detection, just a single category
    category_dicts = [{'supercategory':'vehicle', 'id':0, 'name':vehicle}]
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

def build(folders, image_set, rrpn=False):
    #build absolute path to root
    detr_root = Path(get_radar_dicts.__globals__['__file__']).parent.parent
    root_dir = os.path.join( detr_root, 'data')
    #annotations is a list of dictionary objects
    radar_dicts = get_radar_dicts(root_dir, folders, rrpn)
    #print(radar_dicts)
    return Radiate(root_dir, radar_dicts, transforms = make_coco_transforms(image_set))
