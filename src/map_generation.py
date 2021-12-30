import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
import glob
import tempfile
import time
import warnings
import json

import cv2

# Segmentation
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config

# Saliency
from vggnet import SalGan, image_preprocess, post_process

def semantic_seg(inp_img):
  print("Segmenting the image...")
  model = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file(model))
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
  add_panoptic_deeplab_config(cfg)
  
  image = cv2.imread(inp_img)
  predictor = DefaultPredictor(cfg)
  panoptic_matrix, segments_info = predictor(image)["panoptic_seg"]

  panoptic_array = np.array(panoptic_matrix.cpu())
  panoptic_img = np.stack([panoptic_array, panoptic_array, panoptic_array])
  panoptic_img = np.einsum("ijk->jki", panoptic_img)
              
  for i in range(len(segments_info)):
    mask = np.zeros_like(panoptic_img)
    mask[panoptic_img==float(segments_info[i]["id"])] = 1
    path = os.path.join("segments", str(i) + ".png")
    cv2.imwrite(path, mask*255)
  
  print("Segmentation done!", i, "objects identified")



def sal_map_generator(inp_img):
  print("Generating saliency map...")
  salGAN = SalGan()
  salGAN.load_state_dict(torch.load('checkpoints/GAN_model/gan_torch_model.pkl'))
  
  image = cv2.imread(inp_img, cv2.IMREAD_COLOR)

  input_image = image_preprocess(image)
  input_image = torch.autograd.Variable(input_image.unsqueeze(0))   
  result = salGAN(input_image)
  saliency_map = post_process(result.data.numpy()[0, 0], image.shape[0], image.shape[1])
  
  result_dir = 'saliency/'
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)

  cv2.imwrite(os.path.join(result_dir, inp_img.split("/")[1]), saliency_map)
  print("Saliency map done!")



def get_maps(inp_img):
  print("Generating painting maps...")
  title = "saliency/" + inp_img.split("/")[1]
  sal_map = cv2.imread(title, cv2.IMREAD_GRAYSCALE)
  predictions = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob("segments/*.png")]

  final_objs = np.zeros_like(sal_map)
  final_blend = np.zeros_like(sal_map)

  print("Mean saliency of objects:")
  for img_object in predictions:
    saliency_obj = sal_map.copy()
    obj_mean = np.mean(saliency_obj[img_object!=0]/np.max(saliency_obj))
    print(obj_mean)

    if (obj_mean > 0.1):
      final_objs += img_object
      saliency_obj[img_object==0] = 0
      final_blend += saliency_obj

  result_dir = 'maps/'
  cv2.imwrite(os.path.join(result_dir, "finalblend.jpg"), final_blend)
  cv2.imwrite(os.path.join(result_dir, "final.jpg"), final_objs)
  print("Painting maps done!")
