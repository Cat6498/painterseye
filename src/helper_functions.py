import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

def resize_inp_img(inp_img):
  image = cv2.imread(inp_img)
  h,w,c = image.shape
  max_size = 512
  smaller_size = h if h < w else w
  if smaller_size <= max_size:
    return
  factor = smaller_size/max_size
  resized_img = cv2.resize(image, None, fx=1/factor, fy=1/factor)
  cv2.imwrite(inp_img, resized_img)

def get_background_color(inp_img):
  image = cv2.imread(inp_img, cv2.IMREAD_GRAYSCALE)
  if np.mean(image) <= 127:
    return "white"
  else:
    return "black"

def evaluate_result(inp_img, res_name, baseline, picture_name, weights, use_map, ext_G, ext_psnr, trans_type=0, method="direct"):
  target = cv2.imread(inp_img, cv2.IMREAD_COLOR)
  result = cv2.imread(res_name, cv2.IMREAD_COLOR)
  target = cv2.resize(target, (result.shape[1], result.shape[0]))

  if trans_type != 2: 
    mse = np.mean((result - target)**2)
    mrse = np.sqrt(mse)

    l1 = np.mean(np.abs(result - target))

    psnr = 20 * np.log10(255.0/mrse)

    ssim = structural_similarity(result, target, data_range=target.max() - target.min(), multichannel=True)

    if trans_type == 0: # normal painting
      data = pd.DataFrame({"image name":[picture_name], "baseline": [baseline], "weights":[weights], "map":[use_map], "ext G":[ext_G], "ext psnr":[ext_psnr],
              "mse":[mse], "mrse":[mrse], "l1":[l1], "psnr":[psnr], "ssim":[ssim]})
      data.to_csv("./results.csv", index=False, mode='a', header=None)

    elif trans_type == 1: # style transfer
      data = pd.DataFrame({"image name":[picture_name], "baseline": [baseline], "method":[method], "weights":[weights], "map":[use_map], "ext G":[ext_G], "ext psnr":[ext_psnr],
              "mse":[mse], "mrse":[mrse], "l1":[l1], "psnr":[psnr], "ssim":[ssim]})
      data.to_csv("./results_style.csv", index=False, mode='a', header=None)
    
  else: # only on masked objects
    map_name = "maps/" + picture_name + "/" + use_map + ".jpg"
    mapb = cv2.imread(map_name, cv2.IMREAD_GRAYSCALE)
    mapb = cv2.resize(mapb, (result.shape[1], result.shape[0]))

    mse = np.mean((result[mapb != 0] - target[mapb != 0])**2)
    mrse = np.sqrt(mse)

    l1 = np.mean(np.abs(result[mapb != 0] - target[mapb != 0]))

    psnr = 20 * np.log10(255.0/mrse)

    ssim = structural_similarity(result[mapb != 0], target[mapb != 0], data_range=target.max() - target.min(), multichannel=True)

    data = pd.DataFrame({"image name":[picture_name], "baseline": [baseline], "weights":[weights], "map":[use_map], "ext G":[ext_G], "ext psnr":[ext_psnr],
              "mse":[mse], "mrse":[mrse], "l1":[l1], "psnr":[psnr], "ssim":[ssim]})
    data.to_csv("./results_objects.csv", index=False, mode='a', header=None)
