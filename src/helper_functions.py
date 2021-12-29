import cv2
import numpy as np

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