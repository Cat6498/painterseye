import argparse
import cv2
import numpy as np
import torch
import torch.optim as optim
from painter import *
from helper_functions import get_background_color


def get_weights(map, pt, strokes_per_layer, method="equal", map_type="final", s=128):
  weights = []
  total_map = np.sum(map)
  total_map_nonzero = cv2.countNonZero(map)
  if total_map < 0.1:
    weight = round(strokes_per_layer/pt.m_grid**2)
    for i in range(pt.m_grid**2):
      weights.append(weight)

  # Equal weight distribution
  elif method == "equal":
    foreground = []
    background = []
    patch_id = 0

    if map_type == "final":
      threshold = (s*s/2)
    else:
      threshold = 1/(pt.m_grid**2)

    # Iterate throgh each patch
    for y_id in range(pt.m_grid):
      for x_id in range(pt.m_grid):
        patch = map[y_id * s:y_id * s + s, x_id * s:x_id * s + s]
        
        if map_type == "final":
          weight = s*s - cv2.countNonZero(patch)
        else:
          weight = np.sum(patch)/total_map

        if weight < threshold:
          background.append(patch_id)
        else:
          foreground.append(patch_id)
        patch_id += 1

    # Assign percentage of brushstrokes to foreground and background.
    # If there is no background give full importance to foreground (and viceversa)
    fore = 0.7
    back = 0.3
    if (len(foreground) == 0):
      back = 1
    if (len(background) == 0):
      fore = 1
          
    back *= strokes_per_layer
    fore *= strokes_per_layer

    if  (len(foreground) != 0):
      foreground_strokes = round(fore/len(foreground))
    else:
      foreground_strokes = 0
      
    if (len(background) != 0):
      background_strokes = round(back/len(background))
    else:
      background_strokes = 0

    for patch_id in range(pt.m_grid**2):
      if patch_id in foreground:
        weights.append(foreground_strokes)
      else:
        weights.append(background_strokes)

  # Individual weight distribution    
  else:
    # Iterate through patches
    for y_id in range(pt.m_grid):
      for x_id in range(pt.m_grid):
        patch = map[y_id * s:y_id * s + s, x_id * s:x_id * s + s]
        
        if map_type == "final":
          weight = cv2.countNonZero(patch)/total_map_nonzero
        else:
          weight = np.sum(patch)/total_map
        
        # Get the weight for the patch and if less than 5% set it to 5% of the total strokes (to prevent it from leaving empty spaces)
        if weight < 0.05:
          weight = 0.05
        weight = round(weight*strokes_per_layer)
        weights.append(weight)

  # Finally, get the total number of strokes and the maximum number of strokes in one patch
  # Append 0 at the start for indexing purposes (a sort of "head" pointer)
  total_strokes_layer = sum(weights)
  max_in_patch = max(weights)
  weights.insert(0,0)

  return weights, total_strokes_layer, max_in_patch



def set_painter_args(inp_img, max_n_strokes, style="oilpaintbrush"):
    parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
    args = parser.parse_args(args=[])
    args.img_path = inp_img # path to input photo
    args.renderer = style # [watercolor, markerpen, oilpaintbrush, rectangle]
    args.canvas_color = get_background_color(inp_img) # [black, white]
    args.canvas_size = 512 # size of the canvas for stroke rendering'
    args.keep_aspect_ratio = True # keep original image size
    args.max_m_strokes = max_n_strokes # max number of strokes
    args.max_divide = 5 # number of grid divisions to perform
    args.beta_L1 = 1.0 # weight for L1 loss
    args.with_ot_loss = True # also use transportation loss
    args.beta_ot = 0.1 # weight for optimal transportation loss
    args.net_G = 'zou-fusion-net' # renderer architecture
    args.renderer_checkpoint_dir = './checkpoints/checkpoints_G_' + style # dir to load the pretrained neu-renderer
    args.lr = 0.005 # learning rate for stroke searching
    args.output_dir = './output' # dir to save painting results
    args.disable_preview = True # disable cv2.imshow
    return args


def optimize_x(pt, method="equal", map_type="final"):
    # Load thee checkpoint for the neural renderer
    pt._load_checkpoint()
    pt.net_G.eval()

    print('begin drawing...')

    # Pick the map, load it and resize it
    if map_type == "final":
      map_path = './maps/final.jpg'
    else:
      map_path = './maps/finalblend.jpg'

    map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    #input_aspect_ratio = map.shape[0] / map.shape[1]
    map = cv2.resize(map, (128 * pt.max_divide, 128 * pt.max_divide), cv2.INTER_AREA)
    
    #Set params to an array of shape (1, 0, shape_actions)
    PARAMS = np.zeros([1, 0, pt.rderr.d], np.float32)

    # Create a starting empty canvas
    if pt.rderr.canvas_color == 'white':
        CANVAS_tmp = torch.ones([1, 3, 128, 128]).to(device)
    else:
        CANVAS_tmp = torch.zeros([1, 3, 128, 128]).to(device)
    
    # Divide the painting in batches of size m_grid**2, 3, s, s (so 1, 4, 9, ...)
    for pt.m_grid in range(1, pt.max_divide + 1):
        pt.img_batch = utils.img2patches(pt.img_, pt.m_grid, pt.net_G.out_size).to(device)
        image_batches = {i: pt.img_batch[i] for i in range(pt.m_grid**2)}
        
        strokes_per_layer = pt.m_strokes_per_block*pt.m_grid**2
        s = 128
        map = cv2.resize(map, (pt.m_grid * s, pt.m_grid * s))
        
        # Get the weigths, the total number of strokes in the layer, and the maximum number of stroke for a patch
        weights, total_strokes_layer, max_strokes = get_weights(map, pt, strokes_per_layer, method, map_type, s)

        pt.G_final_pred_canvas = CANVAS_tmp

        # Initialize parameters to empty tensors of the correct size to hold the brushstrokes
        pt.initialize_params(total_strokes_layer)
        pt.x_ctt.requires_grad = True
        pt.x_color.requires_grad = True
        pt.x_alpha.requires_grad = True
        utils.set_requires_grad(pt.net_G, False) # The renderer is already trained

        # Define optimizer
        pt.optimizer_x = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha], lr=pt.lr, centered=True)

        pt.step_id = 0

        # for each stroke set (1-per-patch)
        for pt.anchor_id in range(0, max_strokes):
            # sample the action vectors
            pt.stroke_sampler(image_batches, weights)
            
            # get how many iterations each stroke gets (optimisation)
            iters_per_stroke = int(500 / pt.m_strokes_per_block)
            # optimise the action vectors
            for stroke_iter in range(iters_per_stroke):
                pt.G_pred_canvas = CANVAS_tmp

                pt.optimizer_x.zero_grad()

                pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
                pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
                pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

                # use the action vectors to predict strokes
                pt._forward_pass(weights)
                
                # render the strokes
                pt._drawing_step_states(max_strokes)
                # calculate the loss and backpropagate
                pt._backward_x()

                pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
                pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
                pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

                # optimise parameters
                pt.optimizer_x.step()
                pt.step_id += 1

        # after all strokes on patches are done, get them together and reshape them 
        strokes = pt._normalize_strokes(pt.x, weights)
        strokes = pt._shuffle_strokes_and_reshape(strokes)
        # concatenate to the whole parameter list
        PARAMS = np.concatenate([PARAMS, strokes], axis=1)
        # render on the initial canvas
        CANVAS_tmp = pt._render(PARAMS, save_jpgs=False, save_video=False)
        # divide it into patches to use it as a base in the next patches
        CANVAS_tmp = utils.img2patches(CANVAS_tmp, pt.m_grid + 1, pt.net_G.out_size).to(device)

    pt._save_stroke_params(PARAMS)
    # in the end render final image with all parameters
    final_rendered_image = pt._render(PARAMS, save_jpgs=False, save_video=True)

    return final_rendered_image