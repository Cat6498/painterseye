import argparse
import cv2
import numpy as np
import torch
import torch.optim as optim
from painter import *
from helper_functions import get_background_color


def get_weights(map, pt, method="equal", map_type="final", s=128):
  weights = []
  total_map = np.sum(map)
  total_map_nonzero = cv2.countNonZero(map)

  # Revert to baseline approach if the maps are empty (or almost)
  if total_map < 0.1:
    weight = round(pt.max_m_strokes/pt.m_grid**2)
    for i in range(pt.m_grid**2):
      weights.append(weight)

  # Equal weight distribution
  elif method == "equal":
    foreground = []
    background = []
    patch_id = 0
    threshold = 0.5

    # Iterate throgh each patch
    for y_id in range(pt.m_grid):
      for x_id in range(pt.m_grid):
        patch = map[y_id * s:y_id * s + s, x_id * s:x_id * s + s]
        
        if map_type == "final":
          weight = cv2.countNonZero(patch)/total_map_nonzero
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
          
    back *= pt.max_m_strokes
    fore *= pt.max_m_strokes
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
        if weight < 0.01:
          weight = 0.01
        weight = round(weight*pt.max_m_strokes)
        weights.append(weight)

  # Finally, get the total number of strokes and the maximum number of strokes in one patch
  # Append 0 at the start for indexing purposes (a sort of "head" pointer)
  total_strokes_layer = sum(weights)
  max_in_patch = max(weights)
  weights.insert(0,0)

  return weights, total_strokes_layer, max_in_patch


# PAINTER - intrinsic and explicit style transfer

def set_painter_args(inp_img, name, max_n_strokes, style_trans=False, sty_img="", style="oilpaintbrush"):
    parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
    args = parser.parse_args(args=[])
    args.img_path = inp_img # path to input photo
    args.style_transfer = style_trans
    if style_trans:
      args.style_img_path = sty_img
      args.transfer_mode = 1
    args.renderer = style # [watercolor, markerpen, oilpaintbrush, rectangle]
    args.canvas_color = get_background_color(inp_img) # [black, white]
    args.canvas_size = 512 # size of the canvas for stroke rendering'
    args.keep_aspect_ratio = True # keep original image size
    args.max_m_strokes = max_n_strokes # max number of strokes
    args.m_grid = 5 # number of grid divisions to perform
    args.beta_L1 = 1.0 # weight for L1 loss
    args.with_ot_loss = True # also use transportation loss
    args.beta_ot = 0.1 # weight for optimal transportation loss
    args.with_sty_loss = style_trans
    args.beta_sty = 0.05
    args.net_G = 'zou-fusion-net' # renderer architecture
    args.renderer_checkpoint_dir = './checkpoints/checkpoints_G_' + style # dir to load the pretrained neu-renderer
    args.lr = 0.005 # learning rate for stroke searching
    args.output_dir = './output/' + name # dir to save painting results
    args.disable_preview = True # disable cv2.imshow
    return args



def paint(pt, name, method="equal", map_type="final"): 
    # Load thee checkpoint for the neural renderer
    pt._load_checkpoint()
    pt.net_G.eval()

    if not os.path.exists(pt.output_dir):
        os.makedirs(pt.output_dir)

    print('begin drawing...')

    s = 128 # size of output network

    # Pick the map, load it and resize it
    if map_type == "final":
      map_path = './maps/' + name + '/final.jpg'
    else:
      map_path = './maps/' + name + '/finalblend.jpg'

    map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    map = cv2.resize(map, (pt.m_grid * s, pt.m_grid * s))
    
    # Get the weigths, the total number of strokes in the layer, and the maximum number of stroke for a patch
    weights, total_strokes, max_strokes = get_weights(map, pt, method, map_type, s)
    pt.initialize_params(total_strokes)
      
    pt.x_ctt.requires_grad = True
    pt.x_color.requires_grad = True
    pt.x_alpha.requires_grad = True
    utils.set_requires_grad(pt.net_G, False) # The renderer is already trained

    # Define optimizer
    centered = False if pt.args.style_transfer else True
    pt.optimizer_x = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha], lr=pt.lr, centered=centered)

    pt.step_id = 0

    # for each stroke round
    for pt.anchor_id in range(0, max_strokes):
        # sample the action vectors
        pt.stroke_sampler(weights)
            
        # how many iterations each stroke gets (optimisation)
        iters_per_stroke = int(total_strokes/max_strokes)

        for i in range(iters_per_stroke):

            pt.optimizer_x.zero_grad()

            pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
            pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
            pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

            if pt.canvas_color == 'white':
                pt.G_pred_canvas = torch.ones(
                    [pt.m_grid ** 2, 3, pt.net_G.out_size, pt.net_G.out_size]).to(device)
            else:
                pt.G_pred_canvas = torch.zeros(
                    [pt.m_grid ** 2, 3, pt.net_G.out_size, pt.net_G.out_size]).to(device)

            pt._forward_pass(weights)
            pt._drawing_step_states(max_strokes)
            pt._backward_x()
            pt.optimizer_x.step()

            pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
            pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
            pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

            pt.step_id += 1

    v = pt.x.detach().cpu().numpy()
    pt._save_stroke_params(v)
    v_n = pt._normalize_strokes(pt.x, weights)
    v_n = pt._shuffle_strokes_and_reshape(v_n)
    final_rendered_image = pt._render(v_n, weights, save_jpgs=True, save_video=True)
    return weights


# STYLE TRANSFER - on precomputed brushstrokes

def set_nst_args(inp_img, name, sty_img, vec_path, trans_mode=2, style="oilpaintbrush"):
    parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
    args = parser.parse_args(args=[])
    args.content_img_path = inp_img # path to input photo
    args.style_img_path = sty_img
    args.vector_file = vec_path
    args.transfer_mode = trans_mode
    args.renderer = style # [watercolor, markerpen, oilpaintbrush, rectangle]
    args.canvas_color = get_background_color(inp_img) # [black, white]
    args.canvas_size = 512 # size of the canvas for stroke rendering'
    args.keep_aspect_ratio = True # keep original image size

    args.m_grid = 5 # number of grid divisions to perform

    args.beta_L1 = 1.0 # weight for L1 loss
    args.beta_sty = 0.5 # weight for style loss
    args.net_G = 'zou-fusion-net' # renderer architecture
    args.renderer_checkpoint_dir = './checkpoints/checkpoints_G_' + style # dir to load the pretrained neu-renderer
    args.lr = 0.005 # learning rate for stroke searching
    args.output_dir = './output/' + name + "/style_trans" # dir to save painting results
    args.disable_preview = True # disable cv2.imshow
    return args



def style_transfer(pt, weights, trans_mode=2):

    pt._load_checkpoint()
    pt.net_G.eval()

    if not os.path.exists(pt.output_dir):
        os.makedirs(pt.output_dir)

    if trans_mode == 1: # transfer color only
        pt.x_ctt.requires_grad = False
        pt.x_color.requires_grad = True
        pt.x_alpha.requires_grad = False
    else: # transfer both color and texture
        pt.x_ctt.requires_grad = True
        pt.x_color.requires_grad = True
        pt.x_alpha.requires_grad = True

    pt.optimizer_x_sty = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha], lr=pt.lr)

    iters_per_stroke = 100
    for i in range(iters_per_stroke):
        pt.optimizer_x_sty.zero_grad()

        pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
        pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
        pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

        if pt.canvas_color == 'white':
            pt.G_pred_canvas = torch.ones([pt.m_grid*pt.m_grid, 3, pt.net_G.out_size, pt.net_G.out_size]).to(device)
        else:
            pt.G_pred_canvas = torch.zeros(pt.m_grid*pt.m_grid, 3, pt.net_G.out_size, pt.net_G.out_size).to(device)

        pt._forward_pass(weights)
        pt._style_transfer_step_states()
        pt._backward_x_sty()
        pt.optimizer_x_sty.step()

        pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
        pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
        pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

        pt.step_id += 1

    print('saving style transfer result...')
    v_n = pt._normalize_strokes(pt.x, weights)
    v_n = pt._shuffle_strokes_and_reshape(v_n)
    final_rendered_image = pt._render(v_n, weights, save_jpgs=False, save_video=False)
    pt._save_style_transfer_images(final_rendered_image)
