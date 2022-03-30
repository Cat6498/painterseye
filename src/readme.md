# Readme

The painter is implemented entirely on Python and Pytorch. The structure of the src folder is as follow:

* brushes -> contains png heightmap of brush strokes to provide a texture to the rendered brushstrokes
* checkpoints -> stores the networks checkpoints
    * GAN_model -> strores SalGAN's checkpoint   
    * other folders are generated at runtime to hold the other checkpoints
* input -> the input image goes here. By default, contains the image of some people on a boat. 
* maps -> the generated painting maps get stored here, in a folder with the image name
* output -> the final result will be found here
* saliency -> the generated saliency map is stored here
* segments -> a folder for the binary masks of the objects
* style_images -> the style image for style transfer goes here. By default, contains a Monet painting.
* helper_functions.py -> contains helper functions for the painter (to get the background color and resize the image)
* loss.py -> from Zou et al (2021), contains implementation of loss functions (L1, style loss)
* map_generation.py -> contains functions to perform panoptic segmentation on the image, obtain saliency map, and generate painting maps
* morphology.py -> from Zou et al (2021), contains functions to edit brushstroke parameters
* networks.py -> from Zou et al (2021), contains network implementation
* paint.py -> contains function to distribute weights, paint functions to paint and optimize brushstrokes (including style transfer), and to set painter's arguments
* painter.py -> from Zou et al (2021) - edited, contains Painter class, that contains the network and samples, concatenates, and predicts brushstrokes
* python_batch_sinkhorn.py -> from Zou et al (2021), contains python implementation of Sinkhorn loss
* renderer.py -> from Zou et al (2021), contains the Renderer class, used to sample brushstroke parameters and render them on the canvas in the style required
* utils.py -> from Zou et al (2021), contains useful functions, such as generating training datasets
* vggnet.py -> contains pythorch implementation of SalGAN (from [this GitHub](https://github.com/niujinshuchong/SalGan_pytorch/blob/master/vggnet.py))

### Requirements and build instructions

* Python 3.7
* Packages: listed in `requirements.txt` 
* pyyaml 5.1
* git+https://github.com/cocodataset/panopticapi.git
* numpy
* detectron2, with your TORCH_VERSION and CUDA_VERSION
    * !pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html

NOTE: the painter needs over 10GB of GPU to run, I could not run it on my computer. I created a Colaboratory notebook, and it is the easiest way to run the painter. The notebook is available at [link the notebook], and a copy is provided inside this folder. Opening it through the link, it will guide you step by step through packages installation and checkpoints download, tell you how to load your image and let you customise the painting process. Note however that due to high GPU requirement I need a Colab Pro subscription to run the painter.

## Build instructions

The general instructions to run the painter are:

* intall required packages
* download the checkpoint from [add link] and store it in checkpoint/GAN_model
* download your preferred style checkpoint from [add link] and store it in checkpoints/checkpoint_G_style/, replacing 'style' with the style you chose (eg oilpaintbrush)
* upload the image you want to paint in input/
* if you want to perform style transfer, add your style image in style_images/
* run the painter



