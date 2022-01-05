# The Painter's Eye

## Introduction

With the recent advances in the fields of Neural Networks and Computer Vision, there has been a rising interest in making AIs generate paintings. However, humans do not perceive or create a painting pixel per pixel. We focus on objects and details, on the things that call out for us from the canvas.
The aim of this research project is to teach an artificial intelligence to paint as a real painter would. Using semantic segmentation to identify the shapes in an image and saliency maps to get the most attracting points, I will try to model an attention mechanism that will drive the painting process of a neural renderer, making it focus more on the most relevant objects.

## Structure

The final painter is composed by three main neural networks (one for segmentation, one for saliency and one for painting) and some helper functions to process the image. The following diagram illustrates the high-level structure of the software from input to output.

### 1. Resizing the image

To prevent Colab's GPUs from running out of memory while painting, the image is first passed through a function that resizes it. This function maintains the height-width ratio of the original image, and scales it down so that the smallest side is 512.

### 2. Detectron2

Detectron2 is a powerful AI library by Meta Research that provides algorithms for object detection and segmentation in pictures and videos. Developed in PyTorch, the implementation can be found in their [GitHub page](https://github.com/facebookresearch/detectron2). 

The `semantic_seg()` function in `map_generation.py` relies on Detectron2 to perform panoptic segmentation of the input image. Panoptic segmentation blends instance detection with semantic segmentation: it tries to divide the pixels in the image based on the class to which they belong (semantic segmentation), while also identifying instances of the classes (instance detection). First, the function downloads the configuration file from the Detectron's model zoo, and it uses it to download the network checkpoint and to instantiate the predictor. Then, it loads the input image and processes it with the predictor, obtaining the `panoptic_matrix` and the `segments_info`. The latter is a list with info on all the objects identified, while the former is a matrix of size height\*width of the input image, and it stores the id of each object in the positiong where the object is in the image (see picture). 
The function creates a panoptic image by stacking three panoptic matrices along a new dimension (to act like the color channels). Then, for each object in the `segments_info` list, it creates a binary mask of the object, creating a copy of the panoptic image filled with zeros, and setting it to one where the panoptic image matches the object id. Finally, it stores the segments in the `segments` folder.

### 3. SalGAN

SalGAN is deep convolutional neural network for visual saliency prediction, trained in adversarial fashion. Given an input image, it produces a prediction of the image's saliency map, defined as the most important parts of the image (the places where the eyes of an observer are most likely to focus on). 

The `sal_map_generator()` function in `map_generation.py` relies on the [PyTorch implementation of SalGAN](https://github.com/niujinshuchong/SalGan_pytorch). It loads the network checkpoint, then loads the input image and feeds it to the network to get the saliency map. Lastly, it stores the result in the `saliency` folder. 

### 4. Map Generation

The map generation function creates two different maps to drive the painting process using the saliency map and the binary masks of the objects. It takes the mean of the pixels that compose each object (according to the binary masks) in the saliency map, obtaining the mean saliency of each object. If an object's mean saliency is higher than 0.1, then the object is considered important and is included in the final maps. The first of the final maps (called final) is just another binary mask including all the objects that were found to be important. The second map (blend) is the saliency map shaped according to the final map (set to 0 where the final binary mask is 0). 

### 5. Painter

*Coming soon...*

## References

*Note: this section only includes the main references. The final dissertation will include all of them*

Nakano, R., 2019. Neural Painters: A learned differentiable constraint for generating brushstroke paintings. arXiv:1904.08410.

> Nakano, R., Neural renderer [GitHub](https://github.com/reiinakano/neural-painters-pytorch/tree/master/neural_painters)

> LibreAI neural renderer [GitHub](https://github.com/libreai/neural-painters-x)

Zou, Z., Shi, T., Qiu, S., Yuan, Y., Michigan, Z.S.U. of, Arbor, A., Lab, N.F.A., University, B., 2021. Stylized Neural Painting, in: CVPR.

> Zou, Z., Neural renderer [GitHub](https://github.com/jiupinjia/stylized-neural-painting)

Coming soon...

## Project-related material

Link to the dissertation material folder on google drive (includes meeting minutes, project briefs, and research material):
[Dissertation Material](https://drive.google.com/drive/folders/1G2O_FanmPbNt1FlOE2I4gvPyI7F_sztO?usp=sharing)

## Run the program

The painting program is available to run on Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Knl2XOdvjRWYU1qFSDlSq8Zwh-ifL1fA?usp=sharing)

It will first make you to clone this git repository and install detectron and other packages. It will make you download all the relevant checkpoints and upload images to paint. Then, you will be able to set all the parameters and run the painting process. If you do not want to input an image, a default one is already available in the input folder (`more-points.jpg`). The average time to paint an image is around 15-20 minutes, depending on the amount of brushstrokes and the weight distribution (usually faster with `equal` weights).

The program is not available to run on local machine yet as it requires GPU support. 
