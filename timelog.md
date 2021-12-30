# Timelog

* The Painter's eye
* Caterina Mongiello
* 2404262m
* Nicolas Pugeault

## Guidance

* This file contains the time log for your project. It will be submitted along with your final dissertation.
* **YOU MUST KEEP THIS UP TO DATE AND UNDER VERSION CONTROL.**
* This timelog should be filled out honestly, regularly (daily) and accurately. It is for *your* benefit.
* Follow the structure provided, grouping time by weeks.  Quantise time to the half hour.

## Week 1

### 28 Sep 2021

* *2 hours* Read the project guidance notes, background reesearch and brainstorm  

### 29 Sep 2021

* *30 mins* meeting with supervisor

<br />

## Week 2

### 02 Oct 21 

* *3 hours* Reading and research on AI painting

### 05 Oct 21 

* *3 hours* Reading and research on AI painting

### 06 Oct 21 

* *30 mins* Second meeting with Nicolas

<br />

## Week 3

### 09 Oct 21 

* *3 hours* Research on Neural networks (in particular GANs)

### 12 Oct 21 

* *4 hours* Research on painting with brushstrokes neural renderers and painting agents

### 13 Oct 21 

* *30 mins* Third meeting with Nicolas

<br />

## Week 4

### 16 Oct 21 

* *3 hours* Research on Neural networks (attention)

### 19 Oct 21

* *4 hours* Research on attention, saliency maps and semantinc segmentation

### 20 Oct 21

* *30 mins* Fourth meeting with Nicolas

<br />

## Week 5

### 23 Oct 21

* *3 hours* Neural rendered first experiment from Nakano and LibreAI

### 26 Oct 21

* *4 hours* Neural renderer experiments, completed generator training, set up GitHub and Colab 

### 27 Oct 21

* *30 mins* Fifth meeting with Nicolas

<br />

## Week 6

### 30 Oct 21

* *4 hours* Neural renderer experiments from Zou

### 2 Nov 21

* *5 hours* Continued neural renderer experiments, successfully trained for 30 epochs c:

### 3 Nov 21

* *30 mins* Sixth meeting with Nicolas

<br />

## Week 7

### 6 Nov 21

* *5 hours* Tried Pyramid Scene Parsing and Analysis for semantic segmentation. Not particularly happy with the results

### 9 Nov 21

* *5 hours* Tried a new approach to semantic segmentation, using Detectron2 and DeepLab by Meta. Results noticeably improved

### 11 Nov 21

* *30 mins* Seventh meeting with Nicolas

<br />

## Week 8

### 13 Nov 21

* *5 hours* Got a SalGAN pytorch model to work 

### 16 Nov 21

* *5 hours* Blended saliency map and panoptic segmentation to get the final maps for the painter

### 17 Nov 21

* *30 mins* Eight meeting with Nicolas

<br />

## Week 9

### 20 Nov 21

* *5 hours* In-depth research on baseline painter code

### 23 Nov 21

* *5 hours* Started modifying baseline painter to paint according to "attention" (weight distribution)
* No meeting this week due to jury duty

<br />

## Week 10

### 27 Nov 21

* *5 hours* Continued work on painter

### 30 Nov 21

* *5 hours* Managed to get brushstrokes distribution according to saliency/segmentation map, but a bug is preventing it from painting properly

### 1 Dec 21

* *5 hours* Found the bug and fixed the painter, final project (unpolished) almost ready c:
* *30 mins* Ninth meeting with Nicolas

<br />

## Week 11-12 [project weeks]

* No meeting in week 11 due to other deadlines

### 9 Dec 21

* *5 hours* Research on metrics and methods to use for qualitative and quantitative evaluation

### 10 Dec 21

* *5 hours* More research on metrics and methods for evaluation. Experimented with some images and fixed bugs

### 11 Dec 21

* *5 hours* Continued experimenting, refactored the weight distribution techniques according to maps to differentiate them more

### 13 Dec 21

* *5 hours* Started writing status report, kept working on map generation and weight distribution

### 14 Dec 21

* *5 hours* Identified a set of 25 images with different problems (low light, many details, etc.) representing 5 different categories (humans, animals, objects, environment and mix) for the evaluation. Run the algorithm on one of them to include results in the report, 

### 15 Dec 21

* *30 mins* Tenth meeting with Nicolas, discussed plan forward

### 16 Dec 21

* *1 hour* finished status report

<br />

## Winter Break

### 28 Dec 21

* *5 hours* Changed the panoptic segmentation implementation to the official Detectron2 one from their notebook (less code required and no need to store the model checkpoint), added functions to decide on the color of the background based on the dominant color of the input image and to resize the input image

### 29 Dec 21

* *3 hours* Refactored the code on Colab, added image upload for input and made form for parameters, migrated the project from Google Drive to GitHub

### 30 Dec 21

* *5 hours* Finished refactoring, some bug fixing in the map generation and weight distribution, added function to automatically download and unzip the checkpoints for the models (directly from the original models), did a bit of restructuring of the project folder and worked on git documentation

<br />





