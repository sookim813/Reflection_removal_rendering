# Reflection_removal_rendering

Code for   
**Single Image Reflection Removal with Physically-Based Training Images (CVPR 2020 oral)**   
*Soomin Kim, Yuchi Huo, and Sung-Eui Yoon*

<https://sgvr.kaist.ac.kr/~smkim/Reflection_removal_rendering/>


Please cite this paper if you use this code in an academic publication.


```
@InProceedings{Kim_2020_CVPR,
author = {Kim, Soomin and Huo, Yuchi and Yoon, Sung-Eui},
title = {Single Image Reflection Removal With Physically-Based Training Images},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
------------------------------------------------------------------------------------------
This code is based on tensorflow, and has been tested on Ubuntu 16.04 LTS.   

## Setup
* `$ cd Reflection_removal_rendering`
* Create a folder called `VGG_Model`
* Download pre-trained VGG-19 model (`imagenet-vgg-verydeep-19`) in [this page](https://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models) in VGG-VD models category.  
* Move the downloaded pre-trained VGG model(`imagenet-vgg-verydeep-19.mat`) to `VGG_Model` folder


## Testing
* Download [pre-trained model](https://drive.google.com/file/d/1xah0rAzWA2NzsGRiA6bGb78DjeITC_n8/view?usp=sharing)
* `$ tar -xvzf pre-trained.tar.gz`
* Check a newly created folder `pre-trained`, whether downloaded model files are in that folder.
* Example test images are provided in `test_imgs/blended`.
* Run `python main.py`
* Test results are in the `Results` folder.

If you want to try your own test images, then change `input_path` (line 301)in `main.py`. Also, if you don't have ground truth images for test images, then comment out the quality assess part (line 335-336 in `main.py`).

## Acknowledgement
This reflection removal framework is based upon [perceptual-reflection-removal (CVPR 2018)](https://github.com/ceciliavision/perceptual-reflection-removal), which is modified for our proposed structure. 
