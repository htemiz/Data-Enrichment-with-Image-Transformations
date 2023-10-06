Repository for supporting files and outcomes for my paper entitled
[Effects of Data Enrichment with Image Transformations on the Performance of Deep Networks](https://journals.orclever.com/ejrnd/article/view/23/17) and published in
[The European Journal of Research and Development](https://journals.orclever.com/ejrnd/article/view/23/17).

Please cite the paper as follows:

*Temiz, H. (2022). Effects of Data Enrichment with 
Image Transformations on the Performance of Deep Networks.The European Journal of Research and Development,2(2), 23–33*
[https://doi.org/10.56038/ejrnd.v2i2.23](https://doi.org/10.56038/ejrnd.v2i2.23)

&nbsp;

## Overview

Images  cannot  always  be  expected  to  come  in  a  certain  standard  format  and  orientation.
Deep  networks need  to  be  trained  to  take  into  account  unexpected  variations  in  orientation  or  format.
The contribution of data augmentation with image transformations to the performance of deep networks in 
the super resolution problem were examined.


## Image Transormations

The following image transformations were examined in the study
![](images/transformations.png)

&nbsp;


## Models
The following two modified deep learning models were used: [DECUSR](https://github.com/htemiz/DECUSR) and [SRCNN]() 

### DECUSR with 3 Repeating Blocks

The model revised to have 3 repeating blocks and ability to process 3-channel images.

Visualization of the model's architecture:

![](images/decusr3rb.png)

&nbsp;


### SRCNN

SRCNN was also modified to have ability to process 3-channel images, as it was originally designed to process 1-channel
images.

The architecture of modified SRCNN:

![](images/srcnn.png)

&nbsp;

## Visual Outputs
Visual outputs of data augmentation with each image transformations are given below. 
The values below the image patches belong to PSNR/SSIM. 

![](images/visual_examples.png)

Data augmentation with all transformations ensure the best performance of the models.
Augmentation with 180 degrees rotation provides the highest performance among single 
transformations.


&nbsp;


Please feel free to contact me at [htemiz@artvin.edu.tr](mailto:htemiz@artvin.edu.tr) for any further information.