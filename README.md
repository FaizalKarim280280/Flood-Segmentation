# Project Title : Flood Segmentation

Flood segmentation is a deep learning model that segments flooded area from aerial images.
<br><br>
![Model Output](https://github.com/FaizalKarim280280/Flood-Segmentation/blob/main/plots/pred.jpg)

# Description
<b>Semantic segmentation</b> is the process of classifying each pixel of an image into distinct classes using deep learning. This aids in identifying regions in an image where certain objects reside.<br/><br/>
The aim of this project is to build a deep learning model which will identify and segment flooded areas in aerial images. Flood segmentation in aerial images is a crucial process in the workflow of target based aerial image analysis. In the last decade, a lot of research has happened related to information extraction from aerial imagery. Still it remains a challenging task due to noise, complex image background and occlusions.

# Contents 
1. Dataset<br/>
2. Tools and libraries<br/>
3. Data Preprcessing<br/>
4. Model Architecture<br/>
5. Training and evaluation<br/>
6. References<br/>

## 1. Dataset
For this project, we have built our own custom dataset from Google Images. This dataset contains aerial images, along with the target masks. The dataset contains `290` aerial images of several flooded and their respective masks. Both the masks and the images are `224x224` in resolution and are present in .png format.<br/><br/>
10% of the dataset was used for validation.

## 2. Tools and libraries
- <b>TensorFlow</b> framework for model building.
- <b>Label Studio</b> for annotation task.
- <b>Albumentations</b> for data augmentation.
- <b>Segmentation Models</b> for defining custom loss function.
- Other secondary libraries mentioned in requirements.txt

## 3. Data Preprocessing
1. Images were reshaped to 224x224 dimension and normalized between [0,1].
2. `tf.data.Dataset` was used for building an efficient data pipeline.

## 4. Model Architecture
1. We have used a standard fully convolutional UNet architecture with backbone model `EfficientNetb2` that receives an RGB colour image as input and generates a same-size semantic segmentation map as output.
2. The model has 4 downsampling blocks and 3 upsampling blocks.  
     - The downsampling section, extracts feature information from the input by decreasing in spatial dimensions but increasing in feature dimensions.
     - The upsampling section, reduces the image in feature dimension while increasing the spatial dimensions. It uses skip connections that allow it to tap into the same-sized output of the contraction section, which allows it to use higher-level locality features in its upsampling.

## 5. Training and Evaluation
1. The model was trained for 15 epochs using a batch size of 16. We have used a smaller batch size because of memory issues.
2. Loss function used during training was binary cross-entropy loss and metrics used were BCE and IoU score.
3. Adam was used as optimizer and learning rate was set to 5e-4.
4. After training for 15 epochs, we obtained a training iou score of 0.7991 and validation iou score of 0.7103.
<br><br>
![Training and Evaluation Plot](https://github.com/FaizalKarim280280/Flood-Segmentation/blob/main/plots/train%20eval%20plot.jpg)

## 6. References
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) - Olaf Ronneberger, Philipp Fischer, Thomas Brox (2015)
- [Tensorflow](https://www.tensorflow.org/)
- [Label Studio](https://labelstud.io/)
- [Albumentation](https://albumentations.ai/docs/)
- [Segmentation Models](https://github.com/qubvel/segmentation_models)

