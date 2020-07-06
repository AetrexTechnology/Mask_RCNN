# Mask R-CNN for Foot detection

To see more details refer to this official documentation [Detail readme](Default_README.md)

The repository includes:
* To setup this repo clone it first
* After cloning is done create a new environment using

  ```conda create -n mask_rcnn python=3.6```
* The activate the environment using 

    ```conda activate mask_rcnn```
* Once envrinment is activated then use
    
    ``` pip install -r requirements.txt```
* At this point you should have a working environment
* Download mask_rcnn_balloon.h5. Save it in the root directory of the repo (the mask_rcnn directory) from [here](https://github.com/matterport/Mask_RCNN/releases)


# Running inferences on data
* [demo.ipynb](samples/demo.ipynb) can be used to see the output of the pretrained model on the images
* All the images for testing are under 
    ```aetrex``` folder

* Inside ```Run Object Detection section``` set the image directory path, then it choose the image from there randomly and show the predictions like below

    ```IMAGE_DIR='../aetrex/images'```

* Note - ```skimage.io.imread()``` gives output image in 4 channels sometime as oppose to opencv which give 3 channel image by default, if you get channel error than use ```skimage.io.imread()[:,:,:3]```
