import cv2,os

path = '/Users/vaneesh_k/PycharmProjects/Mask_RCNN/aetrex/run_this/hsv1_crop.png'
img = cv2.imread(path)
# In case of grayScale images the len(img.shape) == 2
if len(img.shape) > 2 and img.shape[2] == 4:
    #convert the image from RGBA2RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(os.path.split(os.path.abspath(path))[0]+'/'+os.path.split(os.path.abspath(path))[1].split('.')[0]+'_crop.png',img)
    print('------Done-----')