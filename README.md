# face-mask-detection
Hello welcome to my face mask detection, its a python model which can detect faces and classify if the face has a mask or not

# Requirements
Ubuntu      " It's only tested on Ubuntu, so it may not work on Windows "

GPU : Any GPU that is works with PyTorch 

numpy : https://numpy.org/

PyTorch : https://pytorch.org/

torchvision : https://pypi.org/project/torchvision/

face_boxes : https://github.com/zisianw/FaceBoxes.PyTorch

# Accuracy  

train set accuracy is : 100%

test set accuracy is : 99.8%

val set accuracy is : 100%

to download my model : https://drive.google.com/file/d/1iK-lTpm3HtHBLbusS72yUsUQUxl0f842/view?usp=sharing

https://user-images.githubusercontent.com/49597655/131669090-5015200c-cdd1-4020-b3d5-79d7dfab9d0b.mp4

# Usage

* Download the data set and put it in your project dir :
  
      https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset
  
* Run the train file to start training : 
      
      $ python3 face_mask_train.py
          
* Run the test file to Calculate accuracy : 
      
      $ python3 test_model.py
      
* Add a video to test then run the model :

      $ python3 video_tester.py

Note : the name of the video that you want to test should be ' test.mp4 '  
