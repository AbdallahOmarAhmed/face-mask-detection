# face-mask-detection
Hello welcome to my face mask detection, its a python model which can detect faces and classify if the face has a mask or not

# Requirements
Ubuntu      " It's only tested on Ubuntu, so it may not work on Windows "

GPU : Any GPU that is works with PyTorch 

numpy : https://numpy.org/

PyTorch : https://pytorch.org/

torchvision : https://pypi.org/project/torchvision/

face_recognition : https://github.com/ageitgey/face_recognition

# Accuracy  

my model accuracy is : 97.5

to download my model : https://drive.google.com/file/d/1gsuTLRhZX7DJaOv-e7sVV4xvvgj_lBvd/view?usp=sharing

# Usage

* Download the data set and put it in your project dir :
  
      https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset
  
* Run the train file to start training : 
      
      $ python3 face_mask_train.py
          
* Run the test file to Calculate accuracy : 
      
      $ python3 test_model.py
      
* Add a video to test and run the model :

      $ python3 video_tester.py
     
