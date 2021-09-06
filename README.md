# face-mask-detection
Hello welcome to my face mask detection

its a python model which can detect faces with FaceBoxes : https://github.com/zisianw/FaceBoxes.PyTorch and classify if the face has a mask or not

# Requirements
Ubuntu      " It's only tested on Ubuntu, so it may not work on Windows "

GPU : Any GPU that is works with PyTorch with atleast 4GP vram

numpy : https://numpy.org/

PyTorch : https://pytorch.org/

torchvision : https://pypi.org/project/torchvision/

openCV : https://pypi.org/project/opencv-python/

# Accuracy  

data set : https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset

train set accuracy is : 100%

test set accuracy is : 99.8%

val set accuracy is : 100%

to download my model : https://drive.google.com/file/d/15JenLZWt4qh24PbfaNgGaiCg0pZ16dWt/view?usp=sharing

![output](https://user-images.githubusercontent.com/49597655/131670064-d817581b-e0fc-4573-8cf8-f51488723963.gif)

# Usage

* Clone the Repo :

      $ git clone https://github.com/AbdallahOmarAhmed/face-mask-detection
      
* Compile the nms :

      $ bash ./make.sh

* Download the data set and put it in your project dir :
  
      https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset

* Run the train file to start training : 
      
      $ python3 train_model.py
      
* OR : download my model and put it in the weights folder 
          
* Run the test file to Calculate accuracy : 
      
      $ python3 test_model.py
      
* Add a video to test then run the model :

      $ python3 make_video.py
 
* Convert the models to onnx : 

      $ python3 make_onnx_copy.py

Note : the name of the video that you want to test must be ' test.mp4 '  and the output video name will be ' outpy.avi '
