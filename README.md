# Fake video detection
## Fake video:
This project processes videos which is created by Thins-plate motion model, MyHeritage, Revive, TalkingPhoto,etc. These video convert optical motion from the driving video into the target image. You can see a sample of this type of fake video below:
![Sample](image and video/Sample.gif)
## Dataset:
Dataset for training is created by Thins-plate motion model (https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model)
Link download dataset: https://drive.google.com/drive/folders/1HPIymlKiuP5mDGTZ6sn6tn7Yk61qbgDe?usp=sharing
## Model achitechture
### Predict video
Divide video into n segment. Each segment will be predicted by LSTM model. Average score will be used to classify fake or real video.
### Predict segment
To extract Remote Photoplethysmography, this project use Local Group Invariance (doi: 10.1109/CVPRW.2018.00172.). After that, rPPG is converted into frequency domain by using Fourier Transform. Both temporial and frequency domain signal is fed into LSTM model. Faces is also used for CNN model to predict appearance. Output of CNN and LSTM is concatenated into vector for fully connected layer.
![Model](model.png)
Structure of file
- filter: include methods for rPPG signal extraction (such as: CHROM, POS, LGI)
- model: include SVM model and LSTM model
