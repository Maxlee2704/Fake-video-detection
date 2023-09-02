# Fake video detection
## Fake video:
This project processes videos which is created by Thins-plate motion model, MyHeritage, Revive, TalkingPhoto,etc. These video convert optical motion from the driving video into the target image. You can see a sample of this type of fake video below:
![Samplevideo](https://github.com/Maxlee2704/Fake-video-detection/blob/main/image%20and%20video/Sample.gif)
## Dataset:
Dataset for training is created by Thins-plate motion model (https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model)
Link download dataset: https://drive.google.com/drive/folders/1HPIymlKiuP5mDGTZ6sn6tn7Yk61qbgDe?usp=sharing
## Model achitechture
### Predict video
Divide video into n segment. Each segment will be predicted by LSTM model. Average score will be used to classify fake or real video.
### Predict segment
To extract Remote Photoplethysmography, this project use Local Group Invariance (doi: 10.1109/CVPRW.2018.00172.). After that, rPPG is converted into frequency domain by using Fourier Transform. Both temporial and frequency domain signal is fed into LSTM model. Faces is also used for CNN model to predict appearance. Output of CNN and LSTM is concatenated into vector for fully connected layer.
![model](https://github.com/Maxlee2704/Fake-video-detection/blob/main/image%20and%20video/model_new.png)
## Structure of file
- ROI_extract: extract ROI and face detection
- data: include dataset
- image and video: include sample of video, image
- method: include methods for rPPG signal extraction (such as: CHROM, POS, LGI)
- model: training model
- report: detailed report in Vietnamese and brief summary in English
- run.py: predict a video
