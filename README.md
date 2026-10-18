# Fake video detection
Using Remote Photoplethysmography signal, LSTM, and CNN for predict video.
## Fake video:
This project processes videos which is created by Thins-plate motion model, MyHeritage, Revive, TalkingPhoto,etc. These video convert optical motion from the driving video into the target image. You can see a sample of this type of fake video below:
![Samplevideo](https://github.com/Maxlee2704/Fake-video-detection/blob/main/image%20and%20video/Sample.gif)
## Dataset:
Dataset for training is created by the Thin-Plate Spline Motion Model [1]
## Model architechture:
### Predict video
Divide video into n segment. Each segment will be predicted by LSTM model. Average score will be used to classify fake or real video.
### Predict segment
To extract Remote Photoplethysmography, this project use Local Group Invariance [2]. After that, rPPG is converted into frequency domain by using Fourier Transform. Both temporial and frequency domain signal is fed into LSTM model. Faces is also used for CNN model to predict appearance. Output of CNN and LSTM is concatenated into vector for fully connected layer.
![model](https://github.com/Maxlee2704/Fake-video-detection/blob/main/image%20and%20video/model_new.png)
## Structure of file:
- ROI_extract: extract ROI and face detection
- data: include dataset
- image and video: include sample of video, image
- method: include methods for rPPG signal extraction (such as: CHROM, POS, LGI)
- model: training model
- report: detailed report in Vietnamese and brief summary in English
- run.py: predict a video
## References:
[1] Z. Zhao and H. Zhang, “Thin-Plate Spline Motion Model for Image Animation,” arXiv preprint arXiv:2203.14367, 2022, doi: 10.48550/arXiv.2203.14367.
[2] C. S. Pilz, S. Zaunseder, J. Krajewski and V. Blazek, "Local Group Invariance for Heart Rate Estimation from Face Videos in the Wild," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Salt Lake City, UT, USA, 2018, pp. 1335-13358, doi: 10.1109/CVPRW.2018.00172.
