# Fake video detection
This study proposed a DeepFake video detection based on biological signals extracted from videos, with the assumption that DeepFakes cannot mimic biological features (e.g., heart rate).
Specifically, we use the Remote Photoplethysmography signal, which is featured for heart rate variability. Extracted signals are combined with LSTM and CNN to predict video.
## Fake video:
This project processes videos that are created by the Thins-plate motion model, MyHeritage, Revive, TalkingPhoto, etc. These videos convert optical motion from the driving video into the target image. You can see a sample of this type of fake video below:
![Samplevideo](https://github.com/Maxlee2704/Fake-video-detection/blob/main/image%20and%20video/Sample.gif)
## Dataset:
The dataset for training is created by the Thin-Plate Spline Motion Model [1]
## Model architecture:
### Predict video
Divide the video into n segments. Each segment will be predicted by LSTM model. The average score is used to classify the video as fake or real.
### Predict segment
To extract Remote Photoplethysmography, this project uses Local Group Invariance [2]. After that, rPPG is converted into the frequency domain using Fourier Transform. Both the temporal and frequency domain signals are fed into an LSTM model. Faces are also used for a CNN model to predict appearance. The output of the CNN and LSTM is concatenated into a vector for the fully connected layer.
![model](https://github.com/Maxlee2704/Fake-video-detection/blob/main/image%20and%20video/model_new.png)
## Structure of the file:
- ROI_extract: extract ROI and face detection
- data: include dataset
- image and video: include samples of videos and images
- method: include methods for rPPG signal extraction (such as: CHROM, POS, LGI)
- model: training model
- report: detailed report in Vietnamese and brief summary in English
- run.py: predict a video
## References:
[1] Z. Zhao and H. Zhang, “Thin-Plate Spline Motion Model for Image Animation,” arXiv preprint arXiv:2203.14367, 2022, doi: 10.48550/arXiv.2203.14367.

[2] C. S. Pilz, S. Zaunseder, J. Krajewski and V. Blazek, "Local Group Invariance for Heart Rate Estimation from Face Videos in the Wild," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Salt Lake City, UT, USA, 2018, pp. 1335-13358, doi: 10.1109/CVPRW.2018.00172.
