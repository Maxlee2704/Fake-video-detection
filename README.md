# Fake video detection
## Fake video:
This project processes videos which is created by Thins-plate motion model, MyHeritage, Revive, TalkingPhoto,etc. These video convert optical motion from the driving video into the target image. You can see a sample of this type of fake video below:
![model](Sample.gif)

Using Remote Photoplethysmography for Deepfake video detection: (LGI + LSTM)
Divide video into n segment. Each segment will be predicted by LSTM model. Average score will be used to classify fake or real video.
![Model](model.png)
Structure of file
- filter: include methods for rPPG signal extraction (such as: CHROM, POS, LGI)
- model: include SVM model and LSTM model
