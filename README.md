# Deepfake-detection-using-rPPG
Using Remote Photoplethysmography for Deepfake video detection: (LGI + LSTM)
Divide video into n segment. Each segment will be predicted by LSTM model. Average score will be used to classify fake or real video.
![Model](model.png)
Structure of file
- filter: include methods for rPPG signal extraction (such as: CHROM, POS, LGI)
- model: include SVM model and LSTM model
