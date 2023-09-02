#from metric import *
import numpy as np
from keras.models import load_model
from scipy import fft
from filter.LGI import *
from filter.POS_WANG import *
from data.ROI import *
import time
import dlib
import cv2
import keras
import numpy as np
from keras.callbacks import EarlyStopping, Callback
import tensorflow as tf
from keras import optimizers
from keras import regularizers
from keras.layers import BatchNormalization, Input
from keras.layers import Dense, Dropout, Activation, Concatenate, LSTM, Bidirectional
from keras.models import Model
from sklearn.utils import shuffle
import numpy as np
#####################################################################################
#Load model
model = load_model('./model.h5)
#model.summary()
#####################################################################################
#Load face detection model
start =  time.time()
#Init face landmark detection
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("./ROI_extract/shape_predictor_68_face_landmarks.dat")
#####################################################################################
#Init variable
count,BVP_idx =0, 0 #Count frame and BVP
app = [] #Contain frames of face
frames = [[],[],[]] #Contain frames of ROI 
BVP = [] #Contain rPPG
data = []
times = 0
region = ['nose','leftcheek','rightcheek']
score = 'No data' #Score of predict segment
result = 'No data' #Fake or Real value
score = []
win = 50 #Size of window
#####################################################################################
# FFT
def fourier(signal,fs):
    FFT = [[],[],[]]
    for i in range(signal.shape[0]):
        FFT[i].append(abs(fft.fft(signal[i])))
    FFT = np.array(FFT).reshape(3,150)
    return FFT
    
# Bandpass filter
def filter(s,freq):
    LPF = 0.7
    HPF = 2.5
    WinSec = 1.6
    temp = np.zeros_like(s)
    FS = freq
    for j in range(s.shape[0]):
        a = s[j, :]
        NyquistF = 1 / 2 * FS
        B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')
        temp[j, :] = signal.filtfilt(B, A, a)
    return temp
#####################################################################################
if __name__ == "__main__":
    PATH_VD = './video.mp4'
    print("Processing: " + PATH_VD)

    #Information of video
    cap = cv2.VideoCapture(PATH_VD)
    fs = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #Main 
    while True:
        ret, frame = cap.read()
        if width > 600:
            frame = cv2.resize(frame,(width//2,height//2),interpolation=cv2.INTER_LINEAR)
        if frame is not None:
            faces = face_detector(frame, 1)
            for face in faces:
                xf, yf, wf, hf = face.left(), face.top(), face.width(), face.height()
            for k, d in enumerate(faces):
                shape = landmark_detector(frame, d)
                # Find Region of Interest (xy store coordinate of ROI)
            count += 1
            BVP_idx +=1
            framex = frame.copy()

            #Store frame of ROI
            for r in range(3):
                x0, y0, x1, y1 = ROI(shape, region[r])
                frames[r].append(frame[y0:y1, x0:x1, :])
                cv2.rectangle(framex, (x0, y0), (x1, y1), (255, 255, 255), 1)

            #Store frame of face
            frame_temp = cv2.cvtColor(frame[yf:yf + hf, xf:xf + wf, :],cv2.COLOR_BGR2GRAY)
            app.append(cv2.resize(frame_temp, (100, 100), interpolation=cv2.INTER_LINEAR))

            cv2.imshow('Video', framex)
            cv2.putText(framex, str(score), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if len(frames[0]) %150 ==0:
                if times ==0:
                    SR = 0 #Start of window
                    ER = 150 #End of window
                else:
                    SR += win
                    ER += win
                range_test = [SR,ER]
                times +=1

                for stt in range(3):
                    BVP.append(LGI(frames[stt]))

                BVP = np.array(BVP)
                BVP = filter(BVP, fs)
                app = np.array(app)
                FFT = fourier(BVP,fs)

                #Scale for feeding in model
                BVP = np.expand_dims(BVP,axis=0)
                FFT = np.expand_dims(FFT,axis=0)
                app_in = np.expand_dims(app[:100,:,:],axis=0)
                
                BVP = np.swapaxes(BVP, 1, 2)
                FFT = np.swapaxes(FFT, 1, 2)

                #score: include values of segments' score
                #segment_score: values of one segment
                input = [[BVP,FFT],app_in]
                segment_score = model.predict(input, verbose=0)
                segment_score = np.round(segment_score[2][0],2)
                score.append(segment_score)
                if segment_score >0.5:
                    segment_label = 'Fake'
                    print('Frame from ' + str(range_test) + ': ' + str(score[0]) + ' -> ' + '\033[91m' +segment_label + '\033[0m')
                else:
                    segment_label = 'Real'
                    print('Frame from ' + str(range_test) + ': ' + str(score[0]) + ' -> ' + '\033[92m' +segment_label + '\033[0m')
                data.append(score)

                #Using window
                BVP_idx -=win
                frame_win = [[], [], []]
                for i in range(3):
                    frame_win[i] = frames[i][win:]

                app = app[win:,:,:]
                app = app.tolist()
                frames = frame_win

                BVP,FFT = [], []
            cv2.putText(framex, str(segment_score), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(framex, result, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Video',framex)

        if (count==total_frames-3) or (cv2.waitKey(1) & 0xFF == ord('q')):
            count = 0
            break
    cap.release()
    cv2.destroyAllWindows()

    mean_score = np.mean(score) #mean of segments' score
    mean_score = np.int8(np.round(mean_score,0))
    num1 = np.count_nonzero(score) #A number segment which is predicted as Fake
    num0 = score.shape[0] - num1 #A number segment which is predicted as Real
    print(mean_score.T)

    times = 0
    score,BVP, FFT = [],[],[]
    frames = [[], [], []]
    app = []
    score = 'No data'
    result = 'No data'
    
    if num1 > num0:
        video_label = "Fake"
    elif num0 > num1:
        video_label = "Real"
    elif num0 == num1:
        if pre > 0.5:
            video_label = "Fake"
        else:
            video_label = "Real"
    print("Video result: ",video_label)
    
