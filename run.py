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
from attention import Attention
import pandas as pd

model = load_model(r'D:\BKU\Monhoc\223_Final\code_final\model\Thinsplate2908.h5')
model.summary()

#model.summary()


start =  time.time()
#Init face landmark detection
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("D:/BKU/Monhoc/223_Final/code_final/face_landmarker/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")
y_pre,y = [], []
frames = []
sample = []
count =0
BVP_idx = 0
data = []
app = []
frames = [[],[],[],[]]
BVP = []
times = 0
region = ['nose','leftcheek','rightcheek']
score = 'No data'
result = 'No data'
final = []
win = 50
def fourier(signal,fs):
    FFT = [[],[],[]]
    for i in range(signal.shape[0]):
        FFT[i].append(abs(fft.fft(signal[i])))
    FFT = np.array(FFT).reshape(3,150)
    return FFT
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
if __name__ == "__main__":
    dtframe = pd.read_csv('D:/BKU/Monhoc/223_Final/data/Deepfake_Collect/Thin-Plate-Spline-Motion-Model/Test/Book1.csv').values
    n = dtframe.shape[0]
    for i in range(200, n):
        #try:
        name = dtframe[i, 0]
        #PATH_VD = r'D:\BKU\Monhoc\223_Final\data\Deepfake_Collect\Test_case\Fake_Real_Fake\FR01.mp4'
        if dtframe[i, 1] == 1:
            PATH_VD = "D:/BKU/Monhoc/223_Final/data/Deepfake_Collect/Thin-Plate-Spline-Motion-Model/Test/Fake/" + name
        else:
            PATH_VD = "D:/BKU/Monhoc/223_Final/data/Deepfake_Collect/Thin-Plate-Spline-Motion-Model/Test/Real/" + name

        print("Processing: " + PATH_VD)

        cap = cv2.VideoCapture(PATH_VD)


        fs = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #if width>600:
         #   continue
        #print('Tổng số frame: ',total_frames)
        #print('FPS: ',fs)
        #print('Width: ',width)
        #print('Height: ',height)
        #cap.set(cv2.CAP_PROP_POS_FRAMES, 100)


        while True:
            ret, frame = cap.read()
            if width > 600:
                frame = cv2.resize(frame,(width//2,height//2),interpolation=cv2.INTER_LINEAR)
            if frame is not None:
                faces = face_detector(frame, 1)
                for face in faces:
                    xf, yf, wf, hf = face.left(), face.top(), face.width(), face.height()
                    #cv2.rectangle(frame, (xf, yf), (xf+wf, yf+hf), (255, 255, 255), 1)
                for k, d in enumerate(faces):

                    shape = landmark_detector(frame, d)


                    # Find Region of Interest (xy store coordinate of ROI)
                count += 1
                BVP_idx +=1
                #print(count)
                framex = frame.copy()
                for r in range(3):
                    x0, y0, x1, y1 = ROI(shape, region[r])
                    frames[r].append(frame[y0:y1, x0:x1, :])
                    cv2.rectangle(framex, (x0, y0), (x1, y1), (255, 255, 255), 1)
                frames[3].append(frame[yf:yf + hf, xf:xf + wf, :])
                frame_temp = cv2.cvtColor(frame[yf:yf + hf, xf:xf + wf, :],cv2.COLOR_BGR2GRAY)
                #app.append(cv2.resize(frame_temp, (100, 150), interpolation=cv2.INTER_LINEAR))
                app.append(cv2.resize(frame_temp, (100, 100), interpolation=cv2.INTER_LINEAR))

                #print(len(app))
                cv2.imshow('Video', framex)
                cv2.putText(framex, str(score), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if len(frames[0]) %150 ==0:
                    if times ==0:
                        SR = 0
                        ER = 150
                    else:
                        SR += win
                        ER += win
                    range_test = [SR,ER]
                    times +=1

                    for stt in range(4):
                        BVP.append(LGI(frames[stt]))

                    BVP = np.array(BVP)[:3,:]
                    BVP = filter(BVP, fs)
                    app = np.array(app)
                    FFT = fourier(BVP,fs)


                    BVP = np.expand_dims(BVP,axis=0)
                    FFT = np.expand_dims(FFT,axis=0)
                    app_in = np.expand_dims(app[:100,:,:],axis=0)
                    #print(app.shape)

                    BVP = np.swapaxes(BVP, 1, 2)
                    FFT = np.swapaxes(FFT, 1, 2)


                    input = [[BVP,FFT],app_in]
                    temp_score = model.predict(input, verbose=0)
                    score = np.round(temp_score[2][0],2)
                    #print(temp_score)

                    final.append(score)
                    if score >0.5:
                        result = 'Fake'
                        print('Frame từ ' + str(range_test) + ': ' + str(score[0]) + ' -> ' + '\033[91m' +result + '\033[0m')
                    else:
                        result = 'Real'
                        print('Frame từ ' + str(range_test) + ': ' + str(score[0]) + ' -> ' + '\033[92m' +result + '\033[0m')
                    data.append(score)

                    BVP_idx -=win
                    temp = [[], [], [], []]
                    for i in range(4):
                        temp[i] = frames[i][win:]

                    app = app[win:,:,:]
                    app = app.tolist()


                    #print(len(temp))
                    #print(len(temp[0]))
                    frames = temp

                    #frames = [[],[],[],[]]
                    #app = []
                    BVP,FFT = [], []
                cv2.putText(framex, str(score), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(framex, result, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow('Video',framex)
            # Take 300 frames in 10 seconds
            if (count==total_frames-3) or (cv2.waitKey(1) & 0xFF == ord('q')):
                count = 0
                break
        cap.release()
        cv2.destroyAllWindows()

        pre = np.mean(final)
        final = np.int8(np.round(final,0))
        num1 = np.count_nonzero(final)
        num0 = final.shape[0] - num1

        print(final.T)
        #print('Điểm cuối cùng: ',pre)
        times = 0
        final,BVP, FFT = [],[],[]
        frames = [[], [], [], []]
        app = []
        score = 'No data'
        result = 'No data'
        temp_co=[]
        if num1 > num0:
            temp = 1
        elif num0 > num1:
            temp = 0
        elif num0 == num1:
            if pre > 0.5:
                temp = 1
            else:
                temp = 0
        print(temp)
        y_pre.append(temp)
        if temp != dtframe[i,1]:
            print('Lỗi: ',PATH_VD)
        y.append(dtframe[i, 1])

        y_pre = np.array(y_pre)
        y = np.array(y)

        np.save('data/Prediction/Thinsplate_pre',y_pre)
        np.save('data/Prediction/Thinsplate_y',y)
        #print(y_pre.shape)
        y_pre = y_pre.tolist()
        y = y.tolist()
        #except:
         #   print('Error')
          #  final, BVP, FFT, app = [], [], [], []
            #frames = [[], [], [], []]