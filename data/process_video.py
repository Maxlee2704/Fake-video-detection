import cv2

from data.ROI import *
from filter.CHROM import *
from filter.POS_WANG import *
from filter.LGI import *
import pandas as pd
from analysis.PSDmap import *
import time
import matplotlib.pyplot as plt

#Init face landmark detection
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("D:/BKU/Monhoc/223_Final/code_final/face_landmarker/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")
f = []
signalS, signalS2,signalS3, power, label = [], [], [], [], []
apperance,apperance1,apperance2 = [],[],[]
error_video = []
region = ['nose','leftcheek','rightcheek']


if __name__ == "__main__":
    dtframe = pd.read_csv(r"D:\BKU\Monhoc\223_Final\data\Deepfake_Collect\Thin-Plate-Spline-Motion-Model\Dataset\Celeb\Book1.csv").values
    n = dtframe.shape[0]
    for i in range (0,n):
        try:
            count = 0
            frames = []
            BVP,BVP2,BVP3 = [], [], []
            frames=[[],[],[],[]]
            app, app1, app2 = [], [], []
            start = time.time()

            name = dtframe[i,0]

            if dtframe[i,1] ==1:
                PATH_VD = "D:/BKU/Monhoc/223_Final/data/Deepfake_Collect/Thin-Plate-Spline-Motion-Model/Dataset/Celeb/Celeb-synthesis/" + namr
            else:
                PATH_VD = "D:/BKU/Monhoc/223_Final/data/Deepfake_Collect/Thin-Plate-Spline-Motion-Model/Dataset/Celeb/Celeb-real/" + name

            print("Processing: " + PATH_VD)

            cap = cv2.VideoCapture(PATH_VD)
            fs = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(total_frames)
            if total_frames<75+150:
                #print('Fail')
                continue
            print(fs)
            print('Width ', width)
            print('Height ', height)


            cap.set(cv2.CAP_PROP_POS_FRAMES, 75)

            while True:
                ret, frame = cap.read()
                if width > 500:
                    frame = cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
                if frame is not None:
                    faces = face_detector(frame, 1)

                    for face in faces:
                        xf, yf, wf, hf = face.left(), face.top(), face.width(), face.height()

                    for k, d in enumerate(faces):
                        shape = landmark_detector(frame, d)
                        count += 1
                        framex = frame.copy()

                        for r in range(3):
                            x0, y0, x1, y1 = ROI(shape, region[r])
                            frames[r].append(frame[y0:y1, x0:x1, :])
                            cv2.rectangle(framex, (x0, y0), (x1, y1), (255, 255, 255), 1)
                        frames[3].append(frame[yf:yf + hf, xf:xf + wf, :])
                        frame_temp = cv2.resize(frame[yf:yf + hf, xf:xf + wf, :],(100,100),interpolation=cv2.INTER_LINEAR)
                        frame_temp = cv2.cvtColor(frame_temp,cv2.COLOR_BGR2GRAY)
                        app.append(frame_temp)

                        cv2.imshow('Video', framex)
                if (count>total_frames-3) or (count == 150) or (cv2.waitKey(1) & 0xFF == ord('q')):
                    count = 0
                    break
            cap.release()
            cv2.destroyAllWindows()

            for stt in range(len(frames)):
                temp = LGI(frames[stt])
                temp2 = CHROME_DEHAAN(frames[stt],fs)
                temp3 = POS_WANG(frames[stt],fs)


                BVP.append(temp)
                BVP2.append(temp2)
                BVP3.append(temp3)
            if len(app) ==150:
                apperance.append(app)

            else:
                continue
                #print(temp.shape)

            BVP = np.array(BVP)
            BVP2 = np.array(BVP2)
            BVP3 = np.array(BVP3)


            print(BVP.shape)
            #print(apperance.shape)

            if BVP.shape != (4, 150):
                BVP = cv2.resize(BVP, (150, 4), interpolation=cv2.INTER_LINEAR)

            if BVP2.shape != (4, 150):
                BVP2 = cv2.resize(BVP2, (150, 4), interpolation=cv2.INTER_LINEAR)

            if BVP3.shape != (4, 150):
                BVP3 = cv2.resize(BVP3, (150, 4), interpolation=cv2.INTER_LINEAR)
            #if PSD.shape != (3, 100):
             #   PSD = cv2.resize(PSD, (100, 3), interpolation=cv2.INTER_LINEAR)
            #print(BVP.shape)

            label.append(dtframe[i,1])
            f.append(fs)
            signalS.append(BVP)
            signalS2.append(BVP2)
            signalS3.append(BVP3)


            signalS = np.array(signalS)
            signalS2 = np.array(signalS2)
            signalS3 = np.array(signalS3)

            f = np.array(f)
            label = np.array(label)

            np.save(r'D:\BKU\Monhoc\223_Final\code_final\data\final\CelebDF\segment75\LGI6', signalS)
            np.save(r'D:\BKU\Monhoc\223_Final\code_final\data\final\CelebDF\segment75\CHROM6', signalS2)
            np.save(r'D:\BKU\Monhoc\223_Final\code_final\data\final\CelebDF\segment75\POS6', signalS3)


            np.save(r'D:\BKU\Monhoc\223_Final\code_final\data\final\CelebDF\segment75\label6', label)
            np.save(r'D:\BKU\Monhoc\223_Final\code_final\data\final\CelebDF\segment75\freq6', f)


            end = time.time()
            print('Thời gian thực hiện: ',end-start)
            print(signalS.shape)
            signalS = signalS.tolist()
            signalS2 = signalS2.tolist()
            signalS3 = signalS3.tolist()
            #apperance = apperance.tolist()
            #apperance1 = apperance1.tolist()
            #apperance2 = apperance2.tolist()
            label = label.tolist()
            f = f.tolist()

        except:
            print('Error: ', PATH_VD)
            pass
    apperance = np.array(apperance)


    np.save(r'D:\BKU\Monhoc\223_Final\code_final\data\final\CelebDF\segment75\gray6', apperance)


