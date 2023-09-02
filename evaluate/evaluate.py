import numpy as np
import matplotlib.pyplot as plt

def accuracy(y,y_predict):
    count = 0
    N = y.shape[0]
    for i in range(N):
        if y[i] == y_predict[i]:
            count +=1
    return count/N
def confusion_matrix(y,y_predict):
    TP, FP, TN, FN  = 0,0,0,0
    for i in range(y.shape[0]):
        if y[i] == y_predict[i] and y[i] ==1:
            TP +=1
        elif y[i] == y_predict[i] and y[i] ==0:
            TN +=1
        elif y[i] != y_predict[i] and y[i] ==0:
            FP +=1
        elif y[i] != y_predict[i] and y[i] == 1:
            FN += 1
    matrix = np.array([[TP,FN],[FP,TN]])
    return matrix

def Precision(matrix):
    score = matrix[0,0] / (matrix[0,0]+matrix[1,0])
    return score

def Recall(matrix):
    score = matrix[0,0] / (matrix[0,0]+matrix[0,1])
    return score

def F1_Score(recall, precision):
    score = (2*precision*recall) / (precision+recall)
    return score
if __name__ == "__main__":
    class_names = ['Fake', 'Real']
    y = np.load(r"D:\BKU\Monhoc\223_Final\code_final\data\Prediction\Thinsplate_y.npy")
    y_pred = np.load(r"D:\BKU\Monhoc\223_Final\code_final\data\Prediction\Thinsplate_pre.npy")

    matrix = confusion_matrix(y,y_pred)
    print("Confusion matrix = \n",matrix)
    precision = Precision(matrix)
    recall = Recall(matrix)
    f1_score = F1_Score(recall,precision)
    acc = accuracy(y,y_pred)
    ####################################################
    # Confusion matrix
    plt.title("Confusion matrix")
    matrix_percent = np.round(matrix/np.sum(matrix,axis=1).reshape(2,1),2)
    plt.xticks(ticks=range(len(class_names)), labels=class_names)
    plt.yticks(ticks=range(len(class_names)), labels=class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(matrix_percent[i, j]), ha='center', va='center', color='orange')

    plt.imshow(matrix_percent,cmap= "Blues")
    plt.show()
    print("Precision = ",precision)
    print("Recall = ",recall)
    print("F1_score = ",f1_score)
    print("Accuracy = ",acc)
