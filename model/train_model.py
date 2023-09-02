import keras
import numpy as np
from keras.callbacks import EarlyStopping, Callback
import tensorflow as tf
from keras import optimizers
from keras import regularizers
from keras.layers import BatchNormalization, Input, MaxPooling2D,Average, BatchNormalization
from keras.layers import Dense, Dropout, Activation, Concatenate, LSTM, Bidirectional, Conv2D, AveragePooling2D, Flatten
from keras.models import Model
from sklearn.utils import shuffle
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(20)


########################################################################
base_folder = '/content/drive/MyDrive/Fake video detection/model/exp'
index = 0
folder_path = f'{base_folder}{index}'

while os.path.exists(folder_path):
    index += 1
    folder_path = f'{base_folder}{index}'

os.makedirs(folder_path)
print(f"Save_path: {folder_path}")
########################################################################
# Load Data
X = np.load('/content/drive/MyDrive/Fake video detection/data/dataset/LGI_filter.npy')[:,:3,:]
X = np.swapaxes(X,1,2)

Xa = np.load('/content/drive/MyDrive/Fake video detection/data/dataset/LGI_filter_fft.npy')[:,:3,:]
Xa = np.swapaxes(Xa,1,2)

Xb = np.load('/content/drive/MyDrive/Fake video detection/data/dataset/gray.npy')[:,:100,:,:]
#print(Xb.shape)
Y = np.load('/content/drive/MyDrive/Fake video detection/data/dataset/label.npy')
for i in range(5):
  X,Xa,Xb,Y =shuffle(X,Xa,Xb,Y)
########################################################################
# Model
def rPPG_model():
  weight_decay = 0.0001
  inputs0 = Input(shape=(150, 3))
  inputs1 = Input(shape=(150, 3))

  in0 = Bidirectional(LSTM(100))(inputs0)
  in0 = Dense(50,kernel_regularizer=regularizers.l2(weight_decay))(in0)
  in0 = Activation('leaky_relu')(in0)


  in1 = Bidirectional(LSTM(100))(inputs1)
  in1 = Dense(50,kernel_regularizer=regularizers.l2(weight_decay))(in1)
  in1 = Activation('leaky_relu')(in1)



  merged = Concatenate()([in0, in1])
  merged = Dense(128,kernel_regularizer=regularizers.l2(weight_decay))(merged)
  merged = Activation('leaky_relu')(merged)
  merged = Dropout(0.5)(merged)
  output = Dense(1)(merged)
  output = Activation('sigmoid',name='Sout1')(output)
  inputs = [inputs0,inputs1]
  outputs = [output]
  model = Model(inputs, outputs)
  return model


########################################################################
def inception_module(x, filters):
    weight_decay = 0.0005
    branch1x1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
    branch1x1 = Dropout(0.3)(branch1x1)
    branch3x3 = Conv2D(filters[1], (1, 1), padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
    branch3x3 = Conv2D(filters[2], (3, 3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(branch3x3)
    branch3x3 = Dropout(0.3)(branch3x3)
    branch5x5 = Conv2D(filters[3], (1, 1), padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
    branch5x5 = Conv2D(filters[4], (5, 5), padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(branch5x5)
    branch5x5 = Dropout(0.3)(branch5x5)
    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Conv2D(filters[5], (1, 1), padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(branch_pool)

    output = Concatenate(axis=-1)([branch1x1, branch3x3, branch5x5, branch_pool])
    return output



def Appear_model():
  weight_decay = 0.0005
  inputs2 = Input(shape=(100,100,100))
  

  x = Conv2D(64, (7, 7), strides=(2, 2), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(inputs2)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.3)(x)
  x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

  #x = Conv2D(129, (1, 1), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
  #x = BatchNormalization()(x)
  #x = Activation('relu')(x)
  #x = Dropout(0.3)(x)


  x = Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.3)(x)

  x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

  x = inception_module(x, [16, 24, 32, 8, 8, 8])
  #x = inception_module(x, [128, 128, 192, 32, 96, 64])

  # Add more inception modules as needed

  x = AveragePooling2D((7, 7), strides=(1, 1))(x)
  x = Flatten()(x)
  x = Dense(64,kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = Activation('leaky_relu')(x)
  x = Dropout(0.3)(x)
  sub_out2 = Dense(1)(x)
  sub_out2 = Activation('sigmoid', name="Sout2")(sub_out2)


  inputs = [inputs2]
  outputs = [sub_out2]
  model = Model(inputs, outputs)
  return model

########################################################################
loss = tf.keras.losses.BinaryCrossentropy()
opt = optimizers.Adam(1e-4, decay=1e-5,clipnorm=1.0)
metrics = ['accuracy',tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()]
early_stop = EarlyStopping(monitor='val_loss',patience=5, verbose=1,restore_best_weights=True)


modelA = rPPG_model()
modelB = Appear_model()
weight_decay = 0.0005

output = Concatenate()([modelA.layers[-4].output,modelB.layers[-4].output])
output = Dense(128,kernel_regularizer=regularizers.l2(weight_decay))(output)
output = BatchNormalization()(output)

output = Activation('leaky_relu')(output)
output = Dropout(0.5)(output)

output = Dense(1)(output)
output = Activation('sigmoid',name='f_out')(output)
model = Model(inputs = [modelA.input, modelB.input], outputs = [modelA.output,modelB.output,output])
########################################################################
# Trainining
model.compile(loss=[loss,loss,loss],optimizer=opt, metrics=metrics)
model.summary()
X_in = [[X,Xa],Xb]
Y_in = [Y,Y,Y]
history = model.fit(X_in,Y_in,batch_size=128, epochs=1000,validation_split=0.3,shuffle=True, verbose=1, callbacks=[early_stop])



########################################################################
plot_model(model, to_file=folder_path+'/model_architecture.png', show_shapes=True)
model.save(folder_path + "/model.h5")
# Save the training and validation history to a text file
with open(folder_path+'/loss_acc.txt', 'w') as f:
    # Write the header with column names
    f.write('epoch\tloss\tacc\tval_loss\tval_acc\n')

    # Write the values for each epoch
    for epoch, loss, accuracy, val_loss, val_accuracy, AUC, Precision, Recall in zip(range(1, len(history.history['f_out_loss']) + 1),
                                                             history.history['f_out_loss'],
                                                             history.history['f_out_accuracy'],
                                                             history.history['val_f_out_loss'],
                                                             history.history['val_f_out_accuracy'],
                                                             history.history['val_f_out_auc'],
                                                             history.history['val_f_out_precision'],
                                                             history.history['val_f_out_recall']):
        f.write(f'{epoch}\t{loss:.4f}\t{accuracy:.4f}\t{val_loss:.4f}\t{val_accuracy:.4f}\t{AUC:.4f}\t{Precision:.4f}\t{Recall:.4f}\n')
#Plot and save
# Load the training history from the text file
epochs = []
train_loss = []
train_acc = []
val_loss = []
val_acc = []
AUC = []
Precision = []
Recall = []

with open(folder_path+'/loss_acc.txt', 'r') as f:
    # Skip the header row
    f.readline()

    for line in f:
        epoch, loss, accuracy, val_loss_value, val_accuracy_value, AUC_value, Precision_value, Recall_value = line.strip().split('\t')
        epochs.append(int(epoch))
        train_loss.append(float(loss))
        train_acc.append(float(accuracy))
        val_loss.append(float(val_loss_value))
        val_acc.append(float(val_accuracy_value))
        AUC.append(float(AUC_value))
        Precision.append(float(Precision_value))
        Recall.append(float(Recall_value))
        F1_score.append(float(2*(Precision_value*Recall_value)/(Precision_value+Recall_value)))

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 8))


axes[0, 0].plot(epochs, train_loss)
axes[0, 0].set_title('Training loss')

axes[1, 0].plot(epochs, val_loss)
axes[1, 0].set_title('Validation loss')

axes[0, 1].plot(epochs, train_acc)
axes[0, 1].set_title('Training Acc')

axes[1, 1].plot(epochs, val_acc)
axes[1, 1].set_title('Validation Acc')

axes[0, 2].plot(epochs, AUC)
axes[0, 2].set_title('AUC')

axes[1, 2].plot(epochs, Precision)
axes[1, 2].set_title('Precision')

axes[0, 3].plot(epochs, Recall)
axes[0, 3].set_title('Recall')

axes[1, 3].plot(epochs, F1_score)
axes[1, 3].set_title('F1-Score')

plt.tight_layout()
plt.savefig(folder_path+'/Loss_acc.png')
