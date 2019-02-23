from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam
import numpy as np
from keras.layers import  Dense,concatenate,Flatten
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import metrics
import tensorflow as tf
import keras as k
vgg_conv = VGG16(weights='imagenet',
                 include_top=True,
                 input_shape=(224, 224, 3))

train_dir = './data/train'
validation_dir = './data/val'


nTrain =14160#14178
nVal = 1800#1838
nbr_classe = 431



datagen = ImageDataGenerator(rotation_range=20.,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=0.2,
                                        horizontal_flip=True)
batch_size =20

train_features = np.zeros(shape=(nTrain, 224,224,3))
train_labels = np.zeros(shape=(nTrain, 1))

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

i = 0
nImages = nTrain
vgg_conv.layers.pop()
#vgg_conv.layers.pop()
vgg_conv.outputs = [vgg_conv.layers[-1].output]
vgg_conv.layers[-1].outbound_nodes = []
vgg_conv.summary()
for layer in vgg_conv.layers[:-7]:#-7
  layer.trainable = False
  print(layer)
vgg_conv.summary()
for inputs_batch, labels_batch in train_generator:
    print(i*batch_size)
    #features_batch = vgg_conv.predict(inputs_batch)
    train_features[i * batch_size: (i + 1) * batch_size] = inputs_batch
    for m in range(i*batch_size,(i+1)*batch_size):
         train_labels[m] = np.where(labels_batch[m-i*batch_size]==1)
    #train_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nImages:
        break

#train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))


val_features = np.zeros(shape=(nVal, 224,224,3))
val_labels = np.zeros(shape=(nVal, 1))

val_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

i = 0
nImages = nVal
for inputs_batch, labels_batch in val_generator:
    print(i*batch_size)
    #features_batch = vgg_conv.predict(inputs_batch)
    val_features[i * batch_size: (i + 1) * batch_size] = inputs_batch
    for m in range(i*batch_size,(i+1)*batch_size):
         val_labels[m] = np.where(labels_batch[m-i*batch_size]==1)
    #val_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nImages:
        break

#val_features = np.reshape(val_features, (nVal, 4096))

from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dropout, Dense
from keras.models import Model
fc1 = vgg_conv.layers[-2]
fc2 = vgg_conv.layers[-1]
dropout1 = Dropout(0.5)
dropout2 = Dropout(0.5)
x = dropout1(fc1.output)
x = fc2(x)
x = dropout2(x)
#model = vgg_conv.layers[-1].output
#model = Flatten()(model)
#model = Dense(4096,activation="relu")(model)
#model = Dropout(0.5)(model)
#model = Dense(4096,activation="relu")(model)
#model = Dropout(0.5)(model)
#model = CompactBilinearPooling(d = 16000)(model)
model = Dense(nbr_classe,activation="softmax")(x)

model_final = Model(input=vgg_conv.input, output= model)
model_final.summary()
#model_final.load_weights("all_fc2.h5")
#model_final.compile(optimizer=optimizers.SGD(lr=0.1, momentum=0.9,decay=0.0,nesterov=False),loss='categorical_crossentropy')

def sparse_loss(y_true,y_pred):
   return tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred)
decoder_target = tf.placeholder(dtype='int32',shape=(None,nbr_classe))

model_final.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),metrics=["accuracy"],loss='sparse_categorical_crossentropy')
checkpoint = ModelCheckpoint("all_fc2.h5", monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
print("training started !")
tbCallBack = TensorBoard(log_dir='./Graph',histogram_freq=0, write_graph=True,write_images=True)

history = model_final.fit(train_features,
                    train_labels,
                    #steps_per_epochs=100,
                    epochs=1200,
                    batch_size=batch_size,
                    validation_data=(val_features, val_labels),
                    callbacks = [checkpoint, early,tbCallBack])
