from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.layers import  Dense,concatenate,Flatten
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import metrics
vgg_conv = VGG16(weights='imagenet',
                 include_top=True,
                 input_shape=(224, 224, 3))

train_dir = './data/train'
validation_dir = './data/val'


nTrain = 14178
nVal = 1838
nbr_classe = 431


datagen = ImageDataGenerator(rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
batch_size = 64

train_features = np.zeros(shape=(nTrain, 224,224,3))
train_labels = np.zeros(shape=(nTrain, nbr_classe))

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

i = 0
nImages = nTrain
'''vgg_conv.layers.pop()
vgg_conv.layers.pop()
vgg_conv.layers.pop()
vgg_conv.outputs = [vgg_conv.layers[-1].output]
vgg_conv.layers[-1].outbound_nodes = []
vgg_conv.summary()'''
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
    train_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nImages:
        break

#train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))


val_features = np.zeros(shape=(nVal, 224,224,3))
val_labels = np.zeros(shape=(nVal, nbr_classe))

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
    val_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
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
#model = CompactBilinearPooling(d = 16000)(model)
model = Dense(nbr_classe,activation="softmax")(x)

model_final = Model(input=vgg_conv.input, output= model)

#model_final.load_weights("all.h5")
#model_final.compile(optimizer=optimizers.SGD(lr=0.1, momentum=0.9,decay=0.0005,nesterov=False),loss='categorical_crossentropy')
model_final.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9,decay=0.0005),metrics=["accuracy"],loss='categorical_crossentropy')             
checkpoint = ModelCheckpoint("all.h5", monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
print("training started !")
tbCallBack = TensorBoard(log_dir='./Graph',histogram_freq=0, write_graph=True,write_images=True)
history = model_final.fit(train_features,
                    train_labels,
                    epochs=800,
                    batch_size=batch_size,
                    validation_data=(val_features, val_labels),
                    callbacks = [checkpoint, early,tbCallBack])
