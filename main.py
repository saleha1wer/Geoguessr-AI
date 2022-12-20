from network import initialize_model
from load_data import load_train_test_splits
import numpy as np
import tensorflow as tf
import pickle 
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.vgg16 import VGG16

early_stop = tf.keras.callbacks.EarlyStopping(
monitor='val_loss',
min_delta=0,
patience=15,
verbose=1,
mode='auto',
baseline=None,
restore_best_weights=False
)

red_on_plat = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=0,
    mode='auto',
    min_delta=0,
    cooldown=0,
    min_lr=0,
)

print("TensorFlow version: {}".format(tf.__version__))
# Check that TensorFlow was build with CUDA to use the gpus
print("Device name: {}".format(tf.test.gpu_device_name()))
print("Build with GPU Support? {}".format(tf.test.is_built_with_gpu_support()))
print("Build with CUDA? {} ".format(tf.test.is_built_with_cuda()))

# initialize model

# feature models are VGG16 or InceptionResNetV2 or 'efn'
model = initialize_model(weights_initial='imagenet',trainable=True,activation='relu',feature_model=InceptionResNetV2)

# load data
dir_path = 'images'
augment = True
batch_size = 5
epochs = 100

X_train,y1_train,y2_train,X_test,y1_test,y2_test,X_val,y1_val,y2_val = load_train_test_splits(dir_path, test_size=0.1,val_size=0.1,seed=0,normalize_y2=False,augment=augment)
# train
print('Staring First Training')
history = model.fit(X_train,[y1_train,y2_train],batch_size=batch_size,epochs=epochs,validation_data=(X_val,[y1_val,y2_val]), callbacks=[early_stop,red_on_plat])
# test 
print('Evaluation')
model.evaluate(X_test,[y1_test,y2_test])
# save model 

name = 'all_one_go'
model.save('{}_Model'.format(name))

with open('history_'+name, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# with open('history_first_'+name, 'wb') as file_pi:
#     pickle.dump(first_history.history, file_pi)

# with open('history_second_'+name, 'wb') as file_pi:
#     pickle.dump(second_history.history, file_pi)
