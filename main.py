from network import initialize_model
from load_data import load_train_test_splits
import numpy as np
import tensorflow as tf
import pickle 
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler

def degrees_to_radians(deg):
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180
def km_away(observation, prediction):    
    obv_rad = tf.map_fn(degrees_to_radians, observation)
    prev_rad = tf.map_fn(degrees_to_radians, prediction)

    dlon_dlat = obv_rad - prev_rad 
    v = dlon_dlat / 2
    v = tf.sin(v)
    v = v**2
    a = v[:,1] + tf.cos(obv_rad[:,1]) * tf.cos(prev_rad[:,1]) * v[:,0] 

    c = tf.sqrt(a)
    c = 2* tf.math.asin(c)
    c = c*6378.1
    final = tf.reduce_sum(c)
    #if you're interested in having MAE with the haversine distance in KM
    #uncomment the following line
    final = final/tf.dtypes.cast(tf.shape(observation)[0], dtype= tf.float32)

    return final

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

callback = tf.keras.callbacks.EarlyStopping(
monitor='val_loss',
min_delta=0,
patience=10,
verbose=1,
mode='auto',
baseline=None,
restore_best_weights=False
)

lr_sched = step_decay_schedule(initial_lr=3e-3, decay_factor=0.9, step_size=5)

print("TensorFlow version: {}".format(tf.__version__))
# Check that TensorFlow was build with CUDA to use the gpus
print("Device name: {}".format(tf.test.gpu_device_name()))
print("Build with GPU Support? {}".format(tf.test.is_built_with_gpu_support()))
print("Build with CUDA? {} ".format(tf.test.is_built_with_cuda()))

# initialize model

model = initialize_model(weights_initial='imagenet',trainable=True,activation='relu')

# load data
dir_path = 'images'
X_train,y1_train,y2_train,X_test,y1_test,y2_test,X_val,y1_val,y2_val = load_train_test_splits(dir_path, test_size=0.1,val_size=0.1,seed=0,normalize_y2=False)


for layer in model.layers:
    # if layer.trainable:
    #     trainables.append(layer.name)
    # else:
    #     non_trainables.append(layer.name)
    if layer.name in ['map_grid','coordinates'] or 'mid_layer' in layer.name :
        layer.trainable = False

# train
model.summary()
print('Staring First Training')
first_history = model.fit(X_train,[y1_train,y2_train],batch_size=5,epochs=30,validation_data=(X_val,[y1_val,y2_val]), callbacks=[callback])

print('Pretaining evaluation')
model.evaluate(X_test,[y1_test,y2_test])

for layer in model.layers:
    if layer.name in ['map_grid','coordinates'] or 'mid_layer' in layer.name:
        layer.trainable = True
    # else:
    #     layer.trainable = False

model.summary()
print('Staring Middle Training')
both_history = model.fit(X_train,[y1_train,y2_train],batch_size=5,epochs=70,validation_data=(X_val,[y1_val,y2_val]), callbacks=[callback])
for layer in model.layers:
    if not (layer.name in ['map_grid','coordinates'] or 'mid_layer' in layer.name):
        layer.trainable = False

print('Evaluation before freezing')
model.evaluate(X_test,[y1_test,y2_test])
model.save('Model_simple_nofinetune')
model.summary()
print('Staring Last Training')
second_history = model.fit(X_train,[y1_train,y2_train],batch_size=5,epochs=25,validation_data=(X_val,[y1_val,y2_val]), callbacks=[callback])
# test 
print('Evaluation')
model.evaluate(X_test,[y1_test,y2_test])
# save model 

model.save('Model_deep')

with open('history_first', 'wb') as file_pi:
    pickle.dump(first_history.history, file_pi)

with open('history_both', 'wb') as file_pi:
    pickle.dump(both_history.history, file_pi)

with open('history_second', 'wb') as file_pi:
    pickle.dump(second_history.history, file_pi)
