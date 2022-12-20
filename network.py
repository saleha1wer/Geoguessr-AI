
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense,Concatenate, Dropout
from tensorflow.keras import Model,Input
import numpy as np
import PIL.Image as Image
import tensorflow as tf
import os 


def initialize_model(inp_shape=(400,1200,3),out_1_shape=(88),out_2_shape=(2),weights_initial='imagenet',trainable=True,activation=None,feature_model=InceptionResNetV2):
    print('Initialising Network')
    input_layer = Input(shape=inp_shape)
    
     
    feature_extraction = feature_model(include_top=False,weights=weights_initial,input_shape=inp_shape,pooling='avg',input_tensor=input_layer)
    feature_extraction.trainable = trainable

    embed = feature_extraction.output
    out_1 = Dense(out_1_shape,activation='softmax',name='Grid')(embed)
    
    # dense_1 = Dense(88,name='map_grid')(out_1)

    concat = Concatenate()([embed,out_1])
    dense_2 =  Dense(250,name='mid_layer')(concat)
    drop = Dropout(0.5)(dense_2)
    dense_3 =  Dense(200,name='mid_layer_1')(drop)
    dense_4 =  Dense(150,name='mid_layer_2')(dense_3)
    if activation is not None:
        dense_4 = tf.nn.leaky_relu(dense_4, alpha=0.25)

    dense_7 =  Dense(100,kernel_regularizer='l2',name='mid_layer_3')(dense_4)
    dense_8 =  Dense(80,kernel_regularizer='l2',name='mid_layer_4')(dense_7)
    drop = Dropout(0.5)(dense_8)
    dense_11 = Dense(50,kernel_regularizer='l2',name='mid_layer_5')(drop)
    out_2 =  Dense(out_2_shape,activation='linear',name='coordinates')(dense_11)


    if feature_model == 'efn':
        model = Model(inputs=feature_extraction.input,outputs=[out_1,out_2])
    else:
        model = Model(inputs=input_layer,outputs=[out_1,out_2])

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


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),loss=[tf.keras.metrics.categorical_crossentropy,'mae'],metrics=['accuracy',km_away])
    model.summary()
    print('Network Initialised and compiled. Input shape: {}, Output shape: {}'.format(model.input_shape,model.output_shape))
    print('Loss: Categorical Cross Entropy and MSE')
    return model

