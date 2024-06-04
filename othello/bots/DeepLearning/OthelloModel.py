from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Reshape,
    GlobalAveragePooling2D,
    Dense,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPool2D, 
    add,
    LSTM,
    Dropout,
    Flatten,
    TimeDistributed
)
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

import numpy as np, os

class OthelloModel():
    def __init__(self, input_shape=(10, 10)):
        self.model_name='model_'+( 'x'.join(list(map(str, input_shape))) )+'.weights.h5'
        self.input_boards = Input(shape=input_shape)
        x_image = Reshape(input_shape+(1,))(self.input_boards)
        resnet_v12 = self.resnet_v1(inputs = x_image, num_res_blocks = 2)
        gap1 = GlobalAveragePooling2D()(resnet_v12)
        self.pi = Dense(input_shape[0]*input_shape[1], activation='softmax', name='pi')(gap1)
        self.model = Model(inputs=self.input_boards, outputs=[self.pi])
        self.model.compile(loss=['categorical_crossentropy'], optimizer=Adam(0.002))
        # print(self.model.summary())
        # self.model = Sequential()
        # self.model.add(LSTM(512, return_sequences=True, input_shape = input_shape))
        # self.model.add(LSTM(512, return_sequences=True))  
        # self.model.add(Dropout(0.2))
        # self.model.add(Activation('relu'))
        # self.model.add(LSTM(256, return_sequences=True))
        # self.model.add(LSTM(256, return_sequences=True))  
        # self.model.add(Dropout(0.2))
        # self.model.add(Activation('relu'))
        # self.model.add(LSTM(128, return_sequences=True))  
        # self.model.add(LSTM(128))  
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(input_shape[0]*input_shape[1], activation='softmax'))
        # self.model.compile(loss=['categorical_crossentropy'], optimizer=Adam(0.002))
        print(self.model.summary())
        
    def predict(self, board):
        return self.model.predict(np.array([board]).astype('float32'))[0]
    
    def fit(self, data, batch_size, epochs):
        input_boards, target_policys = list(zip(*data))
        input_boards = np.array(input_boards)
        target_policys = np.array(target_policys)
        history = self.model.fit(x = input_boards, y = [target_policys], batch_size = batch_size, epochs = epochs, validation_split=0.33)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('train loss vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','validation'], loc='upper right')
        plt.show()

    def set_weights(self, weights):
        self.model.set_weights(weights)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def save_weights(self):
        self.model.save_weights('othello/bots/DeepLearning/models/'+self.model_name)
    
    def load_weights(self):
        self.model.load_weights('othello/bots/DeepLearning/models/'+self.model_name)
    
    def reset(self, confirm=False):
        if not confirm:
            raise Exception('this operate would clear model weight, pass confirm=True if really sure')
        else:
            try:
                os.remove('othello/bots/DeepLearning/models/'+self.model_name)
            except:
                pass
        print('cleared')

    
    def resnet_v1(self, inputs, num_res_blocks):
        x = inputs
        x = MaxPool2D(strides = 2)(Conv2D(kernel_size=7, filters = 64, strides=2)(x))
        for  i in range(3):
            resnet = self.resnet_layer(inputs = x, num_filter = 64)    
            resnet = self.resnet_layer(inputs = resnet, num_filter = 64, activation = None)     
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet

        resnet = self.resnet_layer(inputs = x, num_filter = 128,strides=2)
        resnet = self.resnet_layer(inputs = resnet, num_filter = 128, activation = None)
        x = self.resnet_layer(inputs = x, num_filter = 128, strides=2)
        resnet = add([resnet, x])
        resnet = Activation('relu')(resnet)
        x = resnet

        for  i in range(3):
            resnet = self.resnet_layer(inputs = x, num_filter = 128)    
            resnet = self.resnet_layer(inputs = resnet, num_filter = 128, activation = None)     
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet

        resnet = self.resnet_layer(inputs = x, num_filter = 256,strides=2)
        resnet = self.resnet_layer(inputs = resnet, num_filter = 256, activation = None)
        x = self.resnet_layer(inputs = x, num_filter = 256, strides=2)
        resnet = add([resnet, x])
        resnet = Activation('relu')(resnet)
        x = resnet

        for i in range(5):
            resnet = self.resnet_layer(inputs = x, num_filter = 256)
            resnet = self.resnet_layer(inputs = resnet, num_filter = 256, activation = None)
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet

        resnet = self.resnet_layer(inputs = x, num_filter = 512,strides=2)
        resnet = self.resnet_layer(inputs = resnet, num_filter = 512, activation = None)
        x = self.resnet_layer(inputs = x, num_filter = 512, strides=2)
        resnet = add([resnet, x])
        resnet = Activation('relu')(resnet)
        x = resnet

        for i in range(2):
            resnet = self.resnet_layer(inputs = x, num_filter = 512)
            resnet = self.resnet_layer(inputs = resnet, num_filter = 512, activation = None)
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet

        return x


    def resnet_layer(self, inputs, num_filter = 16, kernel_size = 3, strides = 1, activation = 'relu', batch_normalization = True, conv_first = True, padding = 'same'):
        
        conv = Conv2D(num_filter, 
                    kernel_size = kernel_size, 
                    strides = strides, 
                    padding = padding,
                    use_bias = False,  
                    kernel_regularizer = l2(1e-4))
        
        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization(axis=3)(x)
            if activation is not None:
                x = Activation(activation)(x)
            
        else:
            if batch_normalization:
                x = BatchNormalization(axis=3)(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x      


        
