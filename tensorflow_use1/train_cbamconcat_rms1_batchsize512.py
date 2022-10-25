
# %%
from ast import arg
import numpy as np

import random

import tensorflow as tf

import os



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.keras import optimizers
from scipy import io
from tensorflow.keras.models import load_model, Model
import h5py
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint

from cbam_model_concat import model_CBAM



# max_data = io.loadmat('D:/A_DLinverse/bigdataset/Fix_m_81C/1.45+0i/speak_Nnoise_and_6noise3.mat')
max_data = io.loadmat('./DataSet/speak_Nnoise_and_6noise3.mat')
Q_data = io.loadmat('./Q_MATRIX50DEHS130.mat')
Q_data = Q_data['Q_matrix']
Q_data = ops.convert_to_tensor(Q_data)  
Q_data = tf.linalg.matrix_transpose(Q_data)
Q_data = tf.cast(Q_data, tf.float32)

KL_loss = []
RMS_loss = []

def KL_list_function(x):
    print(x)
    x = np.sum(x)
    KL_loss.append(x)
def RMS_list_function(x):
    print(x)
    x = np.sum(x)
    RMS_loss.append(x)

def my_loss(y_true, y_pred):
    

    y = tf.constant(1e-50, shape=(81,))
    y_pred = tf.where(y_pred == 0, y, y_pred)
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    
    p_pred = tf.matmul(y_pred, Q_data)
    p_true = tf.matmul(y_true, Q_data)
    
    # p_true = p_true.numpy()
    # p_pred = p_pred.numpy()

    p_true_sum = tf.reduce_sum(p_true, axis=1)
    p_true_sum = tf.reshape(p_true_sum, (-1, 1))
    p_true = p_true / p_true_sum

    p_pred_sum = tf.reduce_sum(p_pred, axis=1)
    p_pred_sum = tf.reshape(p_pred_sum, (-1, 1))
    p_pred = p_pred / p_pred_sum



    p_true = ops.convert_to_tensor_v2(p_true)
    p_true = math_ops.cast(p_true, y_pred.dtype)
    p_true = K.clip(p_true, K.epsilon(), 1)
    
    p_pred = ops.convert_to_tensor_v2(p_pred)
    p_pred = math_ops.cast(p_pred, y_pred.dtype)
    p_pred = K.clip(p_pred, K.epsilon(), 1)
    
    tf.print("KL:", math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1))
    tf.print("RMS:", tf.sqrt(tf.reduce_mean(tf.square(p_pred-p_true), axis=-1)) / tf.sqrt(tf.reduce_mean(tf.square(p_true), axis=-1)))
    KL_lo = math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)
    RMS_lo = tf.sqrt(tf.reduce_mean(tf.square(p_pred-p_true), axis=-1)) / tf.sqrt(tf.reduce_mean(tf.square(p_true), axis=-1))
    tf.numpy_function(KL_list_function, [KL_lo], [])
    tf.numpy_function(RMS_list_function, [RMS_lo], [])
 

    return KL_lo + RMS_lo


def Euclidean(y_true, y_pred):
    
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true)))


def main():
    # %%
    
    
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)

    p = max_data['p'][:4920]
    psd = max_data['psd'][:4920]


    TRAIN_SAMPLE_LEN = int(p.shape[0] * 0.6)
    VAL_SAMPLE_LEN = int(psd.shape[0] * 0.3)

    # psd归一化
    psd_sum = np.sum(psd, axis=1).reshape(-1, 1)
    psd = psd / psd_sum
    psd[psd == 0] = 1e-50
    # %%
    p_train, p_m, psd_train, psd_m = train_test_split(p, psd,
                                                      random_state=66,
                                                      test_size=0.4,
                                                      shuffle=True)

    p_test, p_val, psd_test, psd_val = train_test_split(p_m, psd_m,
                                                        random_state=88,
                                                        test_size=0.75,
                                                        shuffle=True)
    
    def train_data_generator(p_train, psd_train, sample_len, batch_size):
        while True:
            for index in range(0, sample_len, batch_size):
                p = p_train[index: index + batch_size]
                psd = psd_train[index: index + batch_size]
                c = np.column_stack((p, psd))  # 将y添加到x的最后一列
                np.random.shuffle(c)
                p = c[:, :81]  # 乱序后的x
                psd = c[:, 81:]  # 同等乱序后的y
            
                
                noise5 = 0.1 * np.random.rand(int(p.shape[0] * 0.8), 81) + 0.95
                ones = np.ones((p.shape[0]-int(p.shape[0] * 0.8), 81))
                weights = np.concatenate((ones, noise5))

                p = p * weights

                p_sum = np.sum(p, axis=1).reshape(-1, 1)
                p = p / p_sum

                
                p = np.expand_dims(p[:, 0:81].astype(float), axis=2)


                yield np.array(p), np.array(psd)

    def val_data_generator(p_val, psd_val, sample_len, batch_size):
        while True:
            for index in range(0, sample_len, batch_size):
                p = p_val[index: index + batch_size]
                psd = psd_val[index: index + batch_size]

                c = np.column_stack((p, psd))  # 将y添加到x的最后一列
                np.random.shuffle(c)
                p = c[:, :81]  # 乱序后的x
                psd = c[:, 81:]  # 同等乱序后的y
        
                noise5 = 0.1 * np.random.rand(int(p.shape[0] * 0.8), 81) + 0.95
                ones = np.ones((p.shape[0] - int(p.shape[0] * 0.8), 81))
                weights = np.concatenate((ones, noise5))

                p = p * weights
                p_sum = np.sum(p, axis=1).reshape(-1, 1)
                p = p / p_sum

                
                p = np.expand_dims(p[:, 0:81].astype(float), axis=2)
                
                yield np.array(p), np.array(psd)

    def my_loss1(y_true, y_pred):
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.abs(K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)))

  
    model = model_CBAM()
    
 

    model_name = 'loop_noise5_cbamconcat_rms1_batchsize512'

    log_dir = './logs/' + model_name + '/'
    model_dir = './model/' + model_name + '.h5'
    predict_dir = './test_data_predict/' + model_name + '.mat'
    
    print(model.summary())
    opt = optimizers.Adam(learning_rate=0.001)
    model.compile(loss=my_loss,
                  optimizer=opt,
                  metrics=[Euclidean, 'cosine_similarity'])
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, min_delta=1e-5, patience=40, cooldown=0, min_lr=1e-12)
    checkpoint = ModelCheckpoint(
        filepath=model_dir,
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        save_weights_only=True,
        period=20
    )
    # early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=10, mode='min')

    BATCH_SIZE = 512
    EPOCHS = 3

    model.fit(train_data_generator(p_train, psd_train, TRAIN_SAMPLE_LEN, BATCH_SIZE),
              steps_per_epoch=TRAIN_SAMPLE_LEN // BATCH_SIZE, epochs=EPOCHS, verbose=1,
              validation_data=val_data_generator(p_val, psd_val, VAL_SAMPLE_LEN, BATCH_SIZE),
              validation_steps=VAL_SAMPLE_LEN // BATCH_SIZE,
              callbacks=[tensorboard_callback, rlr, checkpoint])

    model.save_weights(model_dir)
    # model.save_weights(weights_dir)

    # test_data_predict
    # 测试集数据归一化

    noise1 = 0.02 * np.random.rand(int(p_test.shape[0]), 81) + 0.99
    noise3 = 0.06 * np.random.rand(int(p_test.shape[0]), 81) + 0.97
    noise5 = 0.1 * np.random.rand(int(p_test.shape[0]), 81) + 0.95
    ones = np.ones((p_test.shape[0], 81))
    weights = np.concatenate((ones, noise1, noise3, noise5))
    p_test = np.concatenate((p_test, p_test, p_test, p_test))
    p_test = p_test * weights
    p_test_sum = np.sum(p_test, axis=1).reshape(-1, 1)
    p_test = p_test / p_test_sum

    p_test = np.expand_dims(p_test[:, 0:81].astype(float), axis=2)

    psd_pre = model.predict(p_test)

    io.savemat(predict_dir, {'psd_test': psd_test, 'psd_pre': psd_pre})


if __name__ == '__main__':
    main()
    print('KL: ', KL_loss)
    print('RMS: ', RMS_loss)
