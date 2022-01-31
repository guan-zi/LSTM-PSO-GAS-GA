# 0131
# 出处：https://blog.csdn.net/Vertira/article/details/122403571

# 本章节GA_LSTM是关于遗传算法优化lstm算法的层数和全连接层数及每层神经元的个数
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as plt
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras import optimizers, losses, metrics, models, Sequential

'''
本文的主要内容如下：
1.本文章是对lstm网络的优化，优化的参数主要有：lstm层的层数，lstm隐藏层的神经元个数，dense层的层数，dense层的神经元个数
2.本文章利用的是遗传算法进行优化，其中编码形式并未采用2进制编码，只是将2数组之间的元素交换位置。
3.本文的lstm和dense的层数都在1-3的范围内，因为3层的网络足以拟合非线性数据
4.程序主要分为2部分，第一部分是lstm网络的设计，第二部分是遗传算法的优化。
5.代码的解释已详细写在对应的部分，有问题的同学可以在评论区进行交流
'''


# 导入数据集，本文用的是mnist手写数据集，该数据主要是对手写体进行识别0-9的数字
def load_data():
    # 从tensorflow自带的数据集中导入数据
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # 主要进行归一化操作
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, x_test, y_test, y_train


# 定义LSTM模型
def lstm_mode(inputs, units_num, sequences_state):
    # input主要是用来定义lstm的输入，input的一般是在第一层lstm层之前，units_num即是隐藏层神经元个数，sequence_state即是lstm层输出的方式
    lstm = LSTM(units_num, return_sequences=sequences_state)(inputs)
    print("lstm:", lstm.shape)
    return lstm


# 定义全连接层、BN层
def dense_mode(input, units_num):
    # 这里主要定义全连接层的输入，input参数定义dense的第一次输入，units_num代表隐藏层神经元个数
    # 这里定义全连接层，采用L2正则化来防止过拟合，激活函数为relu
    dense = Dense(units_num, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')(input)
    print("dense：", dense.shape)
    # 定义dropout层，概率为0.2
    drop_out = Dropout(rate=0.2)(dense)
    # 定义BN层，可以理解为是隐藏层的标准化过程
    dense_bn = BatchNormalization()(drop_out)
    return dense, drop_out, dense_bn


# 这里定义的即是评价lstm效果的函数——也是遗传算法的适应度函数
def aim_function(x_train, y_train, x_test, y_test, num):
    # 这里传入数据和参数数组num,num保存了需要优化的参数
    # 这里我们设置num数组中num[0]代表lstm的层数。
    lstm_layers = num[0]
    # num[2:2 + lstm_layers]分别为lstm各层的神经元个数，有同学不知道num(1)去哪了(num(1)为全连接层的层数)
    lstm_units = num[2:2 + lstm_layers]
    # 将num
    lstm_name = list(np.zeros((lstm_layers,)))
    # 设置全连接层的参数
    # num(1)为全连接的参数
    lstm_dense_layers = num[1]
    # 将lstm层之后的地方作为全连接层各层的参数
    lstm_dense_units = num[2 + lstm_layers: 2 + lstm_layers + lstm_dense_layers]
    #
    lstm_dense_name = list(np.zeros((lstm_dense_layers,)))
    lstm_dense_dropout_name = list(np.zeros((lstm_dense_layers,)))
    lstm_dense_batch_name = list(np.zeros((lstm_dense_layers,)))
    # 这主要是定义lstm的第一层输入，形状为训练集数据的形状
    inputs_lstm = Input(shape=(x_train.shape[1], x_train.shape[2]))

    # 这里定义lstm层的输入（如果为第一层lstm层，则将初始化的input输入，如果不是第一层，则接受上一层输出的结果）
    for i in range(lstm_layers):
        if i == 0:
            inputs = inputs_lstm
        else:
            inputs = lstm_name[i - 1]
        if i == lstm_layers - 1:
            sequences_state = False
        else:
            sequences_state = True
        # 通过循环，我们将每层lstm的参数都设计完成
        lstm_name[i] = lstm_mode(inputs, lstm_units[i], sequences_state=sequences_state)

    # 同理设计全连接层神经网络的参数
    for i in range(lstm_dense_layers):
        if i == 0:
            inputs = lstm_name[lstm_layers - 1]
        else:
            inputs = lstm_dense_name[i - 1]

        lstm_dense_name[i], lstm_dense_dropout_name[i], lstm_dense_batch_name[i] = dense_mode(inputs, units_num=
        lstm_dense_units[i])

    # 这里是最后一层：分类层，softmax
    outputs_lstm = Dense(10, activation='softmax')(lstm_dense_batch_name[lstm_dense_layers - 1])
    print("last_dense", outputs_lstm.shape)

    # 利用函数式调试神经网络，调用inputs和outputs之间的神经网络
    LSTM_model = tf.keras.Model(inputs=inputs_lstm, outputs=outputs_lstm)
    # 编译模型
    LSTM_model.compile(optimizer=optimizers.Adam(),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    print("训练集形状", x_train.shape)

    history = LSTM_model.fit(x_train, y_train, batch_size=32, epochs=1, validation_split=0.1, verbose=1)
    # 验证模型,model.evaluate返回的值是一个数组，其中score[0]为loss,score[1]为准确度
    acc = LSTM_model.evaluate(x_test, y_test, verbose=0)
    return acc[1]

