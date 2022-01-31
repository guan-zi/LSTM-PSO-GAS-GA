import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import random

minMax1 = MinMaxScaler()  # minmax
tpot_data = pd.read_excel('S2_date_6-6_new.xlsx')  # data load
tpot_data = tpot_data.dropna()  # delete NAN
tpot_data = tpot_data[:5000]  # delete NAN
print(len(tpot_data))
targcts = tpot_data['S2[MW]'].values
tpot_data["DATE"] = tpot_data["DATE"].astype(np.int64)/np.max(tpot_data["DATE"].astype(np.int64))*10
features = minMax1.fit_transform(tpot_data.drop('S2[MW]', axis=1))  # feature

# num = 18202
num = 4000
training_features, testing_features, training_target, testing_target = \
        features[:num], features[num:-1], targcts[:num], targcts[num:-1]
batch_start = 0
NN = 0
print(len(training_target))
print(len(testing_target))


def get_batch(input_size, batch_size, time_step, target, feature):
    global batch_start
    print("start:", batch_start)
    sxs = np.arange(batch_start, batch_start + time_step * batch_size).reshape((batch_size, time_step))
    # 构造数组 batch * steps = 100 * 10 = 1000行数据
    # sxs reshape (100batch, 10steps)
    # 循环过程每次输入10行数据，输入100次

    xs = feature[batch_start: batch_start+time_step*batch_size]
    ys = target[batch_start: batch_start+time_step*batch_size]
    # 构造数组 batch * 1steps = 10 * 100 = 1000行数据

    # print('时间段 =', batch_start, batch_start + time_step * batch_size)
    # 输出当前状态(起点位置, 终点位置)

    seq = xs.reshape((batch_size, time_step, input_size))
    # xs reshape (100batch, 10steps, len(input))
    res = ys.reshape((batch_size, time_step, 1))
    # ys reshape (100batch, 10steps, len(output))
    batch_start += time_step

    return [seq, res, sxs]
    # feature(batch, step, input), target(batch, step, output), aggregated data(batch, step, input + output)


def function(ps, test, le):
    ss = 100 - np.sum(np.abs(test - ps))/le
    return ss


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, LR):
        self.n_steps = int(n_steps)
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.cell_size = int(cell_size)  # 记忆单元维度
        self.batch_size = int(batch_size)
        if NN != 0:
            tf.reset_default_graph()
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self, ):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)  # 算法
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(   # 方程
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    def ms_error(self, labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


def training(input_size, output_size, x):
    # print(x)
    cell_size = int(x[0])
    lr = x[1]
    time_step = int(x[2])
    batch_size = int(x[3])
    sn = int(len(training_target)/time_step)-batch_size
    tn = int(len(testing_target)/time_step)-batch_size
    global batch_start
    model = LSTMRNN(time_step, input_size, output_size, cell_size, batch_size, lr)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(tf.global_variables_initializer())

    pred_res1 = []
    pred_res2 = []
    # 1 开始训练 200 次
    for tr in range(2):
        for bs in range(sn):
            # print(sn, batch_size, time_step)
            seq, res, sxs = get_batch(input_size, batch_size, time_step, training_target, training_features)
            # 提取 batch data
            if bs == 0:
                feed_dict = {model.xs: seq, model.ys: res, }  # 初始化 data
            else:
                feed_dict = {model.xs: seq, model.ys: res, model.cell_init_state: state}  # 保持 state 的连续性

            # 训练
            _, cost, state, prd1 = sess.run(
                [model.train_op, model.cost, model.cell_final_state, model.pred], feed_dict=feed_dict)

            result = sess.run(merged, feed_dict)
            writer.add_summary(result, tr)
            # print('training times = {:}, cost = {:} : '.format(tr, bs), round(cost, 4))
            # plt.plot(sxs[0, :], res[0].flatten(), 'r', sxs[0, :], prd1.flatten()[:time_step], 'b--')
            # plt.ion()  # 设置连续 plot
            # plt.draw()
            # plt.pause(0.00000000000000000000000003)  # 每 0.3 s 刷新
            # plt.clf()
        # 重新开始
        batch_start = 0

    # 画个图看看
    fig = plt.figure(figsize=(20, 3))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
    axes = fig.add_subplot(1, 1, 1)
    # plt.ion()
    # plt.show()

    # 2 开始预测
    for bs in range(tn):
        # print(tn, batch_size, time_step)
        seq, res, sxs = get_batch(input_size, batch_size, time_step, testing_target, testing_features)
        # 提取 batch data
        feed_dict = {model.xs: seq, model.ys: res, model.cell_init_state: state}  # 保持 state 的连续性
        # state 继承

        # 序列预测
        _, cost, state, prd2 = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred], feed_dict=feed_dict)

        plt.plot(sxs[0, :], res[0].flatten(), 'r', sxs[0, :], prd2.flatten()[:time_step], 'b--')
        pred_res1 = prd2
        pred_res2 = pred_res2 + list(pred_res1[:batch_start])
        # plt.ylim((-1.2, 1.2))
        # plt.draw()
        if bs == tn - 1:
            plt.plot(sxs.flatten()[time_step:], res.flatten()[time_step:], 'r', sxs.flatten()[time_step:],
                     prd2.flatten()[time_step:], 'b--')
        # plt.pause(0.00000000000000000000000003)  # 每 0.3 s 刷新
    # plt.pause(10)  # 每 0.3 s 刷新
    plt.show()
    pred_res2 = pred_res2 + list(pred_res1[batch_start:])
    batch_start = 0
    print("预测序列长度", len(pred_res2))
    return len(pred_res2), pred_res2


########################################################################
#                         PSO SEARCH START
########################################################################
# (1) data size
INPUT_SIZE = 36
OUTPUT_SIZE = 1

# (2) PSO Parameters
MAX_EPISODES = 20
MAX_EP_STEPS = 20
c1 = 2
c2 = 2
w = 0.5
pN = 20  # 粒子数量

# (3) LSTM Parameters
dim = 4  # 搜索维度
X = np.zeros((pN, dim))  # 所有粒子的位置和速度
V = np.zeros((pN, dim))
pbest = np.zeros((pN, dim))  # 个体经历的最佳位置和全局最佳位置
gbest = np.zeros(dim)
p_fit = np.zeros(pN)  # 每个个体的历史最佳适应值
t1 = time.time()

# CELL_SIZE, LR ,TIME_STEP, BATCH_SIZE
UP = [40, 0.1, 30, 120]
DOWN = [5, 0.0001, 2, 10]

# (4) 开始搜索
for i_episode in range(MAX_EPISODES):
    """初始化s"""
    random.seed(8)
    fit = -1e5  # 全局最佳适应值
    # 初始粒子适应度计算
    print("计算初始全局最优")
    for i in range(pN):
        for j in range(dim):
            V[i][j] = random.uniform(0, 1)
            if j == 1:
                X[i][j] = random.uniform(DOWN[j], UP[j])
            else:
                X[i][j] = round(random.randint(DOWN[j], UP[j]), 0)
        pbest[i] = X[i]
        le, pred = training(INPUT_SIZE, OUTPUT_SIZE, X[i])
        NN = 1
        tmp = function(pred, testing_target[:le], le)
        p_fit[i] = tmp
        if tmp > fit:
            fit = tmp
            gbest = X[i]
    print("初始全局最优参数：{:}".format(gbest))

    fitness = []  # 适应度函数
    for j in range(MAX_EP_STEPS):
        fit2 = []
        plt.title("第{}次迭代".format(i_episode))
        for i in range(pN):
            le, pred = training(INPUT_SIZE, OUTPUT_SIZE, X[i])
            temp = function(pred, testing_target[:le], le)
            fit2.append(temp/1000)
            if temp > p_fit[i]:  # 更新个体最优
                p_fit[i] = temp
                pbest[i] = X[i]
                if p_fit[i] > fit:  # 更新全局最优
                    gbest = X[i]
                    fit = p_fit[i]
        print("搜索步数：{:}".format(j))
        print("个体最优参数：{:}".format(pbest))
        print("全局最优参数：{:}".format(gbest))

        for i in range(pN):
            V[i] = w * V[i] + c1 * random.uniform(0, 1) * (pbest[i] - X[i]) + c2 * random.uniform(0, 1) * (gbest - X[i])
            ww = 1
            for k in range(dim):
                if DOWN[k] < X[i][k] + V[i][k] < UP[k]:
                    continue
                else:
                    ww = 0
            X[i] = X[i] + V[i]*ww

        fitness.append(fit)

print('Running time: ', time.time() - t1)


