# GA优化lstm的遗传算法部分
import GA_LSTM_lstm as ga
import numpy as np
import pandas as pd
import matplotlib as plt
import os

# 不显示警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置遗传算法的参数
DNA_size = 2
DNA_size_max = 8  # 每条染色体的长度
POP_size = 2  # 种群数量
CROSS_RATE = 0.5  # 交叉率
MUTATION_RATE = 0.01  # 变异率
N_GENERATIONS = 2  # 迭代次数

# 接收数据
x_train, x_test, y_test, y_train = ga.load_data()


# 定义适用度函数，即aim_function函数，接收返回值
def get_fitness(x):
    return ga.aim_function(x_train, y_train, x_test, y_test, num=x)


# 生成新的种群
def select(pop, fitness):
    # 这里主要是进行选择操作，即从20个种群中随机选取重复随机采样出20个种群进行种群初始化操作，p代表被选择的概率，这里采用的是轮盘赌的方式
    idx = np.random.choice(np.arange(POP_size), size=POP_size, replace=True, p=fitness / fitness.sum())
    # 将选择的种群组成初始种群pop
    return pop[idx]


# 交叉函数
def crossover(parent, pop):
    # 这里主要进行交叉操作，随机数小于交叉概率则发生交叉
    if np.random.rand() < CROSS_RATE:
        # 从20个种群中选择一个种群进行交叉
        i_ = np.random.randint(0, POP_size, size=1)  # 染色体的序号
        # 这里将生成一个8维的2进制数，并转换层成bool类型，true表示该位置交叉，False表示不交叉
        cross_points = np.random.randint(0, 2, size=DNA_size_max).astype(np.bool)  # 用True、False表示是否置换

        # 这一部分主要是对针对不做变异的部分
        for i, point in enumerate(cross_points):
            '''
            第一部分：这里是指该位点为神经元个数的位点，本来该交换，但其中位点为0,
            什么意思呢？即[2,3,32,43,34,230,43,46,67]和[2,2,32,54,55,76,74,26,0],末尾的0位置就
            不应该交叉，因为交叉完后,会对前两位的参数产生影响。

            第二部分：即对前两位不进行交叉操作，因为前两位代表的是层数，层数交叉后会对神经元的个数产生影响
            '''
            # 第一部分
            if point == True and pop[i_, i] * parent[i] == 0:
                cross_points[i] = False
            # 第二部分
            if point == True and i < 2:
                cross_points[i] = False
        # 将第i_条染色体上对应位置的基因置换到parent染色体上
        parent[cross_points] = pop[i_, cross_points]
    return parent


# 定义变异函数
def mutate(child):
    # 变异操作也只是针对后6位参数
    for point in range(DNA_size_max):
        if np.random.rand() < MUTATION_RATE:
            # 2位参数之后的参数才才参与变异
            if point >= 2:
                if child[point] != 0:
                    child[point] = np.random.randint(32, 257)
    return child


# 初始化2列层数参数
pop_layers = np.zeros((POP_size, DNA_size), np.int32)
pop_layers[:, 0] = np.random.randint(1, 4, size=(POP_size,))
pop_layers[:, 1] = np.random.randint(1, 4, size=(POP_size,))

# 种群
# 初始化20x8的种群
pop = np.zeros((POP_size, DNA_size_max))
# 将初始化的种群赋值，前两列为层数参数，后6列为神经元个数参数
for i in range(POP_size):
    # 随机从[32,256]中抽取随机数组组成神经元个数信息
    pop_neurons = np.random.randint(32, 257, size=(pop_layers[i].sum(),))
    # 将2列层数信息和6列神经元个数信息合并乘8维种群信息
    pop_stack = np.hstack((pop_layers[i], pop_neurons))
    # 将这些信息赋值给pop种群进行初始化种群
    for j, gene in enumerate(pop_stack):
        pop[i][j] = gene

# 在迭代次数内，计算种群的适应度函数
for each_generation in range(N_GENERATIONS):
    # 初始化适应度
    fitness = np.zeros([POP_size, ])
    # 遍历20个种群，对基因进行操作
    for i in range(POP_size):
        pop_list = list(pop[i])
        # 第i个染色体上的基因
        # 对赋值为0的基因进行删除
        for j, each in enumerate(pop_list):
            if each == 0.0:
                index = j
                pop_list = pop_list[:j]
        # 将基因进行转换为int类型
        for k, each in enumerate(pop_list):
            each_int = int(each)
            pop_list[k] = each_int
        # 将计算出来的适应度填写在适应度数组中
        fitness[i] = get_fitness(pop_list)
        # 输出结果
        print('第%d代第%d个染色体的适应度为%f' % (each_generation + 1, i + 1, fitness[i]))
        print('此染色体为：', pop_list)
    print('Generation:', each_generation + 1, 'Most fitted DNA:', pop[np.argmax(fitness), :], '适应度为：',
          fitness[np.argmax(fitness)])

    # 生成新的种群
    pop = select(pop, fitness)

    # 复制一遍种群
    pop_copy = pop.copy()
    # 遍历pop中的每一个种群，进行交叉，变异，遗传操作
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent = child
