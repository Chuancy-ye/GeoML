import numpy as np


# 创建神经网络类
# nodes为1*n矩阵，表示每一层有多少个节点，例如[3,4,5]
# 表示三层，第一层有3个节点，第二层4个，第三层5个

class nn():
    def __init__(self, nodes):  # 输入为神经网络结构
        self.layers = len(nodes)
        self.nodes = nodes
        self.u = 1.0  # 学习率
        self.W = list()  # 权值
        self.B = list()  # 偏差值
        self.P = list()
        self.values = list()  # 层值
        self.error = 0  # 误差
        self.loss = 0  # 损失
        self.sparse = 0.2
        self.beta = 2.0
        self.denoise = 0.2

        for i in range(self.layers - 1):
            # 权值初始化，权重范围-0.5~0.5
            self.W.append(np.random.random((self.nodes[i],
                                            self.nodes[i + 1])) - 0.5)

            # 偏差B值初始化
            self.B.append(0)
            self.P.append(0)

        for j in range(self.layers):
            # 节点values值初始化
            self.values.append(0)


# 创建autoencoder类，可以看成是多个神经网络简单的堆叠而来
class autoencoder():
    def __init__(self):
        self.encoders = list()

    def add_one(self, nn):
        self.encoders.append(nn)
