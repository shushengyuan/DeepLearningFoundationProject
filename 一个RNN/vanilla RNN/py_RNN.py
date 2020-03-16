#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/13 下午8:19
# @Author  : YuanYun
# @Site    : 
# @File    : py_RNN.py
# @Software: PyCharm

"""
文件说明：
    由 python 写成的最基本RNN。
    我们利用这个RNN来计算十进制加法
工具包：
    numpy
"""

import copy
import numpy as np

np.random.seed(0)


# 数学工具函数
######################################################################

def sigmoid(x):
    """
    ：激活函数sigmoid
    :param x:
    :return: 1/(1 + e^(-x))
    """
    output = 1 / (1 + np.exp(-x))
    return output


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    """
    :输出sigmoid的导数
    :param output: 函数的输出
    :return:       output的导数
    """
    return output * (1 - output)


##########################################################################

# 训练数据准备
##########################################################################
# training dataset prepare
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
'''
unpackbits函数可以把整数转化成2进制数

a = np.array([[2], [7], [23]], dtype=np.uint8)
b = np.unpackbits(a, axis=1)
print("a",a)
print("b",b)

输出结果：

a   [[ 2]
    [7]
    [23]]
b   [[0 0 0 0 0 0 1 0]
    [0 0 0 0 0 1 1 1]
    [0 0 0 1 0 1 1 1]]
'''
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)

# 建立int和二进制一一对应的字典
for i in range(largest_number):
    int2binary[i] = binary[i]


##########################################################################

# 定义RNN类
##########################################################################
class RNN:
    def __init__(self):
        # 设置维度
        self.alpha = 0.1
        self.input_dim = 2
        self.hidden_dim = 16
        self.output_dim = 1
        # 设置权重
        self.weight_0 = 2 * np.random.random((self.input_dim, self.hidden_dim)) - 1
        self.weight_h = 2 * np.random.random((self.hidden_dim, self.hidden_dim)) - 1
        self.weight_1 = 2 * np.random.random((self.hidden_dim, self.output_dim)) - 1

        '''
        zeros_like:返回与指定数组具有相同形状和数据类型的数组，并且数组中的值都为0。
        '''
        self.weight_0_update = np.zeros_like(self.weight_0)
        self.weight_h_update = np.zeros_like(self.weight_h)
        self.weight_1_update = np.zeros_like(self.weight_1)

    ######################################################################

    # 开始训练
    ######################################################################

    def train(self):
        for j in range(10000):
            # 提取数据样例 (a + b = c)
            a_int = np.random.randint(largest_number / 2)
            a = int2binary[a_int]  # 换成二进制

            b_int = np.random.randint(largest_number / 2)
            b = int2binary[b_int]  # 换成二进制

            # c为正确结果
            c_int = a_int + b_int
            c = int2binary[c_int]

            # d为将要预测的结果
            d = np.zeros_like(c)

            overallError = 0
            layer_2_deltas = list()
            layer_1_values = list()
            # hidden init
            '''
            np.zeros:返回来一个给定形状和类型的用0填充的数组；
            '''
            layer_1_values.append(np.zeros(self.hidden_dim))
            overall_error = self.predict(overallError, layer_1_values, layer_2_deltas, a, b, c, d)
            if j % 1000 == 0:
                self.test(overall_error, c, a_int, b_int, d)

    # 预测结果
    ######################################################################
    def predict(self, overall_error, layer_1_values, layer_2_deltas, a, b, c, d):
        for position in range(binary_dim):
            # 将输入每一位输入到RNN中
            X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
            y = np.array([[c[binary_dim - position - 1]]]).T

            # hidden layer (input ~+ prev_hidden) [-1 ] means take the last unit
            '''
            np.dot():矩阵乘法
            '''
            layer_1 = sigmoid(np.dot(X, self.weight_0) + np.dot(layer_1_values[-1], self.weight_h))
            # [-1]代表每次取上一次的hidden layer参数，RNN需要利用的历史信息就从中而来
            layer_2 = sigmoid(np.dot(layer_1, self.weight_1))

            # 计算误差
            layer_2_error = y - layer_2
            layer_2_deltas.append(layer_2_error * sigmoid_output_to_derivative(layer_2))
            overall_error += np.abs(layer_2_error[0])  # 输出为一维所以是[0],类型为ndarray

            # 将输出存储在d中
            '''
            round:四舍五入
            '''
            d[binary_dim - position - 1] = np.round(layer_2[0][0])  # 类型为float

            # 存储hidden layer的值，以便于下次利用
            '''
            copy.deepcopy: 改变原有被复制对象不会对已经复制出来的新对象产生影响。 
            '''
            layer_1_values.append(copy.deepcopy(layer_1))

        future_layer_1_delta = np.zeros(self.hidden_dim)
        self.loss(layer_1_values, layer_2_deltas, a, b, future_layer_1_delta)
        return overall_error

    ######################################################################

    # 计算误差
    ######################################################################
    def loss(self, layer_1_values, layer_2_deltas, a, b, future_layer_1_delta):

        for position in range(binary_dim):
            X = np.array([[a[position], b[position]]])
            layer_1 = layer_1_values[-position - 1]
            prev_layer_1 = layer_1_values[-position - 2]

            # 输出层的误差
            layer_2_delta = layer_2_deltas[-position - 1]
            # 隐藏层的误差
            layer_1_delta = (future_layer_1_delta.dot(self.weight_h.T) + layer_2_delta.dot(
                self.weight_1.T)) * sigmoid_output_to_derivative(layer_1)

            # 准备更新权重
            self.weight_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
            self.weight_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
            self.weight_0_update += X.T.dot(layer_1_delta)

            future_layer_1_delta = layer_1_delta
        self.backpropagation()

    ######################################################################

    # 反向传播
    ######################################################################
    def backpropagation(self):
        self.weight_0 += self.weight_0_update * self.alpha
        self.weight_1 += self.weight_1_update * self.alpha
        self.weight_h += self.weight_h_update * self.alpha

        self.weight_0_update *= 0
        self.weight_1_update *= 0
        self.weight_h_update *= 0

    ######################################################################

    # 输出测试结果
    ######################################################################
    @staticmethod
    def test(overall_error, c, a_int, b_int, d):

        print("Error:" + str(overall_error))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")


##########################################################################

# 主函数
if __name__ == '__main__':
    nn = RNN()
    nn.train()
