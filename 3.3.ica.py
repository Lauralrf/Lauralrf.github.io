# -*- encoding:utf-8 -*-
import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import fastica
from copy import deepcopy


class ICA():
    def __init__(self, signalRange, timeRange):
        # 信号幅度
        self.signalRange = signalRange
        # 时间范围
        self.timeRange = timeRange
        # 固定每个单位时间10个点,产生x
        self.x = np.arange(0, self.timeRange, 0.1)
        # 所有的点数
        self.pointNumber = self.x.shape[0]

    # 生成正弦波
    def produceSin(self, period=100, drawable=False):
        y = self.signalRange * np.sin(self.x / period * 2 * np.pi)
        if drawable:
            plt.plot(self.x, y)
            plt.show()

        return y

    # 生成方波
    def produceRect(self, period=20, drawable=False):
        y = np.ones(self.pointNumber) * self.signalRange
        begin = 0
        end = self.pointNumber
        step = period * 10
        mid = step // 2
        while begin < end:
            y[begin:mid] = -1 * self.signalRange
            begin += step
            mid += step
        if drawable:
            plt.plot(self.x, y)
            plt.show()

        return y

    # 生成三角波
    def produceAngle(self, period=20, drawable=False):
        lastPoint = period * 10 - 1
        if lastPoint >= self.pointNumber:
            raise ValueError('You must keep at least one period!')
        delta = ((-1 - 1) * self.signalRange) / float(self.x[lastPoint] - self.x[0])
        y = (self.x[:lastPoint+1] - self.x[0]) * delta + self.signalRange
        y = np.tile(y, self.pointNumber // y.shape[0])[:self.pointNumber]

        if drawable:
            plt.plot(self.x, y)
            plt.show()
        return y

    # 生成uniform噪声
    def produceNoise(self, signalRange=None, drawable=False):
        if signalRange is None:
            signalRange = self.signalRange
        y = np.asarray([(np.random.random() - 0.5) * 2 * signalRange for _ in range(self.pointNumber)])

        if drawable:
            plt.plot(self.x, y)
            plt.show()

        return y

    # 混合信号
    def mixSignal(self,majorSignal, *noises, **kwargs):
        mixSig = deepcopy(majorSignal)
        noiseRange = 1.0
        if 'noiseRange' in kwargs and kwargs['noiseRange']:
            noiseRange = kwargs['noiseRange']
        for noise in noises:
            mixSig += noiseRange * np.random.random() * noise
        if 'drawable' in kwargs and kwargs['drawable']:
            plt.plot(self.x, mixSig)
            plt.show()

        return mixSig

    # 让每个信号的样本均值为0,且协方差(各个信号之间)为单位阵
    def whiten(self, X):
        X = X - X.mean(-1)[:, None]
        A = np.dot(X, X.T)
        D, U = np.linalg.eig(A)
        D = np.diag(D)
        D_inv = np.linalg.inv(D)
        D_half = np.sqrt(D_inv)

        W0 = np.dot(D_half, U.T)

        return np.dot(W0, X), W0

    def _tanh(self, x):
        gx = np.tanh(x)
        # gx = (1 - np.exp(x)) / (1 + np.exp(x))
        g_x = gx ** 2
        g_x -= 1
        g_x *= -1
        return gx, g_x.mean(-1)

    def _exp(self, x):
        exp = np.exp(-(x ** 2) / 2)
        gx = x * exp
        g_x = (1 - x ** 2) * exp
        return gx, g_x.mean(axis=-1)


    def _cube(self, x):
        return x ** 3, (3 * x ** 2).mean(axis=-1)

    # W <- (W_1 * W_1')^(-1/2) * W_1
    def decorrelation(self, W):
        U, S = np.linalg.eigh(np.dot(W, W.T))
        U = np.diag(U)
        U_inv = np.linalg.inv(U)
        U_half = np.sqrt(U_inv)
        # rebuild_W = np.dot(np.dot(S * 1. / np.sqrt(U), S.T), W)
        rebuild_W = np.dot(np.dot(np.dot(S, U_half), S.T), W)
        return rebuild_W

    # fastICA
    def fastICA(self, X):
        n, m = X.shape
        p = float(m)
        X *= np.sqrt(X.shape[1])

        # 随机化W,只要保证非奇异即可
        W = np.ones((n,n), np.float32)
        for i in range(n):
            for j in range(i):
                W[i,j] = np.random.random()
        # 迭代计算W
        maxIter = 300
        for ii in range(maxIter):
            gwtx, g_wtx = self._exp(np.dot(W, X))
            W1 = self.decorrelation(np.dot(gwtx, X.T) / p - g_wtx[:, None] * W)
            lim = max(abs(abs(np.diag(np.dot(W1, W.T))) - 1))
            W = W1
            if lim < 0.00001:
                break
        return W

    # 画图
    def draw(self, y, figNum):
        if y.__class__ == list:
            m = len(y)
            n = 0
            if m > 0:
                n = len(y[0])
        elif y.__class__ == np.array([]).__class__:
            m, n = y.shape
        else:
            raise ValueError('The first arg you give must be type of list or np.array.')
        plt.figure(figNum)
        for i in range(m):
            plt.subplot(m, 1, i + 1)
            plt.plot(self.x, y[i])

    # 显示
    def show(self):
        plt.show()



if __name__ == '__main__':
    # 设置信号幅度为2,时间范围为[0, 200)
    ica = ICA(2, 200)
    # 周期为100的正弦波
    gsigSin = ica.produceSin(100, False)
    # 周期为20的方形波
    gsigRect = ica.produceRect(20, False)
    # 周期为20的三角波
    gsigAngle = ica.produceAngle(20, False)
    # 幅度为0.5的uniform噪声
    gsigNoise = ica.produceNoise(0.5, False)
    # 独立信号S
    totalSig = [gsigSin, gsigRect, gsigAngle, gsigNoise]
    # 混合信号X
    mixSig = []
    for i, majorSig in enumerate(totalSig):
        curSig = ica.mixSignal(majorSig, *(totalSig[:i] + totalSig[i+1:]), drawable=False)
        mixSig.append(curSig)
    mixSig = np.asarray(mixSig)


    # 以下是调用自己写的fastICA
    xWhiten, V = ica.whiten(mixSig)
    W = ica.fastICA(xWhiten)
    recoverSig = np.dot(np.dot(W, V), mixSig)
    ica.draw(totalSig, 1)
    ica.draw(mixSig, 2)
    ica.draw(recoverSig, 3)
    ica.show()

    # 以下是调用sklearn包里面的fastICA
    # V对应白化处理的变换矩阵即Z = V * X, W对应S = W * Z
    V, W, S = fastica(mixSig.T)
    assert ((np.dot(np.dot(W, V), mixSig) - S.T) < 0.001).all()
    ica.draw(totalSig, 1)
    ica.draw(mixSig, 2)
    ica.draw(S.T, 3)
    ica.show()








