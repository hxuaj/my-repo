# Batch Normalization
Dec 27, 2018

这是对论文[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)的阅读笔记，包括阅读过程中的推导以及遇到的问题。

## TLDR
先给出intuition，从两个方面来看为什么要用BN：
1. 整体上：神经网络学习过程本质是为了学习数据分布。如果训练数据和测试数据分布不同，则模型的泛化能力会很低。

    若每一个batch训练数据的分布不同，对于整个网络每一次迭代都要学习适应不同的分布，会降低网络的训练速度。
2. 局部上：对于网络内部，一旦网络的某一层输入分布发生改变，那么这一层网络就要去适应并学习新的数据分布。网络的前几层发生的微小改变会随着层数的增加而累积。此时需要更小的learning rate和更小心的参数初始化。所以训练过程中，训练数据的分布一直变化会影响网络拟合的速度。文章中称这样的现象为<em>internal covariate shift</em>。

## 使用BN的优点
在网络中使用BN，从而允许更高的学习速率和更粗放的初始化；某种程度上达到正则化的效果，降低对Dropout的需求。

## BN层的引入
为了达到提高训练的目的，需要减少internal covariate shift的影响。通过白化(Whitening)操作线性地使其均值为0方差为1并去相关，可以达到固定输入分布加速网络拟合的效果。

值得注意的是，在梯度下降的优化过程中需要考虑到normalization。文中举了一个例子：

$\hat{x}$
