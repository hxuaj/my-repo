# Batch Normalization Notebook
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
某一层输入为$u$和一个可学习的偏置$b$，activation为$x = u+b$，减去mean值对其标准化$\hat{x}=x-\operatorname E[x]$。如果不考虑$\operatorname E[x]$和$b$的依赖关系，则通过优化可得到$\Delta b\propto{-\partial{l}/\partial{\hat{x}}}$。以$b \leftarrow b + \Delta b$作为输出，在输出前对其标准化。$\Delta b$会因为减去mean值而被抵消。最终输出没有发生改变，网络并没有得到训练。
这里作者想表达的意思是在需要注意BN的部署以及和优化过程的关系。在梯度下降的优化过程中需要考虑到BN操作的进行。

为解决这一问题，作者提出对于任何参数，我们需要保证网络产生的激活总是有相同的分布。这样做可以使loss对于参数的梯度不仅取决于模型参数$\theta$,而且考虑到标准化。
再使$x$为某一层的输入，$\chi = \{ x_1,\dots,x_n \}$为训练集中部分样本的一个集合。