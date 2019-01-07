# Bessel's Correction

Bessel's Correction描述的是在使用样本信息估计population的方差时，使用$n-1$而不是$n$。

样本的均值：$\overline {x}={\frac {1}{n}}\sum _{i=1}^{n}x_{i}$

有偏置样本的方差：$s_n^2=\frac {1}{n}\sum _{i=1}^{n}\left(x_{i}-{\overline {x}}\right)^{2}={\frac {\sum _{i=1}^{n}\left(x_{i}^{2}\right)}{n}}-{\frac {\left(\sum _{i=1}^{n}x_{i}\right)^{2}}{n^{2}}}$

无偏置的样本方差：$s^2=\frac {1}{n-1}\sum _{i=1}^{n}\left(x_{i}-{\overline {x}}\right)^{2}={\frac {\sum _{i=1}^{n}\left(x_{i}^{2}\right)}{n-1}}-{\frac {\left(\sum _{i=1}^{n}x_{i}\right)^{2}}{(n-1)n}}=\left({\frac {n}{n-1}}\right)\,s_{n}^{2}$

## Why

首先描述两个概念,

**统计误差(Statistical error)**：A statistical error (or disturbance) is the amount by which an observation differs from its expected value, the latter being based on the whole population from which the statistical unit was chosen randomly. 统计误差描述的是一个样本个体的统计信息和无法观测的总体的统计信息之间的误差。

**残差(Residual)**：A residual (or fitting deviation), is an observable estimate of the unobservable statistical error. 残差描述的是样本个体和可观测的样本统计信息之间的误差。

为什么要在计算方差的估计时使用$n-1$呢？可以把Bessel's Correction理解成残差的自由度。
$(x_1-\overline{x},\dots,x_n-\overline{x})$表示$n$个样本的独立观测值，$\overline{x}$表示样本的均值。虽然这里有$n$个独立样本，但是独立的残差只有$n-1$个，因为所有的残差相加等于0。

我们在通过样本信息估计无法观测的总体统计信息时，使用$\frac{1}{n}$估计方差时会总比真实的方差小或等于。因为样本的残差总为0，而样本的统计误差不太会为0（除非碰巧均值的样本估计=总体均值）。

下面用简单的代数来描述这个问题：
$(a+b)^2=a^2+2ab+b^2$，$a$表示样本个体和样本均值的误差（残差），$b$表示样本均值和总体均值的误差。$a+b$其实表示个体和总体均值之间的误差（统计误差）。
对于$(x_1-\overline{x},\dots,x_n-\overline{x})$，真实的无偏置的方差乘以n为$\sum_{i=1}^{n} (a_n+b_n)^2=\sum_{i=1}^{n}a_n^2 +\sum_{i=1}^{n}2a_nb_n +\sum_{i=1}^{n}b_n^2=\sum_{i=1}^{n}a_n^2 +\sum_{i=1}^{n}b_n^2$，因为残差$a$的和为0。此时可以看到个体和总体均值之间的误差，即真实的方差，总是要大于用残差计算的用来估计的“方差”。
因为有样本均值和无法观测的总体均值之间误差$b$的存在，所有残差$a$的平方和除以n总是小于等于无偏置的总体方差估计。

## Reference
https://en.wikipedia.org/wiki/Errors_and_residuals