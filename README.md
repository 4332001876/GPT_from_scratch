# Readme

## Introduction
这里是我从零开始实现的一个GPT模型，用于帮助自己和大家更好地理解GPT模型的原理。

这一项目的实现框架来源于USTC人工智能基础大作业（待重构），debug阶段少部分bug的解决参考过Karpathy的nanoGPT。相较于原版GPT，本项目使用了MoE架构。

我们使用的模型架构如下：
- 4层Transformer Decoder
- 8个Attention Head
- 向量维度为256
- MoE专家为2选1

我们使用的训练参数如下：
- 学习率：5e-4
- 上下文长度：64
- batch size：64
- 梯度累积步数：16
- 训练epoch数：20

我们分别使用了字符级分词器和子词级分词器进行训练。

在经典的莎士比亚文集数据集上，以8：2的比例划分训练集和测试集，字符级分词器训练的validation loss为0.563，而子词级分词器的validation loss为0.215。

从感性的结果上来看，在训练20个epoch时，字符级分词器训出的模型生成的内容语法上已经接近正确，在局部语义上也已经有了一定的连贯性；而子词级分词器训出的模型能完美背诵原文，即使要求补全的内容不在原文中也会背诵选段背诵（过拟合，没有发挥compression is all you need的效果）

我们曾测试了其在中文wiki上的训练效果，但目前暂时未有较好的成果（validation loss在4左右就不再下降，生成的内容质量也很差，考虑是batchsize过小的问题）。

## Future Work
- 加入flash attention
- 优化训练参数，提高模型性能，获得在wiki数据集上的较好效果
- MoE架构未经优化，每一专家都参与运算，考虑可以提高模型性能

