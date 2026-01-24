# AGENTS.md

## 项目概述

本项目是论文《Edit Flows：Flow Matching with Edit Operations》的教育性实现项目。该项目实现了离散流匹配（Discrete Flow Matching）算法，通过插入、删除和替换三种编辑操作来建模离散序列的生成过程。项目使用合成数据集（离散化正弦波序列）来演示编辑流模型的训练和采样过程。

## Iflow 运行规则
 - 用中文回答。
 - 使用 uv 管理 Python 虚拟环境，使用该虚拟环境运行 Python 代码。
 - 不要使用 try-except 这种防御性编程，有错误就让它报出来
 - 避免不必要的函数封装，尽可能写简洁。
 - 不要做没有意义、没有信息量的打印和输出。
 - 有关官方代码，尤其是 langchain 相关的，使用 context7 查阅官方文档。