## 项目结构

```
ai-course-hw2/
├── 人工智能原理课程项目2.pdf                  # 【文档】作业说明
│
├──STL10/
│   ├──test/            				   #测试集
│   └──train/							   #训练集
│
├──outputs/                                #输出
│   ├──baseline/						   #基准训练结果，未进行优化
│   ├──aug_on/							   #采用数据增强后的训练结果
│   ├──model_variant/					   #采用Sigmoid/Leaky ReLU激活函数、average池化、 
│   │    │                                 dropout正则化后的结果
│	│	 ├──sigmoid/					   #采用Sigmoid激活函数获得的结果
│	│	 └──leaky_relu/					   #采用Leaky ReLU激活函数获得的结果
│   ├──optim_variant/					   #采用随机梯度下降优化器，学习率为0.01的结果
│   ├──grad_cam/						   #使用Gard-cam可视化方法输出的结果
│   │	└──grad_cam.png
│   ├──accuracy_curves.png				   
│   ├──confusion_matrix.png				   
│   ├──loss_curves.png					   
│   ├──best_model.pt					   
│   └──classification_report.txt           #模型分类报告
│
├──training.py							   #基准训练代码
│
├──report.pdf                              #实验报告文件
│
├──README.md                               #本文件
│
├──experiments.py						   #模型优化分析相关代码
│
└──grad_cam.py							   #可视化相关代码
```

## 项目简介

本项目基于 STL10 数据集实现卷积神经网络图像分类，并包含多组对比实验（数据增强、激活函数、优化器）以及 Grad-CAM 可解释性可视化。

## 环境与依赖

- Python 3.12
- PyTorch、Torchvision
- NumPy、Matplotlib、Scikit-learn

可用以下命令安装依赖：

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

## 数据集准备

将 STL10 数据集按以下结构放置：

```
STL10/
├── train/
└── test/
```

每个类别为一个子文件夹，内部为对应图片。

## 训练与评估

运行基准训练（训练结束后会保存最优权重与可视化结果）：

```bash
python training.py
```

输出文件会保存在 outputs/ 目录下，包括：

- best_model.pt
- accuracy_curves.png
- loss_curves.png
- confusion_matrix.png
- classification_report.txt

## 实验对比

运行实验脚本（数据增强、激活函数、优化器对比）：

```bash
python experiments.py
```

各实验结果会分别存入 outputs/aug_on、outputs/model_variant、outputs/optim_variant。

## Grad-CAM 可解释性

对指定图片生成 Grad-CAM 热力图：

```bash
python grad_cam.py --image ./STL10/test/dog/04621.png --model ./outputs/best_model.pt --output ./outputs/grad_cam/grad_cam.png
```

## 报告

实验报告见 report.pdf。
