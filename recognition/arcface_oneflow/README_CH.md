# InsightFace 在 OneFlow 中的实现


[English](README.md) **|** [简体中文](README_CH.md)

本文介绍如何在 OneFlow 中训练 InsightFace，并在验证数据集上对训练好的网络进行验证。

## 目录
- [InsightFace 在 OneFlow 中的实现](#insightface-在-oneflow-中的实现)
  - [目录](#目录)

  - [准备工作](#准备工作)
    - [安装 OneFlow](#安装-oneflow)
    - [准备数据集](#准备数据集)
      - [1. 下载数据集](#1-下载数据集)
      - [2. 将训练数据集 MS1M 从 recordio 格式转换为 OFRecord 格式](#2-将训练数据集-ms1m-从-recordio-格式转换为-ofrecord-格式)
    
  - [训练和验证](#训练和验证)
    - [训练](#训练)
    
    - [验证](#验证)
    
    - [OneFLow2ONNX](#OneFLow2ONNX)
    
      
    

我们对所有的开发者开放 PR，非常欢迎您加入新的实现以及参与讨论。

## 准备工作

在开始运行前，请先确定：

1. 安装 OneFlow。
2. 准备训练和验证的 OFRecord 数据集。



###  安装 OneFlow

根据 [Install OneFlow](https://github.com/Oneflow-Inc/oneflow#install-oneflow) 的步骤进行安装最新 master whl 包即可。

```
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu102
```

### 准备数据集

根据 [加载与准备 OFRecord 数据集](https://docs.oneflow.org/v0.4.0/extended_topics/how_to_convert_image_to_ofrecord.html) 准备 训练 的 OFReocord 数据集，用以进行 InsightFace 的测试。

[InsightFace 原仓库](https://github.com/deepinsight/insightface)中提供了一系列人脸识别任务相关的数据集，已经完成了人脸对齐等预处理过程。请从[这里](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)下载相应的数据集，并且转换成 OneFlow 可以识别的 OFRecord 格式。**考虑到步骤繁琐，也可以直接下载已经转好的 OFRecord 数据集**：

[MS1MV3](https://oneflow-public.oss-cn-beijing.aliyuncs.com/facedata/MS1V3/oneflow/ms1m-retinaface-t1.zip)

下面以数据集 MS1MV3 为例，展示如何将下载到的 recordio 格式的数据集转换成 OFRecord 格式。

#### 1. 下载数据集

下载好的 MS1MV3 数据集，内容如下：

```
ms1m-retinaface-t1/
       train.idx
       train.rec
       property
       lfw.bin
       cfp_fp.bin
       agedb_30.bin
```



前三个文件是训练数据集 MS1M 的 MXNet 的 recordio 格式相关的文件，后三个 `.bin` 文件是三个不同的验证数据集。



#### 2. 将训练数据集 MS1M 从 recordio 格式转换为 OFRecord 格式
运行： 
```
python tools/dataset_convert/mx_recordio_2_ofrecord_shuffled_npart.py  --data_dir datasets/ms1m-retinaface-t1 --output_filepath faces_emore/ofrecord/train --num_part 16
```
成功后将得到 `num_part` 数量个 OFRecord，本示例中为 16 个，显示如下：

```
tree ofrecord/train/
|-- _SUCCESS
|-- part-00000
|-- part-00001
|-- part-00002
|-- part-00003
|-- part-00004
|-- part-00005
|-- part-00006
|-- part-00007
|-- part-00008
|-- part-00009
|-- part-00010
|-- part-00011
|-- part-00012
|-- part-00013
|-- part-00014
`-- part-00015

0 directories, 17 files
```



## 训练和验证

### 训练

为了减小用户使用的迁移成本，OneFlow 的脚本已经调整为 Torch 实现的风格，用户可以使用 configs/*.py 直接修改参数。


运行脚本：

#### eager 
```
./train_ddp.sh
```
#### Graph
```
train_graph_distributed.sh
```

### 验证

另外，为了方便查看保存下来的预训练模型精度，我们提供了一个仅在验证数据集上单独执行验证过程的脚本。

运行

```
./val.sh
```

## OneFLow2ONNX

```
pip install oneflow-onnx
./convert.sh
```