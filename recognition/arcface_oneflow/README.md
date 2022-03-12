
# InsightFace in OneFlow

[English](README.md) **|** [简体中文](README_CH.md)

It introduces how to train InsightFace in OneFlow, and do verification over the validation datasets via the well-toned networks.

## Contents

\- [InsightFace in OneFlow](#insightface-in-oneflow)

 \- [Contents](#contents)

 \- [Preparations](#preparations)

  \- [Install OneFlow](#install-oneflow)

  \- [Data preparations](#data-preparations)

   \- [1. Download datasets](#1-download-datasets)

   \- [2. Transformation from MS1M recordio to OFRecord](#2-transformation-from-ms1m-recordio-to-ofrecord)

 \- [Training and verification](#training-and-verification)

  \- [Training](#training)

  \- [OneFLow2ONNX](#OneFLow2ONNX)



## Preparations

First of all, before execution, please make sure that:

1. Install OneFlow

2. Prepare training and validation datasets in form of OFRecord.



### Install OneFlow



According to steps in [Install OneFlow](https://github.com/Oneflow-Inc/oneflow#install-oneflow) install the newest release master whl packages.

```
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu102
```



### Data preparations

According to [Load and Prepare OFRecord Datasets](https://docs.oneflow.org/en/v0.4.0/extended_topics/ofrecord.html), datasets should be converted into the form of OFREcord, to test InsightFace.



It has provided a set of datasets related to face recognition tasks, which have been pre-processed via face alignment or other processions already in [InsightFace](https://github.com/deepinsight/insightface). The corresponding datasets could be downloaded from [here](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) and should be converted into OFRecord, which performs better in OneFlow. Considering the cumbersome steps, **it is suggested to download converted OFrecord datasets**:

[MS1MV3](https://oneflow-public.oss-cn-beijing.aliyuncs.com/facedata/MS1V3/oneflow/ms1m-retinaface-t1.zip)

It illustrates how to convert downloaded datasets into OFRecords, and take MS1MV3 as an example in the following.

#### 1. Download datasets

The structure of the downloaded MS1MV3   is shown as follown：



```
ms1m-retinaface-t1/

​    train.idx

​    train.rec

​    property

​    lfw.bin

​    cfp_fp.bin

​    agedb_30.bin
```

The first three files are MXNet recordio format files of MS1M training dataset, the last three `.bin` files are different validation datasets.



#### 2. Transformation from MS1M recordio to OFRecord


Run 
```
python tools/mx_recordio_2_ofrecord_shuffled_npart.py  --data_dir datasets/ms1m-retinaface-t1 --output_filepath ms1m-retinaface-t1/ofrecord/train --num_part 16
```
And you will get the number of `part_num` parts of OFRecord, it's 16 parts in this example, it showed like this
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

## Training and verification


### Training

To reduce the usage cost of user, OneFlow draws close the scripts to Torch style, you can directly modify parameters via configs/*.py

#### eager 
```
./train_ddp.sh
```
#### Graph
```
train_graph_distributed.sh
```


### Varification

Moreover, OneFlow offers a validation script to do verification separately, val.py, which facilitates you to check the precision of the pre-training model saved.

```
./val.sh

```
## OneFLow2ONNX

```
pip install oneflow-onnx
./convert.sh
```