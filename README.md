OCR_for_AliYun_tianchi_Competition
天池大赛>MTWI 2018 挑战赛一：网络图像的文本识别

## 简介

 赛题背景介绍

在互联网世界中，图片是传递信息的重要媒介。特别是电子商务，社交，搜索等领域，每天都有数以亿兆级别的图像在流动传播。图片中的文字识别（OCR）在商业领域有重要的应用价值，同时也是学术界单研究热点。然而，研究领域尚没有基于网络图片的、以中文为主的OCR数据集。本竞赛将公开基于网络图片的中英数据集，该数据集数据量充分，涵盖数十种字体，几个到几百像素字号，多种版式，较多干扰背景。期待学术界可以在本数据集上作深入的研究，工业界可以藉此发展在图片管控，搜索，信息录入等AI领域的工作。 


基于Tensorflow和Keras实现端到端的不定长中文字符检测和识别

* 文本检测：CTPN
* 文本识别：DenseNet + CTC

## 环境部署（基于Ubuntu操作系统）
``` 
sh setup.sh
```
即：
```
pip install numpy scipy matplotlib pillow
pip install easydict opencv-python keras h5py PyYAML
pip install cython==0.24

# for gpu
pip install tensorflow-gpu==1.3.0
chmod +x ./ctpn/lib/utils/make.sh
cd ./ctpn/lib/utils/ && ./make.sh

# for cpu
pip install tensorflow==1.3.0
chmod +x ./ctpn/lib/utils/make_cpu.sh
cd ./ctpn/lib/utils/ && ./make_cpu.sh
```

## Demo
将测试图片放入test_images目录，检测结果会保存到test_result中

执行命令：
``` 
python demo.py
```

## 模型训练

### CTPN训练
```
cd ctpn/lib/utils
chmod +x make.sh
./make.sh
```
即：
```
cython bbox.pyx
cython cython_nms.pyx
cython gpu_nms.pyx
python setup.py build_ext --inplace
mv utils/* ./
rm -rf build
rm -rf utils
```
详细内容参见ctpn/README.md

### DenseNet + CTC训练

#### 1. 数据准备

数据集来自MTWI 2018 挑战赛一：网络图像的文本识别	：https://tianchi.aliyun.com/competition/entrance/231684/information
* 提供20000张图像作为本次比赛的数据集。其中50%用来作为训练集，50%用来作为测试集。该数据集全部来源于网络图像，主要由合成图像，产品描述，网络广告构成。
* 这些图像是网络上最常见的图像类型。每一张图像或者包含复杂排版，或者包含密集的小文本或多语言文本，或者包含水印，这对文本检测和识别均提出了挑战。对于每一张图像，都会有一个相应的文本文件（.txt）（UTF-8编码与名称：[图像文件名] .txt）。文本文件是一个逗号分隔的文件，其中每行对应于图像中的一个文本串，并具有以下格式：
```
      X1，Y1，X2，Y2，X3，Y3，X4，Y4，“文本”
```
* 其中X1，Y1，Y2，X2，X3，X4，Y3，Y4分别代表文本的外接四边形四个顶点坐标。而“文本”是四边形包含的实际文本内容。图2是标注的图片，红色的框代表标注的文本框。
* 标注时我们对所有语言，所有看不清的文字串均标注了外接框（比如图2中的小字），但对于除了中文，英文以外的其它语言以及看不清的字符并未标注文本内容，而是以“###”代替。
图片解压后放置到train/images目录下，描述文件放到train目录下

#### 2. 训练
执行
``` 
cd train
python train.py
```

#### 3. 结果

| val acc | predict | model |
| -----------| ---------- | -----------|
| 0.983 | 8ms | 18.9MB |

* GPU: GTX TITAN X
* Keras Backend: Tensorflow

#### 4. 生成自己的样本

可参考[SynthText_Chinese_version](https://github.com/JarveeLee/SynthText_Chinese_version)，[TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator)和[text_renderer](https://github.com/Sanster/text_renderer)

## 效果展示

<div>
<img width="420" height="420" src="https://github.com/YCG09/chinese_ocr/blob/master/demo/demo_detect.jpg"/>
<img width="420" height="420" src="https://github.com/YCG09/chinese_ocr/blob/master/demo/demo_rec.jpg"/>
</div>

## 参考

[1] https://github.com/eragonruan/text-detection-ctpn

[2] https://github.com/senlinuc/caffe_ocr

[3] https://github.com/chineseocr/chinese-ocr

[4] https://github.com/xiaomaxiao/keras_ocr

[5] https://blog.csdn.net/NUDTDING2019/article/details/93778402

[6] https://tianchi.aliyun.com/competition/entrance/231684/introduction?spm=5176.12281973.1005.7.3dd54c2ajNPwx7