# RooCode Technical Documentation

# git clone --depth 1 https://github.com/Lukeming-tsinghua/Interpretable-NN-for-IBD-diagnosis.git

## 1. Project Overview
RooCode is a specialized development framework aimed at standardizing code practices and improving development efficiency.

## 2. Project Structure

UIcerative_colitis/
├── data/
│   ├── concatenated_images/
│   ├── CD/
│   │   ├── 1/
│   │   │   └── image/
│   │   ├── 2/
│   │   │   └── image/
│   │   ├── 3/
│   │   │   └── image/
│   │   ├── 4/
│   │   │   └── image/
│   │   ├── 5/
│   │   │   └── image/
│   │   ├── 6/
│   │   │   └── image/
│   │   ├── 7/
│   │   │   └── image/
│   │   ├── 8/
│   │   │   └── image/
│   │   ├── 9/
│   │   │   └── image/
│   │   ├── 10/
│   │   |    └── image/
|   |   └──CD_text/ 
│   ├── UC/
│   │   ├── 1/
│   │   │   └── image/
│   │   ├── 2/
│   │   │   └── image/
│   │   ├── 3/
│   │   │   └── image/
│   │   ├── 4/
│   │   │   └── image/
│   │   ├── 5/
│   │   │   └── image/
│   │   ├── 6/
│   │   │   └── image/
│   │   ├── 7/
│   │   │   └── image/
│   │   ├── 8/
│   │   │   └── image/
│   │   ├── 9/
│   │   │   └── image/
│   │   ├── 10/
│   │   |   └── image/
|   |   └── CD_text/
│   └── ITB/
│       ├── 1/
│       │   └── image/
│       └── ... (similar structure as CD and UC)
├── model/
├── utils/
├── concatenated_images/    # Output folder for processed images
├── train_ViT.py
├── train_resnet.py
└── args.py

## 3. Operation
### 3.1 Setup
所有操作过程中的参数保存在args.py文件中，包括数据集路径、模型保存路径、batch_size、epoch等。
### 3.2 preprocess.py
项目文件结构就在md文件中，在如1这样的文件夹中有若干张肠镜图像和一张文字诊断报告，UC的图像存储在UC1-10中。
现在帮我将此类文件夹中image中的肠镜图像(除文字诊断报告外的图像，为RPT开头,UC1-10中文字诊断为第一张图像)中的有效部分（去除黑色背景）提取出来拼接成一张组合图像，由49张图像组成，如果原始图像书不足49就从头开始复制拼接，如果超出49张就自动计算平均采样拼接，保存的图像格式为jpg。
在CD和ITB文件夹下诊断报告未RPT开头的图像，在UC/UCx/文件夹下第一张图像为诊断报告，将诊断报告中“检查所见”到“检查诊断”之间的文字内容提取到csv文件中，最后将同类的csv文件和拼接图象分别保存到CD/ITB/UC的image_text文件夹中，尝试使用模型的方法，文件夹格式为：
concatenated_images/
├── CD/
│   ├── 1
│   │   ├── 1.jpg
│   │   ├── 1.csv
|   ├── 2
│   │   ├── 2.jpg
│   │   ├── 2.csv
|   ├── 3
│   │   ├── 3.jpg
│   │   ├── 3.csv
......
├── ITB/
│   ├── 1
│   │   ├── 1.jpg
│   │   ├── 1.csv
|   ├── 2
│   │   ├── 2.jpg
│   │   ├── 2.csv
......
└── UC/
|    ├── 1
|    │   ├── 1.jpg
|    │   ├── 1.csv
|    ├── 2
|    │   ├── 2.jpg
|    │   ├── 2.csv
......
### 3.3 text.py
根据Interpretable-NN-for-IBD-diagnosis项目文件中RoBERT-wmm-ext模型，对提取的data/concatenated_images中三个病种目录下的csv文件中的文本进行三分类，并完整执行Interpretable-NN-for-IBD-diagnosis项目中的所有操作，包括蒸馏，去偏，local-interpret等。




