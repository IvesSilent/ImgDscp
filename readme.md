# ImgDscp - 基于深度学习的图像描述项目

该项目旨在通过深度学习技术自动生成给定图像的描述文本。我采用了Flickr30k数据集，利用CNN作为编码器和Transformer作为解码器的架构来完成图像描述任务。

## 项目结构

```
image-captioning-project/
│
├── data/                       # 数据集目录
│   ├── train/                  # 训练集目录
│   │   ├── train_img/          # 训练集图片
│   │   └── train.token        # 训练集标注
│   ├── val/                   # 验证集目录
│   │   ├── val_img/           # 验证集图片
│   │   └── val.token          # 验证集标注
│   └── test/                  # 测试集目录
│       ├── test_img/          # 测试集图片
│       └── test.token         # 测试集输出
│
├── /Result_Fig/                  # 训练过程可视化的图片文档
├── /models/                      # 训练后模型保存目录
├── /models/                      # 训练日志保存目录
├── /singleprd/                   # 存放predict.py生成的描述文本
│
├── models.py                  # 模型架构定义文件
├── ImgDataset.py              # 数据集处理类定义文件
├── train.py                  # 训练及验证过程脚本
├── test.py                   # 测试过程脚本
├── predict.py                 # 单独预测过程脚本
├── vocab.pth                     # 词表文件
├── requirements.txt           # 项目依赖环境文件
└── README.md                 # 项目说明文档
```

## 环境配置

在开始之前，请确保您的环境中安装了以下依赖：

```plaintext
torch==1.9.0
torchvision==0.10.0
pillow==8.1.0
pandas==1.2.3
matplotlib==3.3.4
nltk==3.5
rouge==1.0
pycocoevalcap==1.2.1
tqdm==4.59.0
argparse
```

您可以通过运行以下命令来安装这些依赖：

```bash
pip install -r requirements.txt
```

### 关于PyTorch

`torch`和`torchvision`
的安装可能需要额外的步骤，因为它们通常需要与你的CUDA版本兼容。如果你的系统安装了CUDA，你需要确保安装的`torch`
和`torchvision`版本与CUDA版本相匹配。

你可以通过访问PyTorch官方网站的安装指南来获取正确的安装命令：[PyTorch Get Started](https://pytorch.org/get-started/previous-versions/)。

## 数据处理

数据集是Flickr30k，训练集、验证集和测试集分别位于根目录下的`/train`、`/val`、`/test`文件夹。

训练用图片文件在`/train/train_img`内，`/train/train.token`是训练数据的标注，格式如下：

```plaintext
<TestImage_name><#><序号></t><caption>
```

标注示例如下：

```plaintext
1000092795.jpg#0 Two young guys with shaggy hair look at their hands while hanging out in the yard .
1000092795.jpg#1 Two young , White males are outside near many bushes .
1000092795.jpg#2 Two men in green shirts are standing in a yard .
1000092795.jpg#3 A man in a blue shirt standing in a garden .
1000092795.jpg#4 Two friends enjoy time spent together .
10002456.jpg#0 Several men in hard hats are operating a giant pulley system .
10002456.jpg#1 Workers look down from up above on a piece of equipment .
10002456.jpg#2 Two men working on a machine wearing hard hats .
10002456.jpg#3 Four men on top of a tall structure .
10002456.jpg#4 Three men on a large rig .
```

一张图片对应五句自然语言文本描述， 验证集格式相同。

我在构建项目时所用数据集来自老师发的文件：[东南大学云盘](https://pan.seu.edu.cn:443/link/215E44A851DA77CA52FF410F26F15498)
。下载需要校园网。本仓库内不含数据文件，需要单独下载或将数据集自行处理后放入指定位置。

我使用Torchvision进行图像数据加载和预处理，并在加载图像数据时采用了数据增强技术。

## 模型架构

我选用了Inception作为CNN编码器，以及包含自注意力机制的Transformer作为解码器。模型架构定义在 `models.py` 文件中。

| 技术/优化手段 | 描述                                          | 参数细节                     |
|---------|---------------------------------------------|--------------------------|
| 框架      | PyTorch                                     | 版本1.9.0                  |
| 编码器     | CNN，使用轻量化的预训练架构Inception V3                 | 权重：ImageNet1K v1         |
| 解码器     | Transformer，包括多个解码器层，每层包括自注意力机制、前馈神经网络和层归一化 | 层数：6，头数：8，隐藏层维度：512      |
| 特征适配层   | 提取多层特征并合并以一个注意力层合并为特征序列                     | 适配层数：11，使用ReLU激活和Dropout |
| 硬件加速    | 若设备可用GPU则使用GPU进行训练、验证和预测                    |                          |
| 正则化     | 加入可调的Dropout方法以防止过拟合                        | Dropout率：0.3             |

## 训练及验证

训练和验证过程由 `train.py` 脚本控制。我使用Xavier均匀初始化权重，采用交叉熵损失和Adam优化器进行优化，并使用余弦退火学习率调度机制。训练过程中还会计算BLEU、ROUGE、CIDEr等指标。

| 技术/优化手段 | 描述                                    | 参数细节                    |
|---------|---------------------------------------|-------------------------|
| 权重初始化   | 使用Xavier均匀初始化权重                       |                         |
| 损失函数    | 使用交叉熵损失来优化模型，引入标签平滑技术                 | 平滑参数ε：0.05              |
| 优化器     | 使用Adam优化器                             | 学习率：0.0005，权重衰减：1e-5    |
| 学习率调度   | 余弦退火的学习率调度机制                          | T_max：20，η_min：0.000001 |
| 混合精度计算  | 使用混合精度计算的方式训练                         |                         |
| 学习方式    | 使用Teacher Forcing的学习方式                |                         |
| 进度显示    | 以进度条显示训练的进度                           |                         |
| 模型存储    | 训练完成后存储模型方便调整学习率多次训练，模型存储于/models目录   |                         |
| 早停机制    | 加入早停机制来避免过拟合，并在验证集上性能不再提升时停止训练        | 早停阈值：5个epoch无提升         |
| 性能评估    | 训练同时进行验证，计算BLEU、ROUGE、CIDEr等指标并可视化    |                         |
| 可视化存储   | 将可视化的图片存入/Result_Fig，将训练后的模型存入/models |                         |
| 进度显示    | 以进度条显示验证的进度                           |                         |

在终端启动train.py，配置对应的参数，或直接采用默认参数。

```bash
python train.py \
  --train_root_dir data/train/train_img \
  --train_captions_file data/train/train.token \
  --val_root_dir data/val/val_img \
  --val_captions_file data/val/val.token \
  --batch_size 32 \
  --num_epochs 3 \
  --embed_size 256 \
  --hidden_size 512 \
  --vocab_size 8943 \
  --num_layers 6 \
  --freq_threshold 1 \
  --early_stop_count 0 \
  --early_stop_limit 5 \
  --clip_norm 1.0 \
  --smooth_epsilon 0.1 \
  --lr 0.001 \
  --dropout 0.3 \
  --pretrain  \
  --preModel  models/YOUR_OLD_MODEL.pth\
  --vocab_path vocab.pth
  ```

## 测试

测试过程由 `test.py` 脚本控制，该脚本读取 `test/test_img` 目录内的图片文件，并在 `test` 目录内创建文本文件 `test.token`
，对每一张图片生成五句描述文本。

## 单独预测

单独预测过程由 `predict.py` 脚本控制，它读取指定路径的图片，并在 `/singleprd` 目录内创建文本文件，生成该图片的五句描述。

## 使用说明

- 训练模型：运行 `python train.py`。
- 测试模型：运行 `python test.py`。
- 单独预测：运行 `python predict.py --image_path <your_image_path>`。

## 贡献与反馈

欢迎对本项目提出宝贵的意见和建议。如有任何问题，请通过GitHub Issues进行反馈。

## 许可证

本项目采用 [MIT License](https://opensource.org/licenses/MIT)。

