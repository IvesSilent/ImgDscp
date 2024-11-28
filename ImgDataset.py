# -*coding=utf-8*-

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt


# 【Flickr30kDataset类】
# 用于加载图像和对应的描述
class Flickr30kDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5, vocab=None):
        """
        Args:
            root_dir (string): Directory with all the images. 所有图像的目录
            captions_file (string): Path to the captions file. 标注文档路径
            transform (callable, optional): Optional transform to be applied on a sample.
            freq_threshold (int): The minimum frequency of words to be kept in vocabulary.
            vocab (Vocabulary, optional): Preloaded Vocabulary object.
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file, delimiter='\t', names=['image', 'caption'])
        self.transform = transform

        # Splitting image names and captions
        self.df['image'] = self.df['image'].apply(lambda x: x.split('#')[0])
        self.df['image'] = self.df['image'].map(lambda x: os.path.join(self.root_dir, x))

        # # 【二改】分割文件名、序号和描述文本
        # self.df['image'], self.df['idx'], self.df['caption'] = self.df['caption'].apply(lambda x: x.split('\t')).apply(
        #     lambda x: x[0].split('#'), axis=1).tolist()
        # self.df['image'] = self.df['image'].map(lambda x: os.path.join(self.root_dir, x[0]))
        # self.df['caption'] = self.df['caption'].apply(lambda x: x[1].strip())  # 去除可能的前后空格

        # # 读取标注文件
        # with open(captions_file, 'r', encoding='utf-8') as f:
        #     lines = f.readlines()

        # # 初始化空列表来存储数据
        # data = []
        #
        # # 处理每一行数据
        # for line in lines:
        #     parts = line.strip().split('\t')
        #     if len(parts) != 2:
        #         continue  # 如果格式不正确，跳过这一行
        #     filename, caption = parts[0], parts[1]
        #     image_id, idx = filename.split('#')
        #     data.append({'image': os.path.join(root_dir, image_id), 'idx': idx, 'caption': caption})

        # # 初始化数据框架
        # self.df = pd.DataFrame(data)

        if vocab is None:
            # Initialize vocabulary and build vocab
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary(self.df.caption.tolist())
            # 保存词汇表到文件
            torch.save(self.vocab, f'vocab.pth')
        else:
            # 若已有则直接加载vocab`
            self.vocab = vocab

        # # 添加检查是否已保存vocab的逻辑
        # vocab_path = 'vocab.pth'
        # if os.path.exists(vocab_path):
        #     self.vocab = torch.load(vocab_path)
        # else:
        #     self.vocab = Vocabulary(freq_threshold)
        #     self.vocab.build_vocabulary(self.df.caption.tolist())
        #     torch.save(self.vocab, vocab_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df.iloc[idx, 0]
        caption = self.df.iloc[idx, 1]

        # img_name = self.df.iloc[idx, 'image']
        # caption = self.df.iloc[idx, 'caption']

        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(numericalized_caption)


# 【Vocabulary类】
# 用于构建和管理词汇表，将文本转换为数字。
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
        return [tok.lower() for tok in text.split()]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4  # Start index from 4 because 0-3 are already reserved

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


# 自定义一个简单的数据集类，仅用于加载测试图片
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the test images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_path


# 示例使用
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # 调整为Inception模型推荐的最小尺寸
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
        transforms.RandomResizedCrop(size=(299, 299), scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # 随机裁剪和缩放
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 【归一化】
    # 对于 Inception V3，输入图像的归一化参数通常是：
    #   均值（Mean）：[0.485, 0.456, 0.406]
    #   标准差（Standard Deviation）：[0.229, 0.224, 0.225]

    root_dir = 'data/train/train_img'
    captions_file = 'data/train/train.token'

    dataset = Flickr30kDataset(root_dir=root_dir,
                               captions_file=captions_file,
                               transform=transform)
    # print(dataset[0][0])  # Print the first (image, caption) pair

    # 选择一个样本
    image, caption = dataset[0]

    # 显示原始图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image.permute(1, 2, 0))  # 直接显示原始PIL图像
    plt.axis('off')

    # 应用转换
    transformed_image, _ = dataset[0]  # 重新从数据集中获取样本，并应用转换
    plt.subplot(1, 2, 2)
    plt.title("Transformed Image")
    plt.imshow(transformed_image.permute(1, 2, 0))  # 将通道维度换到前面，并显示转换后的图像
    plt.axis('off')

    plt.show()
