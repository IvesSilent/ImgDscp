# -*coding=utf-8*-

import torch
from torch import nn
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


# 基于Inception的CNN编码器
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, dropout):
        super(EncoderCNN, self).__init__()
        # 加载预训练的InceptionV3模型
        inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        inception.fc = nn.Identity()  # 移除最后的全连接层

        # # 定义从哪些层返回特征
        # return_nodes = {
        #     'Mixed_6e': 'out1',
        #     'Mixed_7a': 'out2',
        #     'Mixed_7b': 'out3',
        #     'Mixed_7c': 'out4'
        # }

        return_nodes = {
            'Mixed_5b': 'out1',
            'Mixed_5c': 'out2',
            'Mixed_5d': 'out3',
            'Mixed_6a': 'out4',
            'Mixed_6b': 'out5',
            'Mixed_6c': 'out6',
            'Mixed_6d': 'out7',
            'Mixed_6e': 'out8',
            'Mixed_7a': 'out9',
            'Mixed_7b': 'out10',
            'Mixed_7c': 'out11'

        }
        # 创建特征提取器
        self.feature_extractor = create_feature_extractor(inception, return_nodes=return_nodes)

        # 冻结参数
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 特征适配层，根据实际输出维度设置
        # self.fc1 = nn.Linear(768, embed_size)  # Mixed_6e: 768 channels
        # self.fc2 = nn.Linear(1280, embed_size)  # Mixed_7a: 1280 channels
        # self.fc3 = nn.Linear(2048, embed_size)  # Mixed_7b: 2048 channels
        # self.fc4 = nn.Linear(2048, embed_size)  # Mixed_7c: 2048 channels
        # self.fc1 = nn.Sequential(nn.Linear(256, embed_size), nn.Dropout(p=dropout))  # Mixed_5b: 256 channels
        # self.fc2 = nn.Sequential(nn.Linear(288, embed_size), nn.Dropout(p=dropout))  # Mixed_5c: 288 channels
        # self.fc3 = nn.Sequential(nn.Linear(288, embed_size), nn.Dropout(p=dropout))  # Mixed_5d: 288 channels
        # self.fc4 = nn.Sequential(nn.Linear(768, embed_size), nn.Dropout(p=dropout))  # Mixed_6a: 768 channels
        # self.fc5 = nn.Sequential(nn.Linear(768, embed_size), nn.Dropout(p=dropout))  # Mixed_6b: 768 channels
        # self.fc6 = nn.Sequential(nn.Linear(768, embed_size), nn.Dropout(p=dropout))  # Mixed_6c: 768 channels
        # self.fc7 = nn.Sequential(nn.Linear(768, embed_size), nn.Dropout(p=dropout))  # Mixed_6d: 768 channels
        # self.fc8 = nn.Sequential(nn.Linear(768, embed_size), nn.Dropout(p=dropout))  # Mixed_6e: 768 channels
        # self.fc9 = nn.Sequential(nn.Linear(1280, embed_size), nn.Dropout(p=dropout))  # Mixed_7a: 1280 channels
        # self.fc10 = nn.Sequential(nn.Linear(2048, embed_size), nn.Dropout(p=dropout))  # Mixed_7b: 2048 channels
        # self.fc11 = nn.Sequential(nn.Linear(2048, embed_size), nn.Dropout(p=dropout))  # Mixed_7c: 2048 channels

        # 【优化方案：注意力机制优化】
        # 动态创建特征适配层
        self.feature_adapters = nn.ModuleDict({
            f'out{i + 1}': nn.Sequential(
                nn.Linear(channel, embed_size),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            ) for i, channel in enumerate([256, 288, 288, 768, 768, 768,
                                           768, 768, 1280, 2048, 2048])
        })

        # # 新增一个全连接层作为特征加权层
        # self.feature_weights = nn.Parameter(torch.ones(len(return_nodes)))

        # 【优化方案：注意力机制优化】
        # 添加一个注意力层来计算每个特征的权重
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.Tanh(),
                nn.Linear(embed_size, 1)
            ) for _ in range(len(return_nodes))
        ])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, images):
        # 提取多尺度特征
        features = self.feature_extractor(images)

        # out1 = F.adaptive_avg_pool2d(features['out1'], (1, 1)).view(features['out1'].size(0), -1)
        # out2 = F.adaptive_avg_pool2d(features['out2'], (1, 1)).view(features['out2'].size(0), -1)
        # out3 = F.adaptive_avg_pool2d(features['out3'], (1, 1)).view(features['out3'].size(0), -1)
        # out4 = F.adaptive_avg_pool2d(features['out4'], (1, 1)).view(features['out4'].size(0), -1)
        # out5 = F.adaptive_avg_pool2d(features['out5'], (1, 1)).view(features['out5'].size(0), -1)
        # out6 = F.adaptive_avg_pool2d(features['out6'], (1, 1)).view(features['out6'].size(0), -1)
        # out7 = F.adaptive_avg_pool2d(features['out7'], (1, 1)).view(features['out7'].size(0), -1)
        # out8 = F.adaptive_avg_pool2d(features['out8'], (1, 1)).view(features['out8'].size(0), -1)
        # out9 = F.adaptive_avg_pool2d(features['out9'], (1, 1)).view(features['out9'].size(0), -1)
        # out10 = F.adaptive_avg_pool2d(features['out10'], (1, 1)).view(features['out10'].size(0), -1)
        # out11 = F.adaptive_avg_pool2d(features['out11'], (1, 1)).view(features['out11'].size(0), -1)

        # # 特征适配
        # out1 = self.fc1(out1)
        # out2 = self.fc2(out2)
        # out3 = self.fc3(out3)
        # out4 = self.fc4(out4)
        # out5 = self.fc5(out5)
        # out6 = self.fc6(out6)
        # out7 = self.fc7(out7)
        # out8 = self.fc8(out8)
        # out9 = self.fc9(out9)
        # out10 = self.fc10(out10)
        # out11 = self.fc11(out11)

        # 【优化方案：注意力机制优化】
        outs = []
        for i, (name, feature) in enumerate(features.items()):
            feature = F.adaptive_avg_pool2d(feature, (1, 1)).view(feature.size(0), -1)
            feature = self.feature_adapters[name](feature)
            outs.append(feature)

        # # 特征融合
        # # features = out1 + out2 + out3 + out4
        # features = (out1 * self.feature_weights[0] + out2 * self.feature_weights[1] +
        #             out3 * self.feature_weights[2] + out4 * self.feature_weights[3] +
        #             out5 * self.feature_weights[4] + out6 * self.feature_weights[5] +
        #             out7 * self.feature_weights[6] + out8 * self.feature_weights[7] +
        #             out9 * self.feature_weights[8] + out10 * self.feature_weights[9] +
        #             out11 * self.feature_weights[10])

        # 【优化方案：注意力机制优化】
        # 注意力权重计算
        attention_weights = torch.cat([
            self.softmax(self.attention_layers[i](outs[i])) for i in range(len(outs))
        ], dim=-1)  # 注意力权重的形状应该是 (batch_size, num_features)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        # 特征融合
        # 调整outs的形状为 (batch_size, num_features, embed_size)

        outs = torch.stack(outs)  # 形状已经是 (num_features, batch_size, embed_size)
        # print(f"outs before adjusted:{outs.shape}")
        # breakpoint()
        outs = outs.transpose(0, 1)  # 调整为 (batch_size, num_features, embed_size)
        # print(f"outs after adjusted:{outs.shape}")
        # print(f"attention_weights.unsqueeze(2).shape:{attention_weights.unsqueeze(2).shape}")

        features = torch.sum(attention_weights.unsqueeze(2) * outs, dim=1)  # 执行矩阵乘法并求和
        # print(features.shape)
        # breakpoint()

        # print("特征融合成功")

        # breakpoint()
        # print(features.shape)
        # breakpoint()

        # 输出尺寸是(batch_size, embed_size)
        return features


# 【Transformer解码器】
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_size, nhead=8, dim_feedforward=hidden_size, dropout=dropout),
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions):
        # captions before embedding:(batch_size, seq_len)
        # 前向传播
        embeddings = self.embed(captions)
        # captions after embedding:(seq_len, batch_size)
        embeddings = self.dropout(embeddings)  # 添加dropout层
        # embeddings = embeddings.permute(1, 0, 2)
        # # captions -> embeddings:(seq_len, batch_size, embed_size)

        # features before unsqueezed: (batch_size, embed_size)
        features = features.unsqueeze(0).repeat(captions.size(1), 1, 1)  # (seq_len, batch_size, embed_size)
        # features after unsqueezed and repeat: (seq_len, batch_size, embed_size)

        features = features.permute(1, 0, 2)

        # 初始化输出张量，维度为 (batch_size, seq_len, embed_size)
        outputs = torch.zeros_like(embeddings)

        # 循环处理每个样本
        for i in range(features.size(0)):  # features.size(0) 是 batch_size
            # 获取第i个样本的特征和caption嵌入
            features_i = features[i].unsqueeze(0)  # (1, seq_len, embed_size)
            captions_i = embeddings[i].unsqueeze(0)  # (1, seq_len, embed_size)
            # 通过Transformer解码器
            output_i = self.transformer(captions_i, features_i)

            # 将第i个样本的输出存储到outputs中对应的位置
            outputs[i] = output_i.squeeze(0)  # 去掉批次维度

        # 通过线性层得到最终的输出分布
        outputs = self.linear(outputs)
        # print(f"outputs.shape = {outputs.shape}")
        # breakpoint()

        return outputs


# 【图像描述模型】
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size, dropout)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, dropout)

    def forward(self, images, captions):
        # 前向传播
        features = self.encoder(images)
        # print("Features:", features)  # 打印features检查是否为None
        outputs = self.decoder(features, captions)
        # print("Outputs:", outputs)  # 打印outputs检查是否为None
        # breakpoint()
        # print("前向传播完成")
        # breakpoint()
        return outputs
