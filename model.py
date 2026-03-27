# model.py
# 【修改2026/03/25】原版本见model.py
# -*coding=utf-8*-

import math
import torch
from torch import nn
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


# 基于Inception的CNN编码器
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, dropout, feature_dim=2048):
        super(EncoderCNN, self).__init__()
        # 加载预训练的InceptionV3模型
        inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

        # 移除辅助分类器（如果存在）
        inception.aux_logits = False
        inception.AuxLogits = None

        # 移除最后的全局池化和fc层，保留空间特征
        # InceptionV3最后特征图: (batch, 2048, 8, 8) 对于输入299x299
        # 此处形参feature_dim即最终特征图的特征维度
        self.features = nn.Sequential(*list(inception.children())[:-2])  # 去掉AdaptiveAvgPool和fc

        # 冻结参数
        for param in self.features.parameters():
            param.requires_grad = False

        # 将CNN特征维度映射到embed_size
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, embed_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # 可学习的图像位置编码（替代原来的attention权重）
        # 假设最终特征图是8x8=64个位置
        self.num_patches = 64  # 8x8
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_size) * 0.02)

    def forward(self, images):
        """
        Args:
            images: (batch_size, 3, 299, 299)
        Returns:
            features: (batch_size, num_patches, embed_size) 序列化特征
        """
        # 提取特征: (batch, 2048, 8, 8)
        cnn_features = self.features(images)

        batch_size = cnn_features.size(0)

        # 展平空间维度: (batch, 2048, 8, 8) -> (batch, 2048, 64) -> (batch, 64, 2048)
        cnn_features = cnn_features.flatten(2).transpose(1, 2)  # (batch, 64, 2048)

        # 投影到embed_size: (batch, 64, embed_size)
        features = self.feature_proj(cnn_features)

        # 加位置编码
        features = features + self.pos_embed

        return features




# 【Transformer解码器】
class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout,max_seq_len=100):
        super(DecoderTransformer, self).__init__()

        self.embed_size = embed_size
        self.vocab_size = vocab_size

        # 词嵌入
        self.embed = nn.Embedding(vocab_size, embed_size)

        # 位置编码（Transformer需要位置信息）
        self.pos_encoding = PositionalEncoding(embed_size, dropout, max_len=max_seq_len)

        # Transformer Decoder
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_size,
                                       nhead=8,
                                       dim_feedforward=hidden_size,
                                       dropout=dropout,
                                       batch_first=True  # 使用batch_first，输入输出为(batch, seq, feature)
                                       ),
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_size, vocab_size)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) #

    def forward(self, features, captions):
        """
        Args:
            features: (batch_size, seq_len_enc, embed_size) 来自encoder的图像特征序列
            captions: (batch_size, seq_len_dec) 已tokenized的caption序列
        Returns:
            outputs: (batch_size, seq_len_dec, vocab_size)
        """

        # 1. 词嵌入 + 位置编码
        # captions: (batch_size, seq_len) -> embeddings: (batch_size, seq_len, embed_size)
        embeddings = self.embed(captions) * math.sqrt(self.embed_size)  # 缩放，Transformer标准做法
        embeddings = self.pos_encoding(embeddings)
        embeddings = self.dropout(embeddings) # 添加dropout层

        # 2. 生成因果掩码（防止看到未来词）
        # tgt_mask: (seq_len_dec, seq_len_dec) 的下三角矩阵
        seq_len = captions.size(1)
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(captions.device)

        # 3. 创建padding掩码（处理变长序列）
        # 假设padding_idx=0，需要根据实际情况调整
        padding_mask = (captions == 0)  # (batch_size, seq_len_dec)

        # 4. Transformer Decoder前向传播
        # memory: encoder输出的图像特征 (batch_size, seq_len_enc, embed_size)
        # tgt: caption嵌入 (batch_size, seq_len_dec, embed_size)
        # tgt_mask: 因果掩码，防止看到未来词
        # tgt_key_padding_mask: 处理padding位置
        outputs = self.transformer(
            tgt=embeddings,
            memory=features,
            tgt_mask=tgt_mask,
            memory_mask=None,  # 可选：如果需要可以添加memory的mask
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=None  # 可选
        )

        # 5. 投影到词表维度
        outputs = self.linear(outputs)  # (batch_size, seq_len_dec, vocab_size)

        return outputs

    def _generate_square_subsequent_mask(self, sz):
        """
        生成因果掩码（下三角矩阵）
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def generate(self, features, vocab, max_len=50, device='cuda', beam_width=1):
        """
        自回归生成caption（用于推理）

        Args:
            features: (batch_size, seq_len_enc, embed_size) 图像特征
            vocab: Vocabulary对象
            max_len: 最大生成长度
            device: 计算设备
            beam_width: beam search宽度（1=greedy）
        Returns:
            list of generated captions
        """
        self.eval()
        batch_size = features.size(0)

        # 初始输入：<SOS>
        captions = torch.full((batch_size, 1), vocab.stoi['<SOS>'],
                              dtype=torch.long, device=device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        results = [[] for _ in range(batch_size)]

        with torch.no_grad():
            for _ in range(max_len):
                # 前向传播（每次传入完整历史）
                outputs = self.forward(features, captions)  # (batch, cur_len, vocab)

                # 取最后一个位置的预测
                next_token_logits = outputs[:, -1, :]  # (batch, vocab)
                next_tokens = next_token_logits.argmax(dim=-1)  # (batch,)

                # 更新已完成的序列
                for i in range(batch_size):
                    if not finished[i]:
                        token = next_tokens[i].item()
                        if token == vocab.stoi['<EOS>']:
                            finished[i] = True
                        else:
                            results[i].append(vocab.itos[token])

                # 如果全部完成，提前退出
                if finished.all():
                    break

                # 扩展captions用于下一次迭代
                captions = torch.cat([captions, next_tokens.unsqueeze(1)], dim=1)

        return [' '.join(tokens) for tokens in results]


# 【transformer所需位置编码】
class PositionalEncoding(nn.Module):
    """Transformer标准位置编码"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 【图像描述模型】
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout,
                 max_seq_len=100
                 ):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size, dropout) # 使用简化版
        self.decoder = DecoderTransformer(embed_size, hidden_size, vocab_size, num_layers, dropout,max_seq_len)

    def forward(self, images, captions):
        # 前向传播
        features = self.encoder(images) # (batch, num_patches, embed_size)
        # print("Features:", features)  # 打印features检查是否为None
        outputs = self.decoder(features, captions) # (batch, seq_len, vocab_size)
        # print("Outputs:", outputs)  # 打印outputs检查是否为None
        # breakpoint()
        # print("前向传播完成")
        # breakpoint()
        return outputs

    def generate_caption(self, image, vocab, max_len=50, device='cuda'):
        """单张图片生成caption的便捷方法"""
        self.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(device)  # (1, 3, H, W)
            features = self.encoder(image)  # (1, num_patches, embed)
            captions = self.decoder.generate(features, vocab, max_len, device)
            return captions[0]  # 返回字符串
