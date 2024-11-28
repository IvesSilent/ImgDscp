# -*coding=utf-8*-

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
from ImgDataset import Flickr30kDataset, Vocabulary
from model import ImageCaptioningModel
import argparse
import datetime
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from nltk.translate.bleu_score import SmoothingFunction
import re
import torch.nn.init as init
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_


# 加载词表(vocab)
def load_vocab(vocab_path, captions_file, freq_threshold):
    if os.path.exists(vocab_path):
        vocab = torch.load(vocab_path)
    else:
        # Create a temporary dataset to build vocab
        temp_dataset = Flickr30kDataset(root_dir='', captions_file=captions_file, freq_threshold=freq_threshold)
        vocab = temp_dataset.vocab
        torch.save(vocab, vocab_path)
    return vocab


# 验证函数
def validate(model, val_loader, criterion, device, vocab, tqdm_bar):
    model.eval()
    total_loss = 0
    references = []
    hypotheses = []

    with torch.no_grad():
        for imgs, captions in tqdm(val_loader, desc="Validating"):
            imgs, captions = imgs.to(device), captions[:, :-1].to(device)
            outputs = model(imgs, captions[:, :-1])  # 移除 <EOS> token

            # print(outputs.shape)  # 应该显示 [batch_size, seq_length, vocab_size]
            # print(captions[:, 1:].shape)  # 应该显示 [batch_size, seq_length]
            # breakpoint()

            # # 改变 outputs 的维度顺序从 [seq_length, batch_size, vocab_size] 到 [batch_size, seq_length, vocab_size]
            # outputs = outputs.permute(1, 0, 2)  # Now outputs.shape should be [32, 21, 9955]

            # 如果 outputs 的序列长度比 captions 长，我们需要对 outputs 进行切片
            # if outputs.shape[1] > captions[:, 1:].shape[1]:
            #     outputs = outputs[:, :captions[:, 1:].shape[1],·
            #               :]  # Adjust sequence length of outputs to match captions

            outputs = outputs[:, :captions.size(1), :]  # 调整 outputs 的序列长度

            # print(outputs.shape)  # 应该显示 [batch_size, seq_length, vocab_size]
            # print(captions[:, 1:].shape)  # 应该显示 [batch_size, seq_length]
            # breakpoint()

            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].reshape(-1))
            total_loss += loss.item()

            tqdm_bar.set_postfix(loss=loss.item())

            # 生成预测结果
            predicted_ids = outputs.argmax(-1)
            for i in range(len(captions)):
                real_caption = [vocab.itos[idx.item()] for idx in captions[i] if
                                idx not in [vocab.stoi['<PAD>'], vocab.stoi['<SOS>'], vocab.stoi['<EOS>']]]
                pred_caption = [vocab.itos[idx.item()] for idx in predicted_ids[i] if
                                idx not in [vocab.stoi['<PAD>'], vocab.stoi['<SOS>'], vocab.stoi['<EOS>']]]
                references.append([real_caption])
                hypotheses.append(pred_caption)

    # 计算评估指标
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)

    # for ref, hyp in zip(references, hypotheses):
    #     print("Reference:", ref[0])  # 应该打印出单词列表
    #     print("Hypothesis:", hyp)  # 应该打印出单词列表
    #     break  # 只查看第一对进行验证
    # breakpoint()

    # nltk.download('wordnet')
    # meteor_scores = [meteor_score([" ".join(ref[0])], " ".join(hyp)) for ref, hyp in zip(references, hypotheses)]
    # meteor_scores = [meteor_score([ref[0]], hyp) for ref, hyp in zip(references, hypotheses)]
    #
    # meteor_score_avg = sum(meteor_scores) / len(meteor_scores)
    rouge = Rouge()

    # 在计算 ROUGE 分数之前过滤空的假设
    filtered_hypotheses = [" ".join(hyp) if hyp else "empty_hypothesis" for hyp in hypotheses]
    filtered_references = [" ".join(ref[0]) if ref[0] else "empty_reference" for ref in references]

    rouge_scores = rouge.get_scores(filtered_hypotheses, filtered_references, avg=True)

    # rouge_scores = rouge.get_scores([" ".join(hyp) for hyp in hypotheses], [" ".join(ref[0]) for ref in references],
    #                                 avg=True)
    cider = Cider()
    # _, cider_score = cider.compute_score({i: [" ".join(ref[0])] for i, ref in enumerate(references)},
    #                                      {i: [" ".join(hyp)] for i, hyp in enumerate(hypotheses)})
    cider_score, _ = cider.compute_score({i: [" ".join(ref[0])] for i, ref in enumerate(references)},
                                         {i: [" ".join(hyp)] for i, hyp in enumerate(hypotheses)})

    print(f"Validation Loss: {total_loss / len(val_loader)}")
    print(f"BLEU Score: {bleu_score}")
    # print(f"METEOR Score: {meteor_score_avg}")
    print(f"ROUGE-L Score: {rouge_scores['rouge-l']['f']}")
    print(f"CIDEr Score: {cider_score}")

    # return total_loss / len(val_loader), bleu_score, meteor_score_avg, rouge_scores, cider_score
    return total_loss / len(val_loader), bleu_score, rouge_scores, cider_score


def my_collate(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)

    # Pad the captions to the length of the longest caption
    captions = pad_sequence(captions, batch_first=True, padding_value=0)  # 0是<PAD>的索引

    return images, captions


# 初始化模型权重
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Embedding):
        init.normal_(m.weight, mean=0, std=0.02)
    elif isinstance(m, nn.MultiheadAttention):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param)
            else:
                init.normal_(param, mean=0, std=0.02)


# 【优化项】在交叉熵损失中引入标签平滑
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon, ignore_index=0):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def forward(self, input, target):
        log_prob = nn.functional.log_softmax(input, dim=-1)
        # 计算平滑权重
        weight = input.new_ones(input.size()) * self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        # 忽略<PAD>
        weight[target == self.ignore_index] = 0
        loss = (-weight * log_prob).sum(dim=-1)
        # 仅计算非<PAD>部分的损失
        loss = loss[target != self.ignore_index].mean()
        return loss


def train(log_file, model, train_loader, val_loader, criterion, optimizer, num_epochs, device,
          last_epoch, scheduler, scaler, clip_norm, min_loss):
    # 在训练循环中记录log
    with open(log_file, 'a') as f:
        # 训练模型
        for epoch in range(num_epochs):
            total_loss = 0
            model.train()
            tqdm_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            for step, (imgs, captions) in enumerate(tqdm_bar):
                imgs, captions = imgs.to(device), captions.to(device)

                optimizer.zero_grad()

                # 【优化项】使用混合精度训练
                # 将autocast上下文管理器应用于模型的前向传播，以自动将计算转换为半精度
                with autocast():
                    outputs = model(imgs, captions[:, :-1])  # 移除 <EOS> token
                    loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].contiguous().view(-1))

                if torch.isnan(loss) or torch.isinf(loss):
                    print("NaN or Inf loss detected")
                    print("Captions:", captions)
                    print("Outputs:", outputs)
                    # 可以选择停止训练
                    break
                    # 或者返回一个特定的值，例如：
                    # loss = torch.tensor(0.0, device=device)
                # # 反向传播
                # loss.backward()

                # 【优化项】使用混合精度训练
                # 将autocast上下文管理器应用于模型的前向传播，以自动将计算转换为半精度
                scaler.scale(loss).backward()

                clip_grad_norm_(model.parameters(), clip_norm)  # 梯度裁剪
                scaler.step(optimizer)
                scaler.update()

                # # 更新优化器和学习率
                # optimizer.step()
                # scheduler.step()

                total_loss += loss.item()
                tqdm_bar.set_postfix(loss=loss.item())
                # 记录每个step的训练loss

                f.write(f"Epoch {epoch + 1}, Step {step + 1}, Train Loss: {loss.item()}\n")

            scheduler.step()

            # 平均损失
            avg_loss = total_loss / len(train_loader)
            f.write(f"Epoch {epoch + 1}, Average Train Loss: {avg_loss}\n")
            train_losses.append(avg_loss)  # 添加这行来保存训练损失

            val_loss, bleu_score, rouge_scores, cider_score = validate(model, val_loader, criterion, device,
                                                                       val_dataset.vocab, tqdm_bar)
            f.write(f"Epoch {epoch + 1}, Validation Loss: {val_loss}\n")
            f.write(f"Epoch {epoch + 1}, BLEU Score: {bleu_score}\n")
            f.write(f"Epoch {epoch + 1}, ROUGE-L Score: {rouge_scores['rouge-l']['f']}\n")
            f.write(f"Epoch {epoch + 1}, CIDEr Score: {cider_score}\n")

            val_losses.append(val_loss)
            bleu_scores.append(bleu_score)
            # meteor_scores.append(meteor_score_avg)
            rouge_l_scores.append(rouge_scores['rouge-l']['f'])
            cider_scores.append(cider_score)

            # 检查是否达到新的最低验证损失
            if val_loss < min_loss:
                min_loss = val_loss
                early_stop_count = 0  # 重置早停计数器
                current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                if pretrain:
                    best_model_save_path = f'./models/model_epoch_{num_old_epoch + epoch + 1}_{current_time}.pth'
                else:
                    best_model_save_path = f'./models/model_epoch_{epoch + 1}_{current_time}.pth'
                # best_model_save_path = f'./models/model_epoch_{epoch + 1}_{current_time}.pth'

                os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
                torch.save(model.state_dict(), best_model_save_path)
                print(f"Best model saved to {best_model_save_path}")
            else:
                early_stop_count += 1

            # 早停检查
            if early_stop_count >= early_stop_limit:
                print("Early stopping triggered.\n已触发早停机制。")
                last_epoch = epoch + 1
                break

    # 不论如何都保存最终模型
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # model_save_path = f'./models/model_epoch_{last_epoch}_{current_time}.pth'
    if pretrain:
        last_epoch = last_epoch + num_old_epoch

    model_save_path = f'./models/model_epoch_{last_epoch + 1}_{current_time}.pth'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\t{num_epochs}ep 训练完成")
    print(f"\t模型已保存至 {model_save_path}\n")


if __name__ == "__main__":
    # ##############################################################################
    # Phase_0 - 参数设置与传递
    ################################################################################
    print("开始执行：Phase_0<参数设置与传递>")
    parser = argparse.ArgumentParser(description='Training: CNN_Transformer Model for Image Captioning')
    parser.add_argument('--train_root_dir', type=str, default='data/train/train_img',
                        help='directory of train images')
    parser.add_argument('--train_captions_file', type=str, default='data/train/train.token',
                        help='Path to the captions file for training')
    parser.add_argument('--val_root_dir', type=str, default='data/val/val_img',
                        help='directory of train images')
    parser.add_argument('--val_captions_file', type=str, default='data/val/val.token',
                        help='Path to the captions file for training')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=80,
                        help='amount of epochs')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='size of embedding')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='size of hidden')
    parser.add_argument('--vocab_size', type=int, default=8943,
                        help='size of vocabulary')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='amount of layers')
    parser.add_argument('--freq_threshold', type=int, default=1,
                        help='threshold frequent')
    parser.add_argument('--early_stop_count', type=int, default=0,
                        help='early stop count')
    parser.add_argument('--early_stop_limit', type=int, default=80,
                        help='early stop limit')

    parser.add_argument('--clip_norm', type=float, default=1.0,
                        help='threshold of grad clip')
    parser.add_argument('--smooth_epsilon', type=float, default=0.05,
                        help='label smooth epsilon')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout rate of the model')

    parser.add_argument('--pretrain', action='store_true',
                        help='load trained models')
    parser.add_argument('--unfrozen', action='store_true',
                        help='Unfrozen the model')
    parser.add_argument('--preModel', type=str, default='models/model_epoch_5_2024-11-28_02-08-37.pth',
                        help='path to trained models')
    parser.add_argument('--vocab_path', type=str, default='vocab.pth',
                        help='path to local vocab')

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.lr
    dropout = args.dropout
    num_epochs = args.num_epochs
    embed_size = args.embed_size
    hidden_size = args.hidden_size
    vocab_size = args.vocab_size  # 根据实际情况调整
    num_layers = args.num_layers
    freq_threshold = args.freq_threshold
    clip_norm = args.clip_norm
    smooth_epsilon = args.smooth_epsilon  # 您可以根据需要调整这个值

    train_root_dir = args.train_root_dir
    train_captions_file = args.train_captions_file
    val_root_dir = args.val_root_dir
    val_captions_file = args.val_captions_file

    pretrain = args.pretrain
    preModel = args.preModel
    unfrozen = args.unfrozen
    vocab_path = args.vocab_path

    # 早停参数
    early_stop_count = args.early_stop_count
    early_stop_limit = args.early_stop_limit
    min_loss = float('inf')

    # 如可用则启用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置训练集数据转换
    transform_train = transforms.Compose([
        transforms.Resize((299, 299)),  # 调整为Inception模型推荐的最小尺寸
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
        transforms.RandomResizedCrop(size=(299, 299), scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # 随机裁剪和缩放
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 设置验证集数据转换
    transform_val = transforms.Compose([
        transforms.Resize((299, 299)),  # 调整为Inception模型推荐的最小尺寸
        # transforms.RandomHorizontalFlip(),  # 随机水平翻转
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
        # transforms.RandomResizedCrop(size=(299, 299), scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # 随机裁剪和缩放
        # transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 在训练循环外部定义两个列表来存储训练和验证损失
    # 在每个epoch后更新它们
    train_losses = []
    val_losses = []

    bleu_scores = []
    # meteor_scores = []
    rouge_l_scores = []
    cider_scores = []

    print("\t传参完成\n")

    # ##############################################################################
    # Phase_1 - 数据加载
    ################################################################################
    print("开始执行：Phase_1<数据加载>")

    # 加载或创建词表
    vocab = load_vocab(vocab_path, args.train_captions_file, freq_threshold)

    # 加载训练数据集
    train_dataset = Flickr30kDataset(
        root_dir=train_root_dir,
        captions_file=train_captions_file,
        transform=transform_train,
        freq_threshold=freq_threshold,
        vocab=vocab
    )
    #
    # 加载验证数据集
    val_dataset = Flickr30kDataset(
        root_dir=val_root_dir,  # 确保这里的路径正确
        captions_file=val_captions_file,
        transform=transform_train,
        freq_threshold=freq_threshold,
        vocab=vocab
    )

    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                              collate_fn=my_collate, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                            collate_fn=my_collate, drop_last=True)

    print("\t数据加载完成\n")

    # ##############################################################################
    # Phase_2 - 初始化
    ################################################################################
    print("开始执行：Phase_2<模型初始化>")

    # 初始化模型
    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers, dropout=dropout).to(device)

    # 【优化项】修改模型权重的初始化策略
    model.apply(weights_init)

    # # 定义损失函
    # criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
    # 【优化项】引入标签平滑
    # 定义损失函数，smooth_epsilon是标签平滑的参数，ignore_index是<PAD>的索引

    ignore_index = train_dataset.vocab.stoi["<PAD>"]  # <PAD>的索引
    criterion = LabelSmoothingCrossEntropy(smooth_epsilon, ignore_index=ignore_index)

    # 【优化项】修改优化器为Adam优化器，采用权重衰减
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 【优化项】在 train.py 中添加学习率调度器
    # https://zhuanlan.zhihu.com/p/538447997
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.000001)  # 余弦退火
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # 指数下降
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # 【优化项】采用混合精度计算
    # 初始化GradScaler，用于调整梯度的缩放，以防止在半精度下进行反向传播时出现梯度下溢
    scaler = GradScaler()

    # 如需要，加载预训练模型
    pretrain = True
    # unfrozen = True
    # print(pretrain)
    # breakpoint()

    if pretrain:
        model_filepath = preModel
        # 从文件名中提取 epoch 数，假设文件名格式为 "model_epoch_XX_YYYY-MM-DD_HH-MM-SS.pth"
        match = re.search(r'model_epoch_(\d+)_', preModel)
        if match:
            num_old_epoch = int(match.group(1))
        else:
            num_old_epoch = 0  # 如果没有找到，设置为0或其他默认值
        if os.path.exists(model_filepath):
            model.load_state_dict(torch.load(model_filepath, map_location=device), strict=False)
            print(f"\t已从{model_filepath}加载预训练模型权重。\n\t已应用保存的模型权重到当前模型。")
            # 重置dropout
            for name, module in model.named_modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout
        else:
            print(
                f"Warning: Pretrained model file not found at {model_filepath}. Starting training with a fresh model.")

    # 模型解冻
    if unfrozen:
        print("\t已确认需解冻模型编码器特征提取器")
        for param in model.encoder.feature_extractor.parameters():
            param.requires_grad = True
        print("\t解冻完成")

    print("\t初始化完成\n")

    # ##############################################################################
    # Phase_3 - 模型训练
    ################################################################################

    print("开始执行：Phase_3<模型训练>")

    # 创建日志目录
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建日志文件
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f'train_{current_time}.txt')

    # # 当前时间
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    last_epoch = 0

    # 使用训练函开始训练
    train(log_file, model, train_loader, val_loader, criterion, optimizer, num_epochs, device,
          last_epoch, scheduler, scaler, clip_norm, min_loss)

    # # 在训练循环中记录log
    # with open(log_file, 'a') as f:
    #     # 训练模型
    #     for epoch in range(num_epochs):
    #         total_loss = 0
    #         model.train()
    #         tqdm_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
    #         for step, (imgs, captions) in enumerate(tqdm_bar):
    #             imgs, captions = imgs.to(device), captions.to(device)
    #
    #             optimizer.zero_grad()
    #
    #             # # 前向传播
    #             # outputs = model(imgs, captions[:, :-1])  # 移除 <EOS> token
    #             # # outputs: (batch_size, seq_len, vocab_size)
    #             # # outputs.reshape(-1, outputs.shape[2]): (batch_size * seq_len, vocab_size)
    #             # # captions: (batch_size, seq_len+1) 多一个之前被切掉的<EOS>
    #             # # captions[:, 1:].contiguous().view(-1): (batch_size * seq_len)
    #             #
    #             # # loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].reshape(-1))
    #             # loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].contiguous().view(-1))
    #
    #             # 【优化项】使用混合精度训练
    #             # 将autocast上下文管理器应用于模型的前向传播，以自动将计算转换为半精度
    #             with autocast():
    #                 outputs = model(imgs, captions[:, :-1])  # 移除 <EOS> token
    #                 loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].contiguous().view(-1))
    #
    #             if torch.isnan(loss) or torch.isinf(loss):
    #                 print("NaN or Inf loss detected")
    #                 print("Captions:", captions)
    #                 print("Outputs:", outputs)
    #                 # 可以选择停止训练
    #                 break
    #                 # 或者返回一个特定的值，例如：
    #                 # loss = torch.tensor(0.0, device=device)
    #             # # 反向传播
    #             # loss.backward()
    #
    #             # 【优化项】使用混合精度训练
    #             # 将autocast上下文管理器应用于模型的前向传播，以自动将计算转换为半精度
    #             scaler.scale(loss).backward()
    #
    #             clip_grad_norm_(model.parameters(), clip_norm)  # 梯度裁剪
    #             scaler.step(optimizer)
    #             scaler.update()
    #
    #             # # 更新优化器和学习率
    #             # optimizer.step()
    #             # scheduler.step()
    #
    #             total_loss += loss.item()
    #             tqdm_bar.set_postfix(loss=loss.item())
    #             # 记录每个step的训练loss
    #
    #             f.write(f"Epoch {epoch + 1}, Step {step + 1}, Train Loss: {loss.item()}\n")
    #
    #         scheduler.step()
    #
    #         # 平均损失
    #         avg_loss = total_loss / len(train_loader)
    #         f.write(f"Epoch {epoch + 1}, Average Train Loss: {avg_loss}\n")
    #         train_losses.append(avg_loss)  # 添加这行来保存训练损失
    #
    #         val_loss, bleu_score, rouge_scores, cider_score = validate(model, val_loader, criterion, device,
    #                                                                    val_dataset.vocab)
    #         f.write(f"Epoch {epoch + 1}, Validation Loss: {val_loss}\n")
    #         f.write(f"Epoch {epoch + 1}, BLEU Score: {bleu_score}\n")
    #         f.write(f"Epoch {epoch + 1}, ROUGE-L Score: {rouge_scores['rouge-l']['f']}\n")
    #         f.write(f"Epoch {epoch + 1}, CIDEr Score: {cider_score}\n")
    #
    #         val_losses.append(val_loss)
    #         bleu_scores.append(bleu_score)
    #         # meteor_scores.append(meteor_score_avg)
    #         rouge_l_scores.append(rouge_scores['rouge-l']['f'])
    #         cider_scores.append(cider_score)
    #
    #         # 检查是否达到新的最低验证损失
    #         if val_loss < min_loss:
    #             min_loss = val_loss
    #             early_stop_count = 0  # 重置早停计数器
    #             current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #
    #             if pretrain:
    #                 best_model_save_path = f'./models/model_epoch_{num_old_epoch + epoch + 1}_{current_time}.pth'
    #             else:
    #                 best_model_save_path = f'./models/model_epoch_{epoch + 1}_{current_time}.pth'
    #             # best_model_save_path = f'./models/model_epoch_{epoch + 1}_{current_time}.pth'
    #
    #             os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
    #             torch.save(model.state_dict(), best_model_save_path)
    #             print(f"Best model saved to {best_model_save_path}")
    #         else:
    #             early_stop_count += 1
    #
    #         # 早停检查
    #         if early_stop_count >= early_stop_limit:
    #             print("Early stopping triggered.\n已触发早停机制。")
    #             last_epoch = epoch + 1
    #             break
    #
    # # 不论如何都保存最终模型
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # # model_save_path = f'./models/model_epoch_{last_epoch}_{current_time}.pth'
    # if pretrain:
    #     last_epoch = last_epoch + num_old_epoch
    #
    # model_save_path = f'./models/model_epoch_{last_epoch + 1}_{current_time}.pth'
    # os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # torch.save(model.state_dict(), model_save_path)
    # print(f"\t{num_epochs}ep 训练完成")
    # print(f"\t模型已保存至 {model_save_path}\n")

    # ##############################################################################
    # Phase_4 - 训练结果可视化
    ################################################################################
    print("开始执行：Phase_4<训练结果可视化>")

    # 绘制损失图
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.figure(figsize=(10, 5))
    # epochs = range(1, num_epochs + 1)
    # steps_per_epoch = len(train_loader)
    # step_numbers = [i for epoch in epochs for i in range(steps_per_epoch)]  # 生成每个epoch的step编号

    # plt.plot(step_numbers, step_train_losses, label='Training Loss', marker='o')  # 使用每个step的训练loss
    # plt.plot(val_losses, label='Validation Loss', marker='o')  # 使用每个epoch的验证loss
    # plt.xlabel('Epochs/Steps')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.xticks(step_numbers[::steps_per_epoch], [f"Epoch {i + 1}" for i in range(num_epochs)])  # 设置横坐标大刻度为epoch
    # plt.legend()
    # plt.savefig(f'Result_Fig/loss_plot_epoch_{num_epochs}_{current_time}.png')

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.savefig(f'Result_Fig/loss_plot_epoch_{num_epochs}_{current_time}.png')

    # 绘制NLP指标图
    plt.figure(figsize=(10, 5))
    plt.plot(bleu_scores, label='BLEU Score')
    # plt.plot(meteor_scores, label='METEOR Score')
    plt.plot(rouge_l_scores, label='ROUGE-L Score')
    plt.plot(cider_scores, label='CIDEr Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('NLP Metric Scores Across Epochs')
    plt.legend()
    plt.savefig(f'Result_Fig/nlp_metrics_{num_epochs}_{current_time}.png')

    plt.show()

    print(f"训练评估指标保存至 Result_Fig/ 目录")
