# predict.py
# -*coding=utf-8*-
import torch.nn.functional as F
import torch
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime
from model import ImageCaptioningModel
from ImgDataset import Vocabulary
import argparse
import random

# 标注生成函数
def generate_caption(image, model, vocab, device):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # 添加批次维度并移动到设备
        features = model.encoder(image)
        captions = torch.tensor([vocab.stoi['<SOS>']]).unsqueeze(0).to(device)

        result_caption = []

        for _ in range(50):  # 假设最长描述长度不超过50个词
            output = model.decoder(features, captions)
            # output = output.permute(1, 0, 2)  # 调整维度以匹配预期的 [batch_size, seq_len, vocab_size]
            # print(output)
            # breakpoint()
            # print(output.shape)
            # breakpoint()
            # print(captions.shape)
            # breakpoint()

            predicted = output.argmax(-1)[:, -1]  # 获取序列中最后一个预测的词
            predicted_word = vocab.itos[predicted.item()]

            if predicted_word == '<EOS>':
                break
            result_caption.append(predicted_word)
            # print(f"captions = {captions}")
            # print(f"predicted = {predicted}")
            # print(f"predicted.unsqueeze(0) = {predicted.unsqueeze(0)}")
            # print(f"predicted.unsqueeze(0).unsqueeze(0) = {predicted.unsqueeze(0).unsqueeze(0)}")
            # breakpoint()

            captions = torch.cat((captions, predicted.unsqueeze(0)), dim=1)  # 更新captions以包含最新预测的词

        return ' '.join(result_caption)


# 标注写入函数
def write_generate_caption(output_filepath, image_filename, image, model, vocab, device, num_caption=5):
    print(f"对{image_filename}生成的{num_caption}句描述如下：")
    with open(output_filepath, 'w') as file:
        model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(device)  # 添加批次维度并移动到设备
            features = model.encoder(image)

            # 进行num_caption次描述
            for i in range(num_caption):
                captions = torch.tensor([vocab.stoi['<SOS>']]).unsqueeze(0).to(device)
                result_caption = []

                # 进行50次取词
                for _ in range(50):  # 假设最长描述长度不超过50个词


                    output = model.decoder(features, captions)
                    # print(output.shape)
                    # print(output.argmax(-1).shape)
                    # breakpoint()

                    predicted = output.argmax(-1)[:, -1]  # 获取序列中最后一个预测的词
                    # print(f"第一次时，captions.shape[1] = {captions.shape[1]}")
                    # print(captions)
                    # breakpoint()

                    # print(f"captions = {captions}")
                    # print(f"output = {output}")
                    # breakpoint()

                    # 仅在预测第一个词时，替换predicted
                    if captions.shape[1] == 1:
                        top_probs, top_idx = output[:, 0].topk(num_caption)
                        # print(f"top_probs = {top_probs}")
                        # print(f"top_idx = {top_idx}")
                        # breakpoint()

                        predicted = torch.tensor([top_idx[0][i]]).to(device)

                        # 防止描述为空
                        if vocab.itos[predicted.item()] == '<EOS>':
                            tp, ti = output[:, 0].topk(num_caption + 1)
                            predicted = torch.tensor([ti[0][num_caption]]).to(device)

                    predicted_word = vocab.itos[predicted.item()]

                    if predicted_word == '<EOS>':
                        break

                    result_caption.append(predicted_word)
                    # print(f"captions = {captions}")
                    # print(f"predicted = {predicted}")

                    captions = torch.cat((captions, predicted.unsqueeze(0)), dim=1)  # 更新captions以包含最新预测的词

                result_caption = ' '.join(result_caption)
                # print(result_caption)
                # breakpoint()
                file.write(f"{image_filename}#{i}\t{result_caption}\n")
                print(f"{image_filename}#{i}\t{result_caption}")


############################
# 【修改0327】生成5句多样化的描述
############################

def generate_diverse_captions(image, model, vocab, device, num_captions=5, max_len=50, strategy='top5',
                              temperature=1.0):
    """
    生成多样化的描述

    Args:
        image: 输入图像 (1, 3, 299, 299) 或 (3, 299, 299)
        model: 模型
        vocab: 词汇表
        device: 设备
        num_captions: 生成描述数量
        max_len: 最大长度
        strategy: 'top5' - 取前5名; 'random_top5' - 从前5随机采样; 'sampling' - 温度采样
        temperature: 采样温度（仅strategy='sampling'时有效）
    """
    model.eval()
    with torch.no_grad():
        # 确保图像维度正确
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(device)

        # 提取图像特征 (1, num_patches, embed_size)
        features = model.encoder(image)

        results = []

        # 第一步：获取第一个词的候选
        captions = torch.tensor([[vocab.stoi['<SOS>']]], dtype=torch.long, device=device)
        output = model.decoder(features, captions)  # (1, 1, vocab_size)

        # 获取第一个位置的预测概率
        logits = output[0, 0, :]  # (vocab_size,)
        probs = torch.softmax(logits / temperature, dim=-1)

        # 选择策略
        if strategy == 'top5':
            # 取概率最高的5个，排除特殊token
            top_probs, top_indices = torch.topk(probs, k=num_captions + 10)

            first_tokens = []
            for idx in top_indices:
                token = idx.item()
                word = vocab.itos[token]
                if word not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>', '']:
                    first_tokens.append(token)
                if len(first_tokens) >= num_captions:
                    break

        elif strategy == 'random_top5':
            # 从前10名中随机选5个
            top_probs, top_indices = torch.topk(probs, k=10)
            valid_indices = [idx.item() for idx in top_indices
                             if vocab.itos[idx.item()] not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>', '']]

            if len(valid_indices) < num_captions:
                valid_indices.extend([random.choice(valid_indices) for _ in range(num_captions - len(valid_indices))])

            # 按概率加权随机选择，或均匀随机
            first_tokens = random.choices(valid_indices, k=num_captions)

        elif strategy == 'sampling':
            # 纯采样策略：每次重新采样第一个词
            first_tokens = []
            for _ in range(num_captions):
                token = torch.multinomial(probs, 1).item()
                # 如果采样到特殊token，重新采样
                while vocab.itos[token] in ['<SOS>', '<EOS>', '<PAD>', '<UNK>', '']:
                    token = torch.multinomial(probs, 1).item()
                first_tokens.append(token)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 为每个first_token生成完整序列
        for i, first_token in enumerate(first_tokens):
            captions = torch.tensor([[vocab.stoi['<SOS>'], first_token]], dtype=torch.long, device=device)
            result_caption = [vocab.itos[first_token]]

            for _ in range(max_len - 1):
                output = model.decoder(features, captions)

                # 这里可以选择贪婪解码或继续采样
                predicted = output.argmax(-1)[:, -1]  # 贪婪解码，保证后续连贯性
                # 或者也可以用: predicted = torch.multinomial(torch.softmax(output[0, -1, :], dim=-1), 1)

                predicted_word = vocab.itos[predicted.item()]

                if predicted_word == '<EOS>':
                    break

                result_caption.append(predicted_word)
                captions = torch.cat((captions, predicted.unsqueeze(0)), dim=1)

            results.append(' '.join(result_caption))

    return results


if __name__ == "__main__":
    # ##############################################################################
    # Phase_0 - 参数设置与传递
    ##############################################################################
    print("开始执行：Phase_0<参数设置与传递>")

    parser = argparse.ArgumentParser(description='Generate captions for a single image')
    parser.add_argument('--image_path', type=str, default='data/test/test_img/256063.jpg',
                        help='path to the input image')
    parser.add_argument('--model_path', type=str,
                        default='models/0408_instance05/train_ep265-310/model_epoch_310_2026-04-14_07-23-53.pth',
                        help='path to trained model')
    parser.add_argument('--vocab_path', type=str, default='vocab.pth',
                        help='path to vocabulary object')
    parser.add_argument('--output_dir', type=str, default='singleprd',
                        help='directory to save the output text file')
    parser.add_argument('--num_captions', type=int, default=5,
                        help='number of captions to generate')
    parser.add_argument('--max_len', type=int, default=50,
                        help='maximum caption length')

    parser.add_argument('--strategy', type=str, default='top5',
                        choices=['top5', 'random_top5', 'sampling'],
                        help='diversity strategy: top5, random_top5, or sampling')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='sampling temperature for strategy=sampling')

    args = parser.parse_args()
    model_path = args.model_path
    vocab_path = args.vocab_path
    image_path = args.image_path
    output_dir = args.output_dir
    num_captions = args.num_captions
    max_len = args.max_len

    strategy = args.strategy
    temperature = args.temperature

    print("\t传参完成\n")

    # ##############################################################################
    # Phase_1 - 加载模型和数据
    ##############################################################################
    print("开始执行：Phase_1<加载模型和数据>")

    # 加载词汇表
    vocab = Vocabulary.load(vocab_path)

    # 加载模型和词汇表
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageCaptioningModel(embed_size=512, hidden_size=1024, vocab_size=len(vocab), num_layers=4, dropout=0.4)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()

    print("\t模型和词汇表加载完成\n")

    # # 打印词汇表内容
    # print("Vocabulary Table:")
    # for idx in range(len(vocab)):
    #     print(f"Index: {idx}, Word: {vocab.itos[idx]}")
    #
    #

    # 设置数据转换
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载图片
    image = Image.open(image_path).convert("RGB")
    image = transform(image)

    print("\t图片加载完成\n")

    # ##############################################################################
    # Phase_3 - 预测描述生成
    ##############################################################################
    print("开始执行：Phase_3<预测描述生成>")

    # 生成描述并保存到文件
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"{os.path.basename(image_path)}_{current_time}.txt"
    output_filepath = os.path.join(output_dir, output_filename)

    os.makedirs(output_dir, exist_ok=True)

    # with open(output_filepath, 'w') as file:
    #     for j in range(5):  # 生成五句不同的描述
    #
    #         caption = generate_caption(image, model, vocab, device)
    #         image_filename = os.path.basename(image_path)
    #
    #         file.write(f"{image_filename}#{j}\t{caption}\n")

    image_filename = os.path.basename(image_path)
    # image_batch = image.unsqueeze(0).to(device)  # (1, 3, 299, 299)

    # 使用新的多样性生成函数
    captions = generate_diverse_captions(
        image, model, vocab, device,
        num_captions=num_captions,
        max_len=max_len,
        strategy=strategy,
        temperature=temperature
    )

    print(f"对{image_filename}生成的{num_captions}句描述如下：")
    with open(output_filepath, 'w') as f:
        for i, caption in enumerate(captions):
            line = f"{image_filename}#{i}\t{caption}"
            f.write(line + "\n")
            print(f"\t{line}")

    #
    # with torch.no_grad():
    #     features = model.encoder(image_batch)
    #
    #     print(f"对{image_filename}生成的{num_captions}句描述如下：")
    #     with open(output_filepath, 'w') as f:
    #         for i in range(num_captions):
    #             caption = model.decoder.generate(
    #                 features,
    #                 vocab,
    #                 max_len=max_len,
    #                 device=device
    #             )[0]  # batch_size=1
    #
    #             line = f"{image_filename}#{i}\t{caption}"
    #             f.write(line + "\n")
    #             print(f"\t{line}")

    # write_generate_caption(output_filepath, image_filename, image, model, vocab, device, num_caption=5)

    print(f"\tCaptions generated and saved to {output_filepath}")
    print(f"\t描述已生成并保存至 {output_filepath}")
