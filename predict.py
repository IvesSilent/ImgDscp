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


if __name__ == "__main__":
    # ##############################################################################
    # Phase_0 - 参数设置与传递
    ##############################################################################
    print("开始执行：Phase_0<参数设置与传递>")

    parser = argparse.ArgumentParser(description='Generate captions for a single image')
    parser.add_argument('--image_path', type=str, default='data/test/test_img/256063.jpg',
                        help='path to the input image')
    parser.add_argument('--model_path', type=str,
                        default='models/model_epoch_186_2024-11-26_16-52-47.pth',
                        help='path to trained model')
    parser.add_argument('--vocab_path', type=str, default='vocab.pth',
                        help='path to vocabulary object')
    parser.add_argument('--output_dir', type=str, default='singleprd',
                        help='directory to save the output text file')

    args = parser.parse_args()
    model_path = args.model_path
    vocab_path = args.vocab_path
    image_path = args.image_path
    output_dir = args.output_dir

    print("\t传参完成\n")

    # ##############################################################################
    # Phase_1 - 加载模型和数据
    ##############################################################################
    print("开始执行：Phase_1<加载模型和数据>")

    # 加载模型和词汇表
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageCaptioningModel(embed_size=256, hidden_size=512, vocab_size=8943, num_layers=6, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    vocab = torch.load(vocab_path)

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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # with open(output_filepath, 'w') as file:
    #     for j in range(5):  # 生成五句不同的描述
    #
    #         caption = generate_caption(image, model, vocab, device)
    #         image_filename = os.path.basename(image_path)
    #
    #         file.write(f"{image_filename}#{j}\t{caption}\n")

    image_filename = os.path.basename(image_path)

    write_generate_caption(output_filepath, image_filename, image, model, vocab, device, num_caption=5)

    print(f"\tCaptions generated and saved to {output_filepath}")
    print(f"\t描述已生成并保存至 {output_filepath}")
