# -*coding=utf-8*-

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from model import ImageCaptioningModel
from ImgDataset import TestDataset, Vocabulary
import argparse
from tqdm import tqdm


# 标注生成函
def generate_caption(image, model, vocab, device):
    model.eval()
    with torch.no_grad():
        # print("Original image shape:", image.shape)  # 打印原始图像形状
        # image = image.unsqueeze(0)  # 添加批次维度
        # print("Image shape after unsqueeze:", image.shape)  # 打印添加批次维度后的形状
        #
        # # 确保图像是三通道的
        # if image.shape[1] != 3:
        #     raise ValueError("Image must have 3 channels (RGB)")

        image = image.to(device)
        features = model.encoder(image)
        captions = torch.tensor([vocab.stoi['<SOS>']]).unsqueeze(0).to(device)

        result_caption = []

        for _ in range(50):  # 假设最长描述长度不超过50个词
            output = model.decoder(features, captions)
            # output = output.permute(1, 0, 2)  # 调整维度以匹配预期的 [batch_size, seq_len, vocab_size]
            predicted = output.argmax(-1)[:, -1]  # 获取序列中最后一个预测的词
            predicted_word = vocab.itos[predicted.item()]

            if predicted_word == '<EOS>':
                break
            result_caption.append(predicted_word)
            captions = torch.cat((captions, predicted.unsqueeze(0)), dim=1)  # 更新captions以包含最新预测的词

        return ' '.join(result_caption)


# 标注写入函数
def write_generate_caption(output_filepath, image_filename, image, model, vocab, device, num_caption=5):
    with open(output_filepath, 'a') as file:
        model.eval()
        with torch.no_grad():

            features = model.encoder(image)

            # 进行num_caption次描述
            for i in range(num_caption):
                captions = torch.tensor([vocab.stoi['<SOS>']]).unsqueeze(0).to(device)
                result_caption = []

                # 进行50次取词
                for _ in range(50):  # 假设最长描述长度不超过50个词
                    output = model.decoder(features, captions)
                    predicted = output.argmax(-1)[:, -1]  # 获取序列中最后一个预测的词
                    # print(f"第一次时，captions.shape[1] = {captions.shape[1]}")
                    # print(captions)
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
                            tp,ti = output[:, 0].topk(num_caption+1)
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
                # print(f"{image_filename}#{i}\t{result_caption}")


if __name__ == "__main__":
    # ##############################################################################
    # Phase_0 - 参数设置与传递
    ################################################################################
    print("开始执行：Phase_0<参数设置与传递>")

    parser = argparse.ArgumentParser(description='Generate captions for test images')
    parser.add_argument('--test_root_dir', type=str, default='data/test/test_img',
                        help='directory of test images')
    parser.add_argument('--model_path', type=str,
                        default='models/modelArchive_2/model_epoch_170_2024-11-22_05-24-18.pth',
                        help='path to trained model')
    parser.add_argument('--vocab_path', type=str, default='vocab.pth',
                        help='path to vocabulary object')
    parser.add_argument('--output_file', type=str, default='data/test/test.token',
                        help='output file path to save generated captions')

    args = parser.parse_args()

    model_path = args.model_path
    vocab_path = args.vocab_path
    test_root_dir = args.test_root_dir
    output_file = args.output_file

    print("\t传参完成\n")

    # ##############################################################################
    # Phase_1 - 加载模型和数据
    ################################################################################
    print("开始执行：Phase_1<加载模型和数据>")

    # 加载模型和词汇表
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageCaptioningModel(embed_size=256, hidden_size=512, vocab_size=8943, num_layers=6, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    vocab = torch.load(vocab_path)

    print("\t模型和词汇表加载完成\n")

    # 设置数据转换
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # 加载数据集
    test_dataset = TestDataset(root_dir=test_root_dir, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    print("\t数据加载完成\n")

    # ##############################################################################
    # Phase_3 - 预测描述生成
    ################################################################################

    print("开始执行：Phase_3<预测描述生成>")

    # # 生成描述并保存到文件
    # with open(output_file, 'w') as file:
    #     # for i, (image, image_path) in enumerate(test_loader):
    #     for image, image_path in tqdm(test_loader, desc="Generating captions", total=len(test_loader)):
    #         # caption = generate_caption(image, model, vocab, device)
    #         # image_filename = os.path.basename(image_path[0])
    #         for j in range(5):  # 生成五句不同的描述
    #
    #             caption = generate_caption(image, model, vocab, device)
    #             image_filename = os.path.basename(image_path[0])
    #
    #             file.write(f"{image_filename}#{j}\t{caption}\n")

    for image, image_path in tqdm(test_loader, desc="Generating captions", total=len(test_loader)):
        image_filename = os.path.basename(image_path[0])
        image = image.to(device)  # 确保图像移动到设备

        # 调用write_generate_caption函数生成描述并保存
        write_generate_caption(output_file, image_filename, image, model, vocab, device, num_caption=5)

    print(f"Captions generated and saved to {output_file}")
    print(f"描述已生成并保存至 {output_file}")

    print(f"\t描述生成完成\n")
