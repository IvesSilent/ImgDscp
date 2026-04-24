# visualize_log.py
# -*- coding: utf-8 -*-
"""
从训练日志提取数据并生成可视化图表
用法: python visualize_log.py --log_path log/train_2026-03-27_00-43-28.txt
"""

import re
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime


def parse_log(log_path):
    """解析日志文件，提取训练指标"""
    train_losses = []
    val_losses = []
    bleu_scores = []
    rouge_l_scores = []
    cider_scores = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # 匹配训练损失 (Average Train Loss)
            train_match = re.search(r'Epoch \d+, Average Train Loss: ([\d.]+)', line)
            if train_match:
                train_losses.append(float(train_match.group(1)))

            # 匹配验证损失
            val_match = re.search(r'Epoch \d+, Validation Loss: ([\d.]+)', line)
            if val_match:
                val_losses.append(float(val_match.group(1)))

            # 匹配BLEU
            bleu_match = re.search(r'Epoch \d+, BLEU Score: ([\d.]+)', line)
            if bleu_match:
                bleu_scores.append(float(bleu_match.group(1)))

            # 匹配ROUGE-L
            rouge_match = re.search(r'Epoch \d+, ROUGE-L Score: ([\d.]+)', line)
            if rouge_match:
                rouge_l_scores.append(float(rouge_match.group(1)))

            # 匹配CIDEr
            cider_match = re.search(r'Epoch \d+, CIDEr Score: ([\d.]+)', line)
            if cider_match:
                cider_scores.append(float(cider_match.group(1)))

    return train_losses, val_losses, bleu_scores, rouge_l_scores, cider_scores


def plot_results(train_losses, val_losses, bleu_scores, rouge_l_scores, cider_scores,
                 output_dir='Result_Fig', num_epochs=None):
    """绘制并保存图表"""

    # 自动创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 从日志文件名或当前时间生成时间戳
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if num_epochs is None:
        num_epochs = len(val_losses)

    # 绘制损失图
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training and Validation Loss (Total {num_epochs} Epochs)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 标注最佳验证点
    best_val_epoch = val_losses.index(min(val_losses)) + 1
    best_val_loss = min(val_losses)
    plt.axvline(x=best_val_epoch, color='g', linestyle='--', alpha=0.5,
                label=f'Best Val @ Epoch {best_val_epoch}')
    plt.scatter([best_val_epoch], [best_val_loss], color='green', s=100, zorder=5)

    loss_plot_path = os.path.join(output_dir, f'loss_plot_epoch_{num_epochs}_{current_time}.png')
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 损失图已保存: {loss_plot_path}")

    # 绘制NLP指标图
    plt.figure(figsize=(12, 6))
    epochs_metrics = range(1, len(bleu_scores) + 1)

    plt.plot(epochs_metrics, bleu_scores, 'b-', linewidth=2, label='BLEU Score', marker='o', markersize=4)
    plt.plot(epochs_metrics, rouge_l_scores, 'g-', linewidth=2, label='ROUGE-L Score', marker='s', markersize=4)
    plt.plot(epochs_metrics, cider_scores, 'r-', linewidth=2, label='CIDEr Score', marker='^', markersize=4)

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'NLP Metric Scores Across Epochs (Total {num_epochs} Epochs)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    metrics_plot_path = os.path.join(output_dir, f'nlp_metrics_{num_epochs}_{current_time}.png')
    plt.tight_layout()
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 指标图已保存: {metrics_plot_path}")

    # 打印统计信息
    print(f"\n{'=' * 50}")
    print("训练统计摘要:")
    print(f"{'=' * 50}")
    print(f" - 总轮数: {num_epochs}")
    print(f" - 训练损失: 初始={train_losses[0]:.4f}, 最终={train_losses[-1]:.4f}, 最佳={min(train_losses):.4f}")
    print(
        f" - 验证损失: 初始={val_losses[0]:.4f}, 最终={val_losses[-1]:.4f}, 最佳={min(val_losses):.4f} @ Epoch {best_val_epoch}")
    print(f" - BLEU: 初始={bleu_scores[0]:.4f}, 最终={bleu_scores[-1]:.4f}, 最佳={max(bleu_scores):.4f}")
    print(f" - ROUGE-L: 初始={rouge_l_scores[0]:.4f}, 最终={rouge_l_scores[-1]:.4f}, 最佳={max(rouge_l_scores):.4f}")
    print(f" - CIDEr: 初始={cider_scores[0]:.4f}, 最终={cider_scores[-1]:.4f}, 最佳={max(cider_scores):.4f}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training log')
    parser.add_argument('--log_path', type=str,
                        default='log/train_2026-04-15_01-19-45.txt',
                        help='Path to training log file')
    parser.add_argument('--output_dir', type=str, default='Result_Fig',
                        help='Directory to save plots')

    args = parser.parse_args()

    print(f"正在解析日志: {args.log_path}")

    if not os.path.exists(args.log_path):
        print(f"错误: 日志文件不存在: {args.log_path}")
        print("请检查路径是否正确，或使用 --log_path 指定正确路径")
        return

    # 解析数据
    train_losses, val_losses, bleu_scores, rouge_l_scores, cider_scores = parse_log(args.log_path)

    # 验证数据
    if len(val_losses) == 0:
        print("错误: 未能从日志中解析到验证数据，请检查日志格式")
        return

    print(f"解析成功:")
    print(f"  - 训练损失记录: {len(train_losses)} 条")
    print(f"  - 验证指标记录: {len(val_losses)} 条")

    # 绘制图表
    plot_results(train_losses, val_losses, bleu_scores,
                 rouge_l_scores, cider_scores, args.output_dir)

    print(f"\n可视化完成！图表保存在: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()