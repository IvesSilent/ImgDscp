#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flickr30k 数据集处理脚本
功能：解包、解析标注、划分数据集、重组目录结构
"""

import os
import tarfile
import shutil
import random
import re
from pathlib import Path
from collections import defaultdict

# ==================== 配置区域 ====================
random.seed(42)  # 保证划分结果可复现

# 路径配置
DATASET_DIR = Path("Flickr30k")  # 原始下载文件位置
OUTPUT_DIR = Path("data")  # 输出目录（将创建 train/val/test）

# 标准Flickr30k划分 (官方标准：28000训练 / 1000验证 / 1000测试)
TRAIN_SIZE = 28000
VAL_SIZE = 1000
TEST_SIZE = 1000


# ================================================


def extract_tar_files():
    """解压tar.gz和tar文件"""
    print("=" * 60)
    print("步骤1: 解压数据集文件")
    print("=" * 60)

    extracted_dirs = []

    # 解压标注文件
    token_tar = DATASET_DIR / "Flickr30k.tar.gz"
    if token_tar.exists():
        print(f"📦 解压 {token_tar.name}...")
        with tarfile.open(token_tar, "r:gz") as tar:
            tar.extractall(DATASET_DIR)
        print("✅ 标注文件解压完成")

    # 解压图片
    images_tar = DATASET_DIR / "Flickr30k-images.tar"
    if images_tar.exists():
        print(f"📦 解压 {images_tar.name}...")
        with tarfile.open(images_tar, "r") as tar:
            tar.extractall(DATASET_DIR)
        print("✅ 图片文件解压完成")

    # 显示解压结果
    print("\n📁 解压后的目录内容:")
    for item in sorted(DATASET_DIR.iterdir()):
        if item.is_dir():
            jpg_count = len(list(item.glob("*.jpg")))
            print(f"   📂 {item.name}/ ({jpg_count} 张 .jpg)")
            extracted_dirs.append(item)
        elif item.suffix == '.token' or 'results' in item.name:
            print(f"   📄 {item.name} ({item.stat().st_size / 1024 / 1024:.1f} MB)")

    return extracted_dirs


def parse_token_file(token_path):
    """
    解析.token文件
    格式: <image_name>#<caption_id>\t<caption>
    返回: {image_name: [{id, caption}, ...]}
    """
    print("\n" + "=" * 60)
    print(f"步骤2: 解析标注文件")
    print("=" * 60)

    image_captions = defaultdict(list)

    with open(token_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # 匹配格式: filename.jpg#0\tcaption text
            # 使用制表符分割，更安全
            if '\t' in line:
                parts = line.split('\t', 1)
                img_part = parts[0]
                caption = parts[1].strip() if len(parts) > 1 else ""

                # 分离图片名和序号
                if '#' in img_part:
                    img_name, cap_id = img_part.rsplit('#', 1)
                    img_name = img_name.strip()
                    cap_id = cap_id.strip()

                    image_captions[img_name].append({
                        'id': cap_id,
                        'caption': caption,
                        'line_num': line_num
                    })

    total_captions = sum(len(caps) for caps in image_captions.values())
    print(f"✅ 解析完成:")
    print(f"   • 图片总数: {len(image_captions)} 张")
    print(f"   • 标注总数: {total_captions} 条")
    print(f"   • 每张图片平均标注: {total_captions / len(image_captions):.1f} 条")

    # 显示示例
    sample_img = list(image_captions.keys())[0]
    print(f"\n📝 示例 ({sample_img}):")
    for cap in image_captions[sample_img][:2]:
        print(f"   #{cap['id']}: {cap['caption'][:60]}...")

    return dict(image_captions)


def split_dataset(image_captions):
    """划分数据集"""
    print("\n" + "=" * 60)
    print("步骤3: 划分数据集")
    print("=" * 60)

    all_images = list(image_captions.keys())
    total = len(all_images)

    # 随机打乱
    random.shuffle(all_images)

    # 按比例划分
    train_imgs = all_images[:TRAIN_SIZE]
    val_imgs = all_images[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    test_imgs = all_images[TRAIN_SIZE + VAL_SIZE:TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

    # 如果总数不足，按比例调整
    if total < TRAIN_SIZE + VAL_SIZE + TEST_SIZE:
        print(f"⚠️  图片总数({total})少于标准划分，按比例分配")
        train_end = int(total * 0.933)  # 28000/30000
        val_end = int(total * 0.967)  # +1000/30000
        train_imgs = all_images[:train_end]
        val_imgs = all_images[train_end:val_end]
        test_imgs = all_images[val_end:]

    splits = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs
    }

    for name, imgs in splits.items():
        print(f"   • {name}: {len(imgs)} 张图片")

    return splits


def create_split_structure(splits, image_captions, images_source_dir):
    """创建划分后的目录结构"""
    print("\n" + "=" * 60)
    print("步骤4: 创建目录结构并复制文件")
    print("=" * 60)

    # 清理旧输出目录
    if OUTPUT_DIR.exists():
        print(f"🗑️  清理旧目录: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 【关键修复】预扫描所有图片，建立 小写文件名 -> 实际路径 的映射
    images_source = Path(images_source_dir)
    print(f"\n📂 扫描图片目录: {images_source}")

    # 建立大小写不敏感的查找表
    file_lookup = {}
    for img_path in images_source.glob("*"):
        if img_path.is_file():
            # 使用小写作为键，保留原始路径作为值
            file_lookup[img_path.name.lower()] = img_path

    print(f"   找到 {len(file_lookup)} 个文件")

    # 检查几个示例
    sample_keys = list(file_lookup.keys())[:3]
    print(f"   示例: {sample_keys}")

    # 统计信息
    stats = {}

    for split_name, img_list in splits.items():
        print(f"\n📂 处理 {split_name} 集({len(img_list)} 张)...")

        # 创建目录结构: data/train/train_img/, data/train/train.token
        split_dir = OUTPUT_DIR / split_name
        img_dir = split_dir / f"{split_name}_img"
        img_dir.mkdir(parents=True, exist_ok=True)

        # 写入.token文件
        token_path = split_dir / f"{split_name}.token"
        caption_count = 0

        with open(token_path, 'w', encoding='utf-8') as f:
            for img_name in img_list:
                captions = image_captions.get(img_name, [])

                if split_name == 'test':
                    # 测试集：如果没有标注，写入空白占位符
                    if captions:
                        for cap in captions:
                            line = f"{img_name}#{cap['id']}\t{cap['caption']}\n"
                            f.write(line)
                            caption_count += 1
                    else:
                        f.write(f"{img_name}#0\t\n")  # 空白标注
                        caption_count += 1
                else:
                    # 训练/验证集：写入所有标注
                    for cap in captions:
                        line = f"{img_name}#{cap['id']}\t{cap['caption']}\n"
                        f.write(line)
                        caption_count += 1

        # 复制图片
        copied = 0
        missing = []

        for img_name in img_list:
            lookup_key = img_name.lower()
            if lookup_key in file_lookup:
                src_path = file_lookup[lookup_key]
                dst_path = img_dir / img_name  # 保持原始文件名

                try:
                    shutil.copy2(str(src_path), str(dst_path))
                    copied += 1
                except Exception as e:
                    print(f"   ❌ 复制失败 {img_name}: {e}")
            else:
                missing.append(img_name)

            # src = images_source_dir / img_name
            # dst = img_dir / img_name
            #
            # if src.exists():
            #     shutil.copy2(src, dst)
            #     copied += 1
            # else:
            #     missing.append(img_name)

        # 记录统计
        stats[split_name] = {
            'images': copied,
            'missing': len(missing),
            'captions': caption_count
        }

        print(f"   ✅ {split_name}.token: {caption_count} 行")
        print(f"   ✅ 图片复制: {copied}/{len(img_list)} 张")

        if missing and len(missing) <= 3:
            print(f"   ⚠️  缺失: {missing}")
        elif missing:
            print(f"   ⚠️  缺失: {len(missing)} 张")

    return stats


def verify_final_structure():
    """验证最终结构"""
    print("\n" + "=" * 60)
    print("步骤5: 验证最终结构")
    print("=" * 60)

    for split in ['train', 'val', 'test']:
        split_dir = OUTPUT_DIR / split
        img_dir = split_dir / f"{split}_img"
        token_file = split_dir / f"{split}.token"

        if not split_dir.exists():
            print(f"❌ {split}/ 目录不存在")
            continue

        # 统计
        img_count = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        token_lines = sum(1 for _ in open(token_file, 'r', encoding='utf-8')) if token_file.exists() else 0

        print(f"\n📁 {split}/")
        print(f"   图片: {img_count} 张 | 标注: {token_lines} 行")

        # 显示token文件样本
        if token_file.exists() and token_lines > 0:
            with open(token_file, 'r', encoding='utf-8') as f:
                sample_lines = f.readlines()[:2]
                print("   内容示例:")
                for line in sample_lines:
                    preview = line.strip()[:70]
                    print(f"      {preview}..." if len(line.strip()) > 70 else f"      {line.strip()}")

    # 最终结构树
    print("\n📊 最终目录结构:")
    print(f"""
image-captioning-project/
└── data/
    ├── train/
    │   ├── train_img/          ({stats.get('train', {}).get('images', 0)} 张图片)
    │   └── train.token         ({stats.get('train', {}).get('captions', 0)} 条标注)
    ├── val/
    │   ├── val_img/            ({stats.get('val', {}).get('images', 0)} 张图片)
    │   └── val.token           ({stats.get('val', {}).get('captions', 0)} 条标注)
    └── test/
        ├── test_img/           ({stats.get('test', {}).get('images', 0)} 张图片)
        └── test.token          ({stats.get('test', {}).get('captions', 0)} 条标注)
    """)


def find_token_file():
    """在解压后的目录中查找token文件"""
    candidates = [
        DATASET_DIR / "results_20130124.token",
        DATASET_DIR / "Flickr30k.token",
    ]
    # 搜索所有.token文件
    candidates.extend(DATASET_DIR.glob("*.token"))
    candidates.extend(DATASET_DIR.rglob("*results*.token"))

    for path in candidates:
        if path.exists():
            return path

    return None


def find_images_dir(extracted_dirs):
    """查找图片目录"""
    # 优先找包含flickr30k-images的目录
    for d in extracted_dirs:
        if 'image' in d.name.lower():
            jpg_count = len(list(d.glob("*.jpg")))
            if jpg_count > 1000:  # 确认是图片目录
                return d

    # 否则在DATASET_DIR根目录查找
    jpg_files = list(DATASET_DIR.glob("*.jpg"))
    if len(jpg_files) > 1000:
        return DATASET_DIR

    # 尝试其他常见目录名
    for dirname in ["Flickr30k-images", "images", "flickr30k_images", "Flickr30k"]:
        d = DATASET_DIR / dirname
        if d.exists() and d.is_dir():
            return d

    return DATASET_DIR


def main():
    print("🚀 Flickr30k 数据集处理工具")
    print("目标: 解包 → 解析 → 划分 → 重组目录结构")

    # 检查原始数据目录
    if not DATASET_DIR.exists():
        print(f"\n❌ 错误: 数据集目录不存在: {DATASET_DIR}")
        print("请确保目录结构如下:")
        print(f"""
project-root/
└── Flickr30k/
    ├── Flickr30k.tar.gz        (标注文件压缩包)
    ├── Flickr30k-images.tar     (图片压缩包)
    └── flickr30k_test.py       (测试脚本)
""")
        return 1

    # 1. 解压
    extracted_dirs = extract_tar_files()

    # 2. 查找token文件
    token_file = find_token_file()
    if not token_file:
        print("\n❌ 错误: 未找到.token标注文件")
        print("解压后的文件列表:")
        for f in DATASET_DIR.rglob("*"):
            if f.is_file():
                print(f"   {f.relative_to(DATASET_DIR)}")
        return 1

    print(f"\n📄 使用标注文件: {token_file.name}")

    # 3. 解析标注
    image_captions = parse_token_file(token_file)

    if not image_captions:
        print("❌ 错误: 未能解析到任何标注数据")
        return 1

    # 4. 查找图片目录
    images_dir = find_images_dir(extracted_dirs)
    print(f"\n📂 图片源目录: {images_dir}")

    # 检查图片数量
    jpg_count = len(list(images_dir.glob("*.jpg")))
    print(f"   找到 {jpg_count} 张 .jpg 图片")

    if jpg_count == 0:
        print("❌ 错误: 未找到任何图片文件")
        return 1

    # 5. 划分数据集
    splits = split_dataset(image_captions)

    # 6. 创建结构
    global stats
    stats = create_split_structure(splits, image_captions, images_dir)

    # 7. 验证
    verify_final_structure()

    print("\n" + "=" * 60)
    print("✅ 全部完成！数据集已准备就绪")
    print(f"📍 位置: {OUTPUT_DIR.absolute()}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())