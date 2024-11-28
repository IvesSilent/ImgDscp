# ImgDscp - �������ѧϰ��ͼ��������Ŀ

����Ŀּ��ͨ�����ѧϰ�����Զ����ɸ���ͼ��������ı����Ҳ�����Flickr30k���ݼ�������CNN��Ϊ��������Transformer��Ϊ�������ļܹ������ͼ����������

## ��Ŀ�ṹ

```
image-captioning-project/
��
������ data/                       # ���ݼ�Ŀ¼
��   ������ train/                  # ѵ����Ŀ¼
��   ��   ������ train_img/          # ѵ����ͼƬ
��   ��   ������ train.token        # ѵ������ע
��   ������ val/                   # ��֤��Ŀ¼
��   ��   ������ val_img/           # ��֤��ͼƬ
��   ��   ������ val.token          # ��֤����ע
��   ������ test/                  # ���Լ�Ŀ¼
��       ������ test_img/          # ���Լ�ͼƬ
��       ������ test.token         # ���Լ����
��
������ /Result_Fig/                  # ѵ�����̿��ӻ���ͼƬ�ĵ�
������ /models/                      # ѵ����ģ�ͱ���Ŀ¼
������ /models/                      # ѵ����־����Ŀ¼
������ /singleprd/                   # ���predict.py���ɵ������ı�
��
������ models.py                  # ģ�ͼܹ������ļ�
������ ImgDataset.py              # ���ݼ������ඨ���ļ�
������ train.py                  # ѵ������֤���̽ű�
������ test.py                   # ���Թ��̽ű�
������ predict.py                 # ����Ԥ����̽ű�
������ vocab.pth                     # �ʱ��ļ�
������ requirements.txt           # ��Ŀ���������ļ�
������ README.md                 # ��Ŀ˵���ĵ�
```

## ��������

�ڿ�ʼ֮ǰ����ȷ�����Ļ����а�װ������������

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

������ͨ������������������װ��Щ������

```bash
pip install -r requirements.txt
```

### ����PyTorch

`torch`��`torchvision`
�İ�װ������Ҫ����Ĳ��裬��Ϊ����ͨ����Ҫ�����CUDA�汾���ݡ�������ϵͳ��װ��CUDA������Ҫȷ����װ��`torch`
��`torchvision`�汾��CUDA�汾��ƥ�䡣

�����ͨ������PyTorch�ٷ���վ�İ�װָ������ȡ��ȷ�İ�װ���[PyTorch Get Started](https://pytorch.org/get-started/previous-versions/)��

## ���ݴ���

���ݼ���Flickr30k��ѵ��������֤���Ͳ��Լ��ֱ�λ�ڸ�Ŀ¼�µ�`/train`��`/val`��`/test`�ļ��С�

ѵ����ͼƬ�ļ���`/train/train_img`�ڣ�`/train/train.token`��ѵ�����ݵı�ע����ʽ���£�

```plaintext
<TestImage_name><#><���></t><caption>
```

��עʾ�����£�

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

һ��ͼƬ��Ӧ�����Ȼ�����ı������� ��֤����ʽ��ͬ��

���ڹ�����Ŀʱ�������ݼ�������ʦ�����ļ���[���ϴ�ѧ����](https://pan.seu.edu.cn:443/link/215E44A851DA77CA52FF410F26F15498)
��������ҪУ԰�������ֿ��ڲ��������ļ�����Ҫ�������ػ����ݼ����д�������ָ��λ�á�

��ʹ��Torchvision����ͼ�����ݼ��غ�Ԥ�������ڼ���ͼ������ʱ������������ǿ������

## ģ�ͼܹ�

��ѡ����Inception��ΪCNN���������Լ�������ע�������Ƶ�Transformer��Ϊ��������ģ�ͼܹ������� `models.py` �ļ��С�

| ����/�Ż��ֶ� | ����                                          | ����ϸ��                     |
|---------|---------------------------------------------|--------------------------|
| ���      | PyTorch                                     | �汾1.9.0                  |
| ������     | CNN��ʹ����������Ԥѵ���ܹ�Inception V3                 | Ȩ�أ�ImageNet1K v1         |
| ������     | Transformer����������������㣬ÿ�������ע�������ơ�ǰ��������Ͳ��һ�� | ������6��ͷ����8�����ز�ά�ȣ�512      |
| ���������   | ��ȡ����������ϲ���һ��ע������ϲ�Ϊ��������                     | ���������11��ʹ��ReLU�����Dropout |
| Ӳ������    | ���豸����GPU��ʹ��GPU����ѵ������֤��Ԥ��                    |                          |
| ����     | ����ɵ���Dropout�����Է�ֹ�����                        | Dropout�ʣ�0.3             |

## ѵ������֤

ѵ������֤������ `train.py` �ű����ơ���ʹ��Xavier���ȳ�ʼ��Ȩ�أ����ý�������ʧ��Adam�Ż��������Ż�����ʹ�������˻�ѧϰ�ʵ��Ȼ��ơ�ѵ�������л������BLEU��ROUGE��CIDEr��ָ�ꡣ

| ����/�Ż��ֶ� | ����                                    | ����ϸ��                    |
|---------|---------------------------------------|-------------------------|
| Ȩ�س�ʼ��   | ʹ��Xavier���ȳ�ʼ��Ȩ��                       |                         |
| ��ʧ����    | ʹ�ý�������ʧ���Ż�ģ�ͣ������ǩƽ������                 | ƽ�������ţ�0.05              |
| �Ż���     | ʹ��Adam�Ż���                             | ѧϰ�ʣ�0.0005��Ȩ��˥����1e-5    |
| ѧϰ�ʵ���   | �����˻��ѧϰ�ʵ��Ȼ���                          | T_max��20����_min��0.000001 |
| ��Ͼ��ȼ���  | ʹ�û�Ͼ��ȼ���ķ�ʽѵ��                         |                         |
| ѧϰ��ʽ    | ʹ��Teacher Forcing��ѧϰ��ʽ                |                         |
| ������ʾ    | �Խ�������ʾѵ���Ľ���                           |                         |
| ģ�ʹ洢    | ѵ����ɺ�洢ģ�ͷ������ѧϰ�ʶ��ѵ����ģ�ʹ洢��/modelsĿ¼   |                         |
| ��ͣ����    | ������ͣ�������������ϣ�������֤�������ܲ�������ʱֹͣѵ��        | ��ͣ��ֵ��5��epoch������         |
| ��������    | ѵ��ͬʱ������֤������BLEU��ROUGE��CIDEr��ָ�겢���ӻ�    |                         |
| ���ӻ��洢   | �����ӻ���ͼƬ����/Result_Fig����ѵ�����ģ�ʹ���/models |                         |
| ������ʾ    | �Խ�������ʾ��֤�Ľ���                           |                         |

���ն�����train.py�����ö�Ӧ�Ĳ�������ֱ�Ӳ���Ĭ�ϲ�����

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

## ����

���Թ����� `test.py` �ű����ƣ��ýű���ȡ `test/test_img` Ŀ¼�ڵ�ͼƬ�ļ������� `test` Ŀ¼�ڴ����ı��ļ� `test.token`
����ÿһ��ͼƬ������������ı���

## ����Ԥ��

����Ԥ������� `predict.py` �ű����ƣ�����ȡָ��·����ͼƬ������ `/singleprd` Ŀ¼�ڴ����ı��ļ������ɸ�ͼƬ�����������

## ʹ��˵��

- ѵ��ģ�ͣ����� `python train.py`��
- ����ģ�ͣ����� `python test.py`��
- ����Ԥ�⣺���� `python predict.py --image_path <your_image_path>`��

## �����뷴��

��ӭ�Ա���Ŀ������������ͽ��顣�����κ����⣬��ͨ��GitHub Issues���з�����

## ���֤

����Ŀ���� [MIT License](https://opensource.org/licenses/MIT)��

