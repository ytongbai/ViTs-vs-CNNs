# Are Transformers More Robust Than CNNs?

Pytorch implementation for NeurIPS 2021 Paper: [Are Transformers More Robust Than CNNs?](https://arxiv.org/pdf/2111.05464.pdf)

Our implementation is based on [DeiT](https://github.com/facebookresearch/deit).

<div align="center">
  <img src="resources/teaser.png"/>
</div>


## Introduction
Transformer emerges as a powerful tool for visual recognition. In addition to demonstrating competitive performance on a broad range of visual benchmarks, recent works also argue that Transformers are much more robust than Convolutions Neural Networks (CNNs). Nonetheless, surprisingly, we find these conclusions are drawn from unfair experimental settings, where Transformers and CNNs are compared at different scales and are applied with distinct training frameworks. In this paper, we aim to provide the first fair & in-depth comparisons between Transformers and CNNs, focusing on robustness evaluations.

With our unified training setup, we first challenge the previous belief that Transformers outshine CNNs when measuring adversarial robustness. More surprisingly, we find CNNs can easily be as robust as Transformers on defending against adversarial attacks, if they properly adopt Transformers' training recipes. While regarding generalization on out-of-distribution samples, we show pre-training on (external) large-scale datasets is not a fundamental request for enabling Transformers to achieve better performance than CNNs. Moreover, our ablations suggest such stronger generalization is largely benefited by the Transformer's self-attention-like architectures per se, rather than by other training setups. We hope this work can help the community better understand and benchmark the robustness of Transformers and CNNs. 

## Pretrained models

We provide both pretrained vanilla models and adversarially trained models.

### Vanilla Training

#### Main Results

|             |                                          Pretrained Model                                           | ImageNet | ImageNet-A | ImageNet-C | Stylized-ImageNet |
| ----------- | :-------------------------------------------------------------------------------------------------: | :------: | :--------: | :--------: | :---------------: |
| Res50-Ori   | [download link](https://drive.google.com/file/d/1iNEvIPKYgi1ivfL4v4dKg8j1QN5bgVpn/view?usp=sharing) |   76.9   |    3.2     |    57.9    |        8.3        |
| Res50-Align | [download link](https://drive.google.com/file/d/1SE8u1jctOM5dsbhmrcYW9Mwo3yWfNAaw/view?usp=sharing) |   76.3   |    4.5     |    55.6    |        8.2        |
| Res50-Best  | [download link](https://drive.google.com/file/d/12g6Gwn-KXwLBrscjvWgjrK_kXM0twgiS/view?usp=sharing) |   75.7   |    6.3     |    52.3    |       10.8        |
| DeiT-Small  | [download link](https://drive.google.com/file/d/1VorHupVJxnBOMS79gYbIhcOaL00h_KTr/view?usp=sharing) |   76.8   |    12.2    |    48.0    |       13.0        |

#### Model Size

ResNets:

- ResNets fully aligned (with DeiT's training recipe) model, denoted as `res*`:

|          | Model Size |                                          Pretrained Model                                           | ImageNet | ImageNet-A | ImageNet-C | Stylized-ImageNet |
| -------- | :--------: | :-------------------------------------------------------------------------------------------------: | :------: | :--------: | :--------: | :---------------: |
| Res18\*  |   11.69M   | [download link](https://drive.google.com/file/d/1Q5gj330KoCkNShr_y9mkvFZ5sUcHtfSn/view?usp=sharing) |  67.83   |    1.92    |   64.14    |       7.92        |
| Res50\*  |   25.56M   | [download link](https://drive.google.com/file/d/1SE8u1jctOM5dsbhmrcYW9Mwo3yWfNAaw/view?usp=sharing) |  76.28   |    4.53    |   55.62    |       8.17        |
| Res101\* |   44.55M   | [download link](https://drive.google.com/file/d/1dk640r6Y504Swhs2lUi-NiPoeNITe6ix/view?usp=sharing) |  77.97   |    8.84    |   49.19    |       11.60       |

- ResNets best model (for Out-of-Distribution (OOD) generalization), denoted as `res-best`:

|             | Model Size |                                          Pretrained Model                                           | ImageNet | ImageNet-A | ImageNet-C | Stylized-ImageNet |
| ----------- | :--------: | :-------------------------------------------------------------------------------------------------: | :------: | :--------: | :--------: | :---------------: |
| Res18-best  |   11.69M   | [download link](https://drive.google.com/file/d/16mtiO-04UaIb19BFFKptlThaE7Surac4/view?usp=sharing) |  66.81   |    2.03    |   62.65    |       9.45        |
| Res50-best  |   25.56M   | [download link](https://drive.google.com/file/d/12g6Gwn-KXwLBrscjvWgjrK_kXM0twgiS/view?usp=sharing) |  75.74   |    6.32    |   52.25    |       10.77       |
| Res101-best |   44.55M   | [download link](https://drive.google.com/file/d/13HbBPkFVijP8VClBQBCAzdKcGc6o7WZy/view?usp=sharing) |  77.83   |   11.49    |   47.35    |       13.28       |

DeiTs:

|            | Model Size |                                          Pretrained Model                                           | ImageNet | ImageNet-A | ImageNet-C | Stylized-ImageNet |
| ---------- | :--------: | :-------------------------------------------------------------------------------------------------: | :------: | :--------: | :--------: | :---------------: |
| DeiT-Mini  |   9.98M    | [download link](https://drive.google.com/file/d/1FzuZP_eH2Vrb0hohIH-NUnSTuI6MG-Eg/view?usp=sharing) |  72.89   |    8.19    |   54.68    |       9.88        |
| DeiT-Small |   22.05M   | [download link](https://drive.google.com/file/d/1VorHupVJxnBOMS79gYbIhcOaL00h_KTr/view?usp=sharing) |  76.82   |   12.21    |   47.99    |       12.98       |

#### Model Distillation

|         |  Architecture   |                                          Pretrained Model                                           | ImageNet | ImageNet-A | ImageNet-C | Stylized-ImageNet |
| ------- | :-------------: | :-------------------------------------------------------------------------------------------------: | :------: | :--------: | :--------: | :---------------: |
| Teacher |   DeiT-Small    | [download link](https://drive.google.com/file/d/1iNEvIPKYgi1ivfL4v4dKg8j1QN5bgVpn/view?usp=sharing) |   76.8 | 12.2 |  48.0 | 13.0        |
| Student | Res50\*-Distill | [download link](https://drive.google.com/file/d/1MK0TQXoEAFfrhEC1Edm7FrcnemW8Zp7Y/view?usp=sharing) |   76.7 | 5.2 | 54.2 | 9.8       |
| Teacher |     Res50\*     | [download link](https://drive.google.com/file/d/1SE8u1jctOM5dsbhmrcYW9Mwo3yWfNAaw/view?usp=sharing) |   76.3 | 4.5 | 55.6 | 8.2        |
| Student | DeiT-S-Distill  | [download link](https://drive.google.com/file/d/1IrOowURrFbdZGe_FK87_6UvZWjz2Y-9n/view?usp=sharing) |   76.2 | 10.9 | 49.3 | 11.9       |

### Adversarial Training

|            |                                          Pretrained Model                                           | Clean Acc | PGD-100 | Auto Attack |
| ---------- | :-------------------------------------------------------------------------------------------------: | :-------: | :-----: | :---------: |
| Res50-ReLU | [download link](https://drive.google.com/file/d/1q8VxQuMWGVpFeU0OZay989dmDjrPrf3d/view?usp=sharing) |   66.77   |  32.26  |    26.41    |
| Res50-GELU | [download link](https://drive.google.com/file/d/1IPExDTAAuxIhUSYrmlweTQaKNscTw24-/view?usp=sharing) |   67.38   |  40.27  |    35.51    |
| DeiT-Small | [download link](https://drive.google.com/file/d/1U5XmAUQkSlw5Q1ZhsVriBEOQEk-bPfFU/view?usp=sharing) |   66.50   |  40.32  |    35.50    |

## Vanilla Training

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the [standard layout](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) for the torchvision, and the training and validation data is expected to be in the `train` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

### Environment

Install dependencies:

```bash
pip3 install -r requirements.txt
```

### Training Scripts

To train a ResNet model on ImageNet run:

```bash
bash script/res.sh
```

To train a DeiT model on ImageNet run:

```bash
bash script/deit.sh
```

## Generalization to Out-of-Distribution Sample

### Data Preparation

Download and extract [ImageNet-A](https://github.com/hendrycks/natural-adv-examples), [ImageNet-C](https://github.com/hendrycks/robustness), [Stylized-ImageNet](https://github.com/rgeirhos/Stylized-ImageNet) val images:

```
/path/to/datasets/
  val/
    class1/
      img1.jpeg
    class/2
      img2.jpeg
```

### Evaluation Scripts

To evaluate pre-trained models, run:

```bash
bash script/generation_to_ood.sh
```
It is worth noting that for ImageNet-C evaluation, the error rate is calculated based on the Noise, Blur, Weather and Digital categories. 

## Adversarial Training

To perform adversarial training on ResNet run:

```bash
bash script/advres.sh
```

To do adversarial training on DeiT run:

```sh
bash scripts/advdeit.sh
```

## Robustness to Adversarial Example


### PGD Attack Evaluation

To evaluate the pre-trained models, run:

```bash
bash script/eval_advtraining.sh
```


### AutoAttack Evaluation

`./autoattack` contains the [AutoAttack](https://github.com/fra31/auto-attack) public package, with a little modification to best support ImageNet evaluation.

```sh
cd autoattack/
bash autoattack.sh
```

### Patch Attack Evaluation

Please refer to [PatchAttack](https://github.com/Chenglin-Yang/PatchAttack)

## Citation

If you use our code, models or wish to refer to our results, please use the following BibTex entry:

```bibtex
@inproceedings{bai2021transformers,
  title     = {Are Transformers More Robust Than CNNs?},
  author    = {Bai, Yutong and Mei, Jieru and Yuille, Alan and Xie, Cihang},
  booktitle = {Thirty-Fifth Conference on Neural Information Processing Systems},
  year      = {2021},
}
```
