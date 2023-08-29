# Score Distillation
Pytorch implementation of "Score distillation".

## Dataset
- CIFAR 10
- CIFAR 100
- Fashion-MNIST

- 80M Tiny Images for OE 

## Teacher Model
  Name | Method | Architecture | Paper Link | Code Link
  :---- | ----- | ----- | :----: | :----:
  CSI | Contrasting shifted instances | ResNet18 | [paper](https://arxiv.org/abs/2007.08176.pdf) | [code](https://github.com/alinlab/CSI)
  PANDA   | Pretrained Anomaly Detection Adaption | ResNet152 | [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Reiss_PANDA_Adapting_Pretrained_Features_for_Anomaly_Detection_and_Segmentation_CVPR_2021_paper.pdf) | [code](https://github.com/talreiss/PANDA)
  MSCL    | Mean-Shifted Contrastive Loss | ResNet152| [paper](https://arxiv.org/pdf/2106.03844.pdf) | [code](https://github.com/talreiss/Mean-Shifted-Anomaly-Detection)

## How To Use
- Train
```bash
python distillation.py --dataset [cifar10 or cifar100 or fashion] --teacher [Teacher Method] --margin 0.5 --teacher CSI --mode train --label [Target Class]
```
- Evaluation
```bash
python distillation.py --dataset [cifar10 or cifar100 or fashion] --teacher [Teacher Method] --margin 0.5 --teacher CSI --mode test --label [Target Class]
```
