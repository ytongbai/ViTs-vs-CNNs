
# ImageNet-A
CUDA_VISIBLE_DEVICES=0,1,2,3  python imagenet_robust.py -a resnet50 --test-batch 256 --data /path/to/imagenet-a --evaluate

# ImageNet-C
CUDA_VISIBLE_DEVICES=0,1,2,3  python imagenet_robust.py -a resnet50 --test-batch 256 --data /path/to/ImageNet-C --evaluate_imagenet_c

# Stylized-ImageNet
CUDA_VISIBLE_DEVICES=0,1,2,3  python imagenet_robust.py -a resnet50 --test-batch 256 --data /path/to/Stylized-ImageNet --evaluate
