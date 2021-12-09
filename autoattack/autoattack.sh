
## For testing DeiT-Small
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -u test_autoattack.py -a 'deit_small_patch16_224_adv' --test-batch 512  --data /path/to/ImageNet  --ckpt './ckpt/deit_small.pth' 


## For testing ResNet-50-GELU
# CUDA_VISIBLE_DEVICES=0,1,2,3 python  -u test_autoattack.py -a 'resnet50' --test-batch 256  --data /path/to/ImageNet  --ckpt './ckpt/res50_gelu.pth' --sing singgbn --forward-time 8 --pretrain-bs 4096 --activation gelu

