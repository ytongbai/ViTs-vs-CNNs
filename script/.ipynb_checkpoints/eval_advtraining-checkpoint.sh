# Adversarial Training Evaluation

# Evaluate Adversarial Trained DeiT-Small
python -m torch.distributed.launch --nproc_per_node=8 --master_port=5672  --use_env main_adv_deit.py --model deit_small_patch16_224_adv --batch-size 128 --data-path /path/to/imagenet --attack-iter 1 --attack-epsilon 4 --attack-step-size 4 --epoch 100  --reprob 0  --no-repeated-aug --sing singln  --drop 0 --drop-path 0 --start_epoch 0 --warmup-epochs 10 --cutmix 0 --resume /path/to/advdeit_small.pth --eval 

# Evaluate Adversarial Trained ResNet-GELU
python -m torch.distributed.launch --nproc_per_node=4 --master_port=3088  --use_env main_adv_res.py --model resnet50 --batch-size 128 --data-path /path/to/imagenet  --attack-iter 1 --attack-epsilon 4 --attack-step-size 4 --epoch 100  --reprob 0  --no-repeated-aug --sing singgbn  --drop 0 --drop-path 0 --start_epoch 0 --warmup-epochs 5 --adjust_lr 256 --lr 0.1 --warmup-lr 0.1 --sched step --mixup 0 --cutmix 0 --opt sgd --weight-decay 1e-4 --activation gelu --prob_start_from_clean 0.2 --resume /path/to/advres50_gelu.pth --eval

# Evaluate Adversarial Trained ResNet-ReLU
python -m torch.distributed.launch --nproc_per_node=4 --master_port=3088  --use_env main_adv_res.py --model resnet50 --batch-size 256 --data-path /path/to/imagenet  --resume /path/to/advres50_relu.pth --attack-iter 1 --attack-epsilon 4 --attack-step-size 4 --epoch 100  --reprob 0  --no-repeated-aug --sing singgbn  --drop 0 --drop-path 0 --start_epoch 0 --warmup-epochs 5 --adjust_lr 256 --lr 0.1 --warmup-lr 0.1 --sched step --mixup 0 --cutmix 0 --opt sgd --weight-decay 1e-4 --activation relu --prob_start_from_clean 0.2 --forward-time 4 --eval