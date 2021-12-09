This is the code for evaluating the results in Table 2.

./autoattack.sh includes the command to run the evaluation for DeiT-Small and ResNet-50-GELU. Please replace the data path to your ImageNet data path.

./ckpt is used for storing the checkpoints of adversarially trained DeiT-Small and ResNet-50-GELU.

./logs contains the evalutation outputs of these two models.

./att contains the AutoAttack public package, with a little modification to best support ImageNet evaluation. More information can be found here: https://github.com/fra31/auto-attack





1. Please unzip the checkpoints in ./ckpt first:

2. Install pytorch and torchvision timm;

4. Then install AutoAttack package with: pip install git+https://github.com/fra31/auto-attack

5. Then execute: bash autoattack.sh

