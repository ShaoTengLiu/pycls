CUDA_VISIBLE_DEVICES=0 \
python tools/train_net.py \
    --cfg configs/cifar10/R-26_w8_1gpu_cifar10_bn.yaml

# python tools/test_net_ftta.py \
# --cfg configs/cifar100/R-26_w4_1gpu_adaptbn_entropy.yaml \
# --corruptions gaussian_noise \
# --levels 5