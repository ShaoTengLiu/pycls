CUDA_VISIBLE_DEVICES=3 \
    python tools/train_net.py \
    --cfg configs/cifar100/R-26_w4_1gpu_cifar100_bn_oracle.yaml

    # python tools/test_net_ftta.py \
    # --cfg configs/cifar100/R-26_w4_1gpu_adaptbn_entropy.yaml \
    # --corruptions gaussian_noise \
    # --levels 5