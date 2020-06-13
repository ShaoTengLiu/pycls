CUDA_VISIBLE_DEVICES=0 \
    python tools/test_net_ftta.py \
    --cfg configs/cifar100/R-26_w4_1gpu_FG1_entropy.yaml \
    --corruptions contrast \
    --levels 5

    # python tools/train_net.py \
    # --cfg configs/cifar100/R-26_w4_1gpu_cifar100_bn_oracle.yaml