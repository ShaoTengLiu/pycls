CUDA_VISIBLE_DEVICES=0 \
    python tools/train_net.py \
    --cfg configs/cifar100/S-19_1gpu_cifar100_bn_pairwise.yaml

    # python tools/test_net_ftta.py \
    # --cfg configs/cifar100/R-26_w1_1gpu_adaptbn_entropy.yaml \
    # --corruptions gaussian_noise \
    # --levels 5