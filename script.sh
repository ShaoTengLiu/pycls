CUDA_VISIBLE_DEVICES=1 \
    python tools/test_net_ftta.py \
    --cfg configs/cifar100/R-26_w4_1gpu_adaptbn_entropy.yaml \
    --corruptions gaussian_noise \
    --levels 5