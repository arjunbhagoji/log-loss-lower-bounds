#!/bin/bash

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=1.0 --attack_iter=40 --eps_step=0.03 --dataset_in=MNIST --model=cnn_3l --train_epochs=50 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=2.0 --attack_iter=40 --eps_step=0.06 --dataset_in=MNIST --model=cnn_3l --train_epochs=50 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=40 --eps_step=0.09 --dataset_in=MNIST --model=cnn_3l --train_epochs=50 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=40 --eps_step=0.12 --dataset_in=MNIST --model=cnn_3l --train_epochs=50 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=5.0 --attack_iter=40 --eps_step=0.15 --dataset_in=MNIST --model=cnn_3l --train_epochs=50 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=40 --eps_step=0.09 --dataset_in=MNIST --model=cnn_3l_large --train_epochs=50 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=40 --eps_step=0.12 --dataset_in=MNIST --model=cnn_3l_large --train_epochs=50 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=5.0 --attack_iter=40 --eps_step=0.15 --dataset_in=MNIST --model=cnn_3l_large --train_epochs=50 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=50 --eps_step=0.15 --dataset_in=MNIST --model=cnn_3l_large --train_epochs=100 --n_classes=2 --num_samples=2000

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=50 --eps_step=0.2 --dataset_in=MNIST --model=cnn_3l_large --train_epochs=100 --n_classes=2 --num_samples=2000

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=5.0 --attack_iter=50 --eps_step=0.25 --dataset_in=MNIST --model=cnn_3l_large --train_epochs=100 --n_classes=2 --num_samples=2000

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=50 --eps_step=0.15 --dataset_in=MNIST --model=cnn_3l_large --train_epochs=100 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=3.2 --attack_iter=50 --eps_step=0.16 --dataset_in=MNIST --model=cnn_3l_large --train_epochs=100 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=3.4 --attack_iter=50 --eps_step=0.17 --dataset_in=MNIST --model=cnn_3l_large --train_epochs=100 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=3.6 --attack_iter=50 --eps_step=0.18 --dataset_in=MNIST --model=cnn_3l_large --train_epochs=100 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=3.8 --attack_iter=50 --eps_step=0.19 --dataset_in=MNIST --model=cnn_3l_large --train_epochs=100 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=50 --eps_step=0.2 --dataset_in=MNIST --model=cnn_3l_large --train_epochs=100 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=5.0 --attack_iter=50 --eps_step=0.25 --dataset_in=MNIST --model=cnn_3l --train_epochs=100 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=50 --eps_step=0.15 --dataset_in=MNIST --model=cnn_3l_conv_16x --train_epochs=100 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=50 --eps_step=0.2 --dataset_in=MNIST --model=cnn_3l_conv_16x --train_epochs=100 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=50 --eps_step=0.2 --dataset_in=MNIST --model=cnn_3l --conv_expand=2 --fc_expand=1 --train_epochs=100 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=50 --eps_step=0.2 --dataset_in=MNIST --model=cnn_3l --conv_expand=4 --fc_expand=1 --train_epochs=100 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=50 --eps_step=0.2 --dataset_in=MNIST --model=cnn_3l --conv_expand=8 --fc_expand=1 --train_epochs=100 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=50 --eps_step=0.2 --dataset_in=MNIST --model=cnn_3l --conv_expand=16 --fc_expand=1 --train_epochs=100 --n_classes=2 --num_samples=2000 --rand_init

# python train_robust.py --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=50 --eps_step=0.2 --dataset_in=MNIST --model=cnn_3l --conv_expand=2 --fc_expand=2 --train_epochs=100 --n_classes=2 --num_samples=2000 --rand_init

python train_robust.py --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=50 --eps_step=0.2 --dataset_in=MNIST --model=cnn_3l --conv_expand=8 --fc_expand=8 --train_epochs=100 --n_classes=2 --num_samples=2000 --rand_init