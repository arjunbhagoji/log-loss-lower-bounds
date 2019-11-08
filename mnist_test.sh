#!/bin/bash

# python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=40 --eps_step=0.09 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=10 --new_eps_step=0.36 --new_rand_init --rand_init

# python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=40 --eps_step=0.09 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=10 --new_eps_step=0.75 --new_rand_init --rand_init

# python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=40 --eps_step=0.09 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=40 --new_eps_step=0.09 --new_rand_init --rand_init

# python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=40 --eps_step=0.09 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=40 --new_eps_step=0.1875 --new_rand_init --rand_init

# python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=40 --eps_step=0.09 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=100 --new_eps_step=0.036 --new_rand_init --rand_init

# python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=40 --eps_step=0.09 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=100 --new_eps_step=0.075 --new_rand_init --rand_init

# python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=40 --eps_step=0.12 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=10 --new_eps_step=0.48 --new_rand_init --rand_init

# python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=40 --eps_step=0.12 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=10 --new_eps_step=1.0 --new_rand_init --rand_init

# python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=40 --eps_step=0.12 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=40 --new_eps_step=0.12 --new_rand_init --rand_init

# python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=40 --eps_step=0.12 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=40 --new_eps_step=0.25 --new_rand_init --rand_init

# python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=40 --eps_step=0.12 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=100 --new_eps_step=0.048 --new_rand_init --rand_init

# python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=40 --eps_step=0.12 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=100 --new_eps_step=0.1 --new_rand_init --rand_init

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=50 --eps_step=0.15 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=10 --new_eps_step=0.75 --new_rand_init --rand_init --is_dropping

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=50 --eps_step=0.15 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=10 --new_eps_step=1.5 --new_rand_init --rand_init --is_dropping

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=50 --eps_step=0.15 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=50 --new_eps_step=0.15 --new_rand_init --rand_init --is_dropping

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=50 --eps_step=0.15 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=50 --new_eps_step=0.3 --new_rand_init --rand_init --is_dropping

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=50 --eps_step=0.15 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=100 --new_eps_step=0.075 --new_rand_init --rand_init --is_dropping

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=50 --eps_step=0.15 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=100 --new_eps_step=0.15 --new_rand_init --rand_init --is_dropping

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=50 --eps_step=0.2 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=10 --new_eps_step=1.0 --new_rand_init --rand_init --is_dropping

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=50 --eps_step=0.2 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=10 --new_eps_step=2.0 --new_rand_init --rand_init --is_dropping

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=50 --eps_step=0.2 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=50 --new_eps_step=0.2 --new_rand_init --rand_init --is_dropping

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=50 --eps_step=0.2 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=50 --new_eps_step=0.4 --new_rand_init --rand_init --is_dropping

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=50 --eps_step=0.2 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=100 --new_eps_step=0.1 --new_rand_init --rand_init --is_dropping

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=50 --eps_step=0.2 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=100 --new_eps_step=0.2 --new_rand_init --rand_init --is_dropping