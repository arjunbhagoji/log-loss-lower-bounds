#!/bin/bash

# python test_robust.py --dataset_in=MNIST --model=cnn_3l --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=1.0 --new_attack=PGD_l2 --new_epsilon=1.0 --new_attack_iter=10 --new_eps_step=0.12

# python test_robust.py --dataset_in=MNIST --model=cnn_3l --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=2.0 --new_attack=PGD_l2 --new_epsilon=2.0 --new_attack_iter=10 --new_eps_step=0.24

# python test_robust.py --dataset_in=MNIST --model=cnn_3l --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=10 --new_eps_step=0.36

# python test_robust.py --dataset_in=MNIST --model=cnn_3l --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=10 --new_eps_step=0.48

# python test_robust.py --dataset_in=MNIST --model=cnn_3l --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=5.0 --new_attack=PGD_l2 --new_epsilon=5.0 --new_attack_iter=10 --new_eps_step=0.6

# python test_robust.py --dataset_in=MNIST --model=cnn_3l --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=1.0 --new_attack=PGD_l2 --new_epsilon=1.0 --new_attack_iter=40 --new_eps_step=0.03

# python test_robust.py --dataset_in=MNIST --model=cnn_3l --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=2.0 --new_attack=PGD_l2 --new_epsilon=2.0 --new_attack_iter=40 --new_eps_step=0.06

# python test_robust.py --dataset_in=MNIST --model=cnn_3l --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=40 --new_eps_step=0.09

# python test_robust.py --dataset_in=MNIST --model=cnn_3l --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=40 --new_eps_step=0.12

# python test_robust.py --dataset_in=MNIST --model=cnn_3l --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=5.0 --new_attack=PGD_l2 --new_epsilon=5.0 --new_attack_iter=40 --new_eps_step=0.15

# python test_robust.py --dataset_in=MNIST --model=cnn_3l --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=1.0 --new_attack=PGD_l2 --new_epsilon=1.0 --new_attack_iter=100 --new_eps_step=0.012

# python test_robust.py --dataset_in=MNIST --model=cnn_3l --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=2.0 --new_attack=PGD_l2 --new_epsilon=2.0 --new_attack_iter=100 --new_eps_step=0.024

# python test_robust.py --dataset_in=MNIST --model=cnn_3l --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=100 --new_eps_step=0.036

# python test_robust.py --dataset_in=MNIST --model=cnn_3l --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=100 --new_eps_step=0.048

# python test_robust.py --dataset_in=MNIST --model=cnn_3l --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=5.0 --new_attack=PGD_l2 --new_epsilon=5.0 --new_attack_iter=100 --new_eps_step=0.06

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=10 --new_eps_step=0.36

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=10 --new_eps_step=0.48

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=5.0 --new_attack=PGD_l2 --new_epsilon=5.0 --new_attack_iter=10 --new_eps_step=0.6

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=40 --new_eps_step=0.09

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=40 --new_eps_step=0.12

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=5.0 --new_attack=PGD_l2 --new_epsilon=5.0 --new_attack_iter=40 --new_eps_step=0.15

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=3.0 --new_attack=PGD_l2 --new_epsilon=3.0 --new_attack_iter=100 --new_eps_step=0.036

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=4.0 --new_attack=PGD_l2 --new_epsilon=4.0 --new_attack_iter=100 --new_eps_step=0.048

python test_robust.py --dataset_in=MNIST --model=cnn_3l_large --n_classes=2 --num_samples=2000 --is_adv --attack=PGD_l2 --epsilon=5.0 --new_attack=PGD_l2 --new_epsilon=5.0 --new_attack_iter=100 --new_eps_step=0.06