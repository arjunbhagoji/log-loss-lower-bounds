#!/bin/bash

# python train_robust.py --is_adv=True --attack=PGD_l2 --epsilon=1.0 --attack_iter=40 --eps_step=0.03 --dataset_in=MNIST --model=cnn_3l --train_epochs=20 --n_classes=2 --num_samples=2000

# python train_robust.py --is_adv=True --attack=PGD_l2 --epsilon=2.0 --attack_iter=40 --eps_step=0.06 --dataset_in=MNIST --model=cnn_3l --train_epochs=30 --n_classes=2 --num_samples=2000

# python train_robust.py --is_adv=True --attack=PGD_l2 --epsilon=3.0 --attack_iter=40 --eps_step=0.09 --dataset_in=MNIST --model=cnn_3l --train_epochs=40 --n_classes=2 --num_samples=2000

# python train_robust.py --is_adv=True --attack=PGD_l2 --epsilon=4.0 --attack_iter=40 --eps_step=0.12 --dataset_in=MNIST --model=cnn_3l --train_epochs=50 --n_classes=2 --num_samples=2000

# python train_robust.py --is_adv=True --attack=PGD_l2 --epsilon=5.0 --attack_iter=40 --eps_step=0.15 --dataset_in=MNIST --model=cnn_3l --train_epochs=60 --n_classes=2 --num_samples=2000

python train_robust.py --is_adv --attack=PGD_l2 --epsilon=3.0 --attack_iter=50 --eps_step=0.2 --dataset_in=MNIST --model=cnn_3l_large --train_epochs=100 --n_classes=2 --num_samples=2000

python train_robust.py --is_adv --attack=PGD_l2 --epsilon=4.0 --attack_iter=50 --eps_step=0.3 --dataset_in=MNIST --model=cnn_3l_large --train_epochs=100 --n_classes=2 --num_samples=2000

python train_robust.py --is_adv --attack=PGD_l2 --epsilon=5.0 --attack_iter=50 --eps_step=0.4 --dataset_in=MNIST --model=cnn_3l_large --train_epochs=100 --n_classes=2 --num_samples=2000