This code accompanies the paper _Lower Bounds on Cross-Entropy Loss in the Presence of Test-time Adversaries_. 


## Lower bounds on cross-entropy loss

To get the optimal cross-entropy loss for the MNIST dataset at an L2 budget of 2, run the following command. The appropriate parameters can be replaced for other experimental settings.

```
python optimal_log_loss.py --num_samples=5000 --n_classes=2 --eps=2.0 --dataset_in=MNIST --num_reps=2 --class_1=1 --class_2=9
```

The optimal probabilities will be stored in the `optimal_probs` folder

To obtain lower bounds on cross-entropy for synthetic Gaussians of dimensions 2, 10 and 100, run:
```
python gaussian_log_loss.py
```

## Robust training of models

Create a directory called `logs`. Run `train.sh` followed by `eval.sh`.
