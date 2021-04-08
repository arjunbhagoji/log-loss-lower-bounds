# we will madry, trades (beta 1.0, 6.0), soft-probs, soft-probs-clip (all in one go)

run_mnist() {
    # $1: epsilon, gpu:2
    CUDA_VISIBLE_DEVICES=$2 python -u train.py --configs configs/configs_mnist_3_7.yml  --print-freq 10 --trainer madry --val-method adv --epsilon $1 --num-steps 40 --step-size 0.4  --exp-name mnist_3_7_trainer_madry_final_steps_40_epsilon_$1 --epoch 100 --fp16 | tee -a ./logs/mnist_3_7_trainer_madry_final_steps_40_epsilon_$1.txt &
    
    CUDA_VISIBLE_DEVICES=$2 python -u train_script.py --configs configs/configs_mnist_3_7.yml  --print-freq 10 --trainer adv --val-method adv --epsilon $1 --num-steps 40 --step-size 0.4  --exp-name mnist_3_7_trainer_trades_final_steps_40_epsilon_$1_beta_1 --epoch 100 --beta 1.0 | tee -a ./logs/mnist_3_7_trainer_trades_final_steps_40_epsilon_$1_beta_1.txt &
    
    CUDA_VISIBLE_DEVICES=$2 python -u train_script.py --configs configs/configs_mnist_3_7.yml  --print-freq 10 --trainer adv --val-method adv --epsilon $1 --num-steps 40 --step-size 0.4  --exp-name mnist_3_7_trainer_trades_final_steps_40_epsilon_$1_beta_6 --epoch 100 --beta 6.0 | tee -a ./logs/mnist_3_7_trainer_trades_final_steps_40_epsilon_$1_beta_6.txt &
    
    CUDA_VISIBLE_DEVICES=$2 python -u train.py --configs configs/configs_mnist_3_7.yml  --print-freq 10 --trainer madry --val-method adv --epsilon $1 --num-steps 40 --step-size 0.4  --exp-name mnist_3_7_trainer_madry_final_steps_40_epsilon_$1_opt_probs --epoch 100 --fp16 --opt-probs | tee -a ./logs/mnist_3_7_trainer_madry_final_steps_40_epsilon_$1_opt_probs.txt &
    
    CUDA_VISIBLE_DEVICES=$2 python -u train.py --configs configs/configs_mnist_3_7.yml  --print-freq 10 --trainer madry --val-method adv --epsilon $1 --num-steps 40 --step-size 0.4  --exp-name mnist_3_7_trainer_madry_final_steps_40_epsilon_$1_opt_probs_clip --epoch 100 --fp16 --opt-probs --clip-soft-labels | tee -a ./logs/mnist_3_7_trainer_madry_final_steps_40_epsilon_$1_opt_probs_clip.txt 

}


run_fmnist() {
    # $1: epsilon, gpu:2
    CUDA_VISIBLE_DEVICES=$2 python -u train.py --configs configs/configs_fmnist_3_7.yml  --print-freq 10 --trainer madry --val-method adv --epsilon $1 --num-steps 40 --step-size 0.4  --exp-name fmnist_3_7_trainer_madry_final_steps_40_epsilon_$1 --epoch 100 --fp16 | tee -a ./logs/fmnist_3_7_trainer_madry_final_steps_40_epsilon_$1.txt &
    
    CUDA_VISIBLE_DEVICES=$2 python -u train_script.py --configs configs/configs_fmnist_3_7.yml  --print-freq 10 --trainer adv --val-method adv --epsilon $1 --num-steps 40 --step-size 0.4  --exp-name fmnist_3_7_trainer_trades_final_steps_40_epsilon_$1_beta_1 --epoch 100 --beta 1.0 | tee -a ./logs/fmnist_3_7_trainer_trades_final_steps_40_epsilon_$1_beta_1.txt &
    
    CUDA_VISIBLE_DEVICES=$2 python -u train_script.py --configs configs/configs_fmnist_3_7.yml  --print-freq 10 --trainer adv --val-method adv --epsilon $1 --num-steps 40 --step-size 0.4  --exp-name fmnist_3_7_trainer_trades_final_steps_40_epsilon_$1_beta_6 --epoch 100 --beta 6.0 | tee -a ./logs/fmnist_3_7_trainer_trades_final_steps_40_epsilon_$1_beta_6.txt &
    
    CUDA_VISIBLE_DEVICES=$2 python -u train.py --configs configs/configs_fmnist_3_7.yml  --print-freq 10 --trainer madry --val-method adv --epsilon $1 --num-steps 40 --step-size 0.4  --exp-name fmnist_3_7_trainer_madry_final_steps_40_epsilon_$1_opt_probs --epoch 100 --fp16 --opt-probs | tee -a ./logs/fmnist_3_7_trainer_madry_final_steps_40_epsilon_$1_opt_probs.txt &
    
    CUDA_VISIBLE_DEVICES=$2 python -u train.py --configs configs/configs_fmnist_3_7.yml  --print-freq 10 --trainer madry --val-method adv --epsilon $1 --num-steps 40 --step-size 0.4  --exp-name fmnist_3_7_trainer_madry_final_steps_40_epsilon_$1_opt_probs_clip --epoch 100 --fp16 --opt-probs --clip-soft-labels | tee -a ./logs/fmnist_3_7_trainer_madry_final_steps_40_epsilon_$1_opt_probs_clip.txt 

}


run_fmnist 3.0 0 &
run_fmnist 3.6 1 &
run_fmnist 4.0 2 &
run_fmnist 4.6 3 &
run_fmnist 5.0 4 &
run_fmnist 5.6 5 &
run_fmnist 6.0 0 ;
wait;
echo "done";

run_mnist 3.0 1 &
run_mnist 3.6 2 &
run_mnist 4.0 3 &
run_mnist 4.6 4 &
run_mnist 5.0 5 ;
wait;
echo "done";
