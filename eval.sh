run_mnist() {
    # $1: epsilon, gpu:2
    CUDA_VISIBLE_DEVICES=$2 python -u eval.py --configs configs/configs_mnist_3_7.yml  --print-freq 10 --val-method auto --epsilon $1 --ckpt ./data/mnist_3_7_trainer_madry_final_steps_40_epsilon_$1/trial_0/checkpoint/checkpoint.pth.tar --exp-name mnist_3_7_trainer_madry_final_eval_epsilon_$1 | tee -a ./logs/mnist_3_7_trainer_madry_final_eval_epsilon_$1.txt &
    
    CUDA_VISIBLE_DEVICES=$2 python -u eval_script.py --configs configs/configs_mnist_3_7.yml  --print-freq 10 --val-method auto --epsilon $1 --ckpt ./results/mnist_3_7_trainer_trades_final_steps_40_epsilon_$1_beta_1/trial_0/checkpoint/checkpoint.pth.tar --exp-name mnist_3_7_trainer_trades_final_eval_epsilon_$1_beta_1 | tee -a ./logs/mnist_3_7_trainer_trades_final_eval_epsilon_$1_beta_1.txt &
    
    CUDA_VISIBLE_DEVICES=$2 python -u eval.py --configs configs/configs_mnist_3_7.yml  --print-freq 10 --val-method auto --epsilon $1 --ckpt ./data/mnist_3_7_trainer_trades_final_steps_40_epsilon_$1_beta_6/trial_0/checkpoint/checkpoint.pth.tar --exp-name mnist_3_7_trainer_trades_final_eval_epsilon_$1_beta_6 | tee -a ./logs/mnist_3_7_trainer_trades_final_eval_epsilon_$1_beta_6.txt &
    
    CUDA_VISIBLE_DEVICES=$2 python -u eval.py --configs configs/configs_mnist_3_7.yml  --print-freq 10 --val-method auto --epsilon $1 --ckpt ./data/mnist_3_7_trainer_madry_final_steps_40_epsilon_$1_opt_probs/trial_0/checkpoint/checkpoint.pth.tar --exp-name mnist_3_7_trainer_madry_final_eval_epsilon_$1_opt_probs | tee -a ./logs/mnist_3_7_trainer_madry_final_eval_epsilon_$1_opt_probs.txt &
    
    CUDA_VISIBLE_DEVICES=$2 python -u eval.py --configs configs/configs_mnist_3_7.yml  --print-freq 10 --val-method auto --epsilon $1 --ckpt ./data/mnist_3_7_trainer_madry_final_steps_40_epsilon_$1_opt_probs_clip/trial_0/checkpoint/checkpoint.pth.tar --exp-name mnist_3_7_trainer_madry_final_eval_epsilon_$1_opt_probs_clip | tee -a ./logs/mnist_3_7_trainer_madry_final_eval_epsilon_$1_opt_probs_clip.txt ;

}


run_fmnist() {
    # $1: epsilon, gpu:2
    CUDA_VISIBLE_DEVICES=$2 python -u eval.py --configs configs/configs_fmnist_3_7.yml  --print-freq 10 --val-method auto --epsilon $1 --ckpt ./data/fmnist_3_7_trainer_madry_final_steps_40_epsilon_$1/trial_0/checkpoint/checkpoint.pth.tar --exp-name fmnist_3_7_trainer_madry_final_eval_epsilon_$1 | tee -a ./logs/fmnist_3_7_trainer_madry_final_eval_epsilon_$1.txt &
    
    CUDA_VISIBLE_DEVICES=$2 python -u eval.py --configs configs/configs_fmnist_3_7.yml  --print-freq 10 --val-method auto --epsilon $1 --ckpt ./data/fmnist_3_7_trainer_trades_final_steps_40_epsilon_$1_beta_1/trial_0/checkpoint/checkpoint.pth.tar --exp-name fmnist_3_7_trainer_trades_final_eval_epsilon_$1_beta_1 | tee -a ./logs/fmnist_3_7_trainer_trades_final_eval_epsilon_$1_beta_1.txt &
    
    CUDA_VISIBLE_DEVICES=$2 python -u eval.py --configs configs/configs_fmnist_3_7.yml  --print-freq 10 --val-method auto --epsilon $1 --ckpt ./data/fmnist_3_7_trainer_trades_final_steps_40_epsilon_$1_beta_6/trial_0/checkpoint/checkpoint.pth.tar --exp-name fmnist_3_7_trainer_trades_final_eval_epsilon_$1_beta_6 | tee -a ./logs/fmnist_3_7_trainer_trades_final_eval_epsilon_$1_beta_6.txt &
    
    CUDA_VISIBLE_DEVICES=$2 python -u eval.py --configs configs/configs_fmnist_3_7.yml  --print-freq 10 --val-method auto --epsilon $1 --ckpt ./data/fmnist_3_7_trainer_madry_final_steps_40_epsilon_$1_opt_probs/trial_0/checkpoint/checkpoint.pth.tar --exp-name fmnist_3_7_trainer_madry_final_eval_epsilon_$1_opt_probs | tee -a ./logs/fmnist_3_7_trainer_madry_final_eval_epsilon_$1_opt_probs.txt &
    
    CUDA_VISIBLE_DEVICES=$2 python -u eval.py --configs configs/configs_fmnist_3_7.yml  --print-freq 10 --val-method auto --epsilon $1 --ckpt ./data/fmnist_3_7_trainer_madry_final_steps_40_epsilon_$1_opt_probs_clip/trial_0/checkpoint/checkpoint.pth.tar --exp-name fmnist_3_7_trainer_madry_final_eval_epsilon_$1_opt_probs_clip | tee -a ./logs/fmnist_3_7_trainer_madry_final_eval_epsilon_$1_opt_probs_clip.txt ;

}


run_fmnist 3.0 0 &
run_fmnist 3.6 1 &
run_fmnist 4.0 2 &
run_fmnist 4.6 3 &
run_fmnist 5.0 4 &
run_fmnist 5.6 5 &
run_fmnist 6.0 6 ;
wait;
echo "done";


run_mnist 3.0 1 &
run_mnist 3.6 2 &
run_mnist 4.0 3 &
run_mnist 4.6 4 &
run_mnist 5.0 5 ;
wait;
echo "done";
