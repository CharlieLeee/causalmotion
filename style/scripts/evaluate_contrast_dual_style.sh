# Evaluate contrastive training on dual styles dataset


GPU=0 # 0. Set GPU
EVALUATION="--metrics accuracy"
exp="pretrain"
dataset="synthetic_lr_v2" # 1. Set Dataset
dset_type="test"
bs=64
reduceall=9000

step="P6" 
epochs_string='0-0-100-50-20-300'
epoch=150 
irm=1.0 # 2. Set IRM
for f_envs in "0.1l" "0.1r" "0.2l" "0.2r" "0.3l" "0.3r" "0.4l" "0.4r" "0.5l" "0.5r" "0.6l" "0.6r" "0.7l" "0.7r" 
do
    DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type --classification 6"
    for seed in 5 #1 2 3 4
    do  
        CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1l-0.1r-0.3l-0.3r-0.5l-0.5r]_ep_[$epochs_string]_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[$reduceall]_relsocial[True]stylefs[all].pth.tar" --seed $seed --visualize_embedding &
    done
    # seed=5
    # CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1l-0.1r-0.3l-0.3r-0.5l-0.5r]_ep_[$epochs_string]_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[$reduceall]_relsocial[True]stylefs[all].pth.tar" --seed $seed --visualize_embedding
done
