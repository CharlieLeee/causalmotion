# PRETRAIN MODELS (v4 MLP)


GPU=0 # 0. Set GPU
EVALUATION="--metrics accuracy"
exp="pretrain"
dataset="left_right_synthetic" # 1. Set Dataset
dset_type="test"
bs=64
reduceall=9000

# Baseline

# step="P3" 
# epochs_string='0-0-100-0-0-0'
# epoch=100 
# irm=0.0 # 2. Set IRM
# for f_envs in "0.15" "0.2" "0.25" "0.3" "0.35" "0.4" 
# do
#     DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
#     for seed in 1 2 3 4
#     do  
#         CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1-0.3-0.5]_ep_[$epochs_string]_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[$reduceall]_relsocial[True]stylefs[all].pth.tar" --seed $seed &
#     done
#     seed=5
#     CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1-0.3-0.5]_ep_[$epochs_string]_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[$reduceall]_relsocial[True]stylefs[all].pth.tar" --seed $seed
# done


# Modular Architecture (Our)

step="P6" 
epochs_string='0-0-100-50-20-300'
epoch=470 
irm=1.0 # 2. Set IRM
for f_envs in "0.15l" "0.15r" "0.2l" "0.2r" "0.25l" "0.25r" "0.3l" "0.3r" "0.35l" "0.35r" "0.4l" "0.4r" 
do
    DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
    for seed in 1 2 3 4
    do  
        CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.15-0.2-0.35]_ep_[$epochs_string]_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[$reduceall]_relsocial[True]stylefs[all].pth.tar" --seed $seed &
    done
    seed=5
    CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.15-0.2-0.35]_ep_[$epochs_string]_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[$reduceall]_relsocial[True]stylefs[all].pth.tar" --seed $seed
done
