# Evaluate contrastive training on dual styles dataset


GPU=1 # 0. Set GPU
EVALUATION="--metrics accuracy"
exp="gt_style_baseline"
dataset="synthetic_lr_v2" # 1. Set Dataset
dset_type="test"
bs=64
reduceall=9000
shuffle=False
enwidth=8
dbottle=16
Visualize="--visualize_embedding --visualize_prediction"


step="P4" 
epochs_string='0-0-0-3700-0-0'
epoch=3700
irm=0.0 # 2. Set IRM
for f_envs in "0.1l" "0.1r" "0.2l" "0.2r" "0.3l" "0.3r" "0.4l" "0.4r" "0.5l" "0.5r" "0.6l" "0.6r" "0.7l" "0.7r" 
do
    MODEL="--gt_style --decoder_bottle $dbottle --gt_encoder $enwidth"
    DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type --classification 6 --shuffle ${shuffle}"
    for seed in 1 #1 2 3 4
    do  
        CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py ${MODEL} $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1l-0.1r-0.3l-0.3r-0.5l-0.5r]_ep_[$epochs_string]_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[$reduceall]_relsocial[True]stylefs[all].pth.tar" --exp ${exp} --seed $seed ${Visualize} 
    done
    # seed=5
    # CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1l-0.1r-0.3l-0.3r-0.5l-0.5r]_ep_[$epochs_string]_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[$reduceall]_relsocial[True]stylefs[all].pth.tar" --seed $seed --visualize_embedding
done
