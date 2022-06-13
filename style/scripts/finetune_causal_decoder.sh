# FINE TUNE
GPU=2 # 0. Set GPU

# data
filter_envs='0.7r-0.7l' # 1. Set env(s) to filter for training
filter_envs_pretrain='0.1l-0.1r-0.3l-0.3r-0.5l-0.5r'
dataset='synthetic_lr_v2'
DATA="--dataset_name $dataset --filter_envs $filter_envs --filter_envs_pretrain $filter_envs_pretrain"
bs=64

# training
step='P6'
lrinteg=0.001
contrastive=0.05

## TO CHANGE DEPENDING ON PREVIOUS STEPS
p6=300 # number of finetuning steps
oldreduceall=9000
epoch_string='0-0-100-200-100-600'
epoch=500 # sum of above
irm=1.0 # 3. Set IRM (used in pretraining)
dbottle=64
norm='none'

for finetune in 'integ+'
do
    for seed in 1 2 3 4 5
    do
        # pretrained model
        exp="contrast_dual_style_causaldecode_${norm}_OOD7_seed_${seed}_db_${dbottle}"
        model_dir="./models/$dataset/$exp/$step/$irm/$finetune/$seed"
        DIR="--tfdir ft_runs/$dataset/$exp/$step/$irm/$finetune/$seed"

        TRAINING="--num_epochs $epoch-0-0-0-0-$p6 --batch_size $bs --finetune $finetune --lrinteg $lrinteg --contrastive $contrastive --irm $irm --decoder_bottle $dbottle --norm_type ${norm} --causal_decoder"

        for reduce in 64 128 192 256 320
        do
            CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $MODEL $DIR --reduce $reduce --original_seed $seed --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1l-0.1r-0.3l-0.3r-0.5l-0.5r]_ep_[$epoch_string]_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[$oldreduceall]_relsocial[True]stylefs[all].pth.tar" 
        done
        CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $MODEL $DIR --reduce 384 --original_seed $seed --resume "models/$dataset/$exp/$step/$irm/SSE_data_${dataset}_irm[${irm}]_filter_envs[0.1l-0.1r-0.3l-0.3r-0.5l-0.5r]_ep_[$epoch_string]_seed_${seed}_tstep_${step}_epoch_${epoch}_reduceall[$oldreduceall]_relsocial[True]stylefs[all].pth.tar"
    done
done