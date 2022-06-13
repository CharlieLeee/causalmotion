## Train Vanilla ERM with ground truth style
## General parameters
GPU=0 # 1. Set GPU

dataset='synthetic_lr_v2' # 2. Set dataset
f_envs='0.1l-0.1r-0.3l-0.3r-0.5l-0.5r'
train_len=9000
DATA="--dataset_name $dataset --filter_envs $f_envs --reduceall ${train_len}"
bs=64

## Method (uncomment the method of choice)

e='0-0-0-1000-0-0'
# irm=1.0 # 3. Set IRM weight
dbottle=64
irm=1.0
lr=1e-3
enwidth=8
norm='none'

for seed in 1 2 3
do
    exp="gt_style_causal_int_norm_${norm}"
    echo $exp
    DIR="--tfdir causal_runs/${dataset}/${exp}/${irm}"
    TRAINING="--num_epochs $e --batch_size $bs --counter false --irm $irm --exp $exp --lrstyle $lr --gt_style --gt_encoder $enwidth --decoder_bottle $dbottle --lrstgat $lr --visualize_prediction --norm_type ${norm} --causal_decoder" # if visualize prediction then add --visualize_prediction
    echo $DIR
    CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL $USUAL --seed $seed &
done
