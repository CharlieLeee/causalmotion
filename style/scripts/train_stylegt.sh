# PRETRAIN

## General parameters
GPU=0 # 1. Set GPU


dataset='synthetic_lr_v2' # 2. Set dataset
f_envs='0.1l-0.1r-0.3l-0.3r-0.5l-0.5r'
DATA="--dataset_name $dataset --filter_envs $f_envs --reduceall 2000"
DIR="--tfdir runs/$dataset/$exp/$irm"
bs=64


### EXPLANATION OF THE TRAINING EPOCHS STEPS
# 1. (deprecated, was used for first step of stgat training)
# 2. (deprecated, was used for second step of stgat training)
# 3. inital training of the entire model, without any style input
# 4. train style encoder using classifier, separate from pipeline
# 5. train the integrator (that joins the style and the invariant features)
# 6. fine-tune the integrator, decoder, style encoder with everything working
### Epochs needs to be define as: e=N1-N2-N3-N4-N5-N6
### EXAMPLE: if you want 20 epochs of step 3, 5 of step 4, 10 of step 5 and 10 of step 6, it will be 0-0-20-5-10-10


## Method (uncomment the method of choice)

### Vanilla
e='0-0-0-2000-0-0'
irm=0.0 # 3. Set IRM weight

for seed in 1 #2 3 4
do
    for decoder_bottle in 2 4 8
    do
        for lr in 1e-3 3e-3 5e-3
        do
            exp="gt_style_120_hidden_2000_1e-3_$decoder_bottle_$lr"
            TRAINING="--num_epochs $e --batch_size $bs --counter false --irm $irm --exp $exp --lrstgat $lr --gt_style --gt_encoder 120 --decoder_bottle $decoder_bottle" # 4. Set Counter

            
        CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL $USUAL --seed $seed 
    done
done
#CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL $USUAL --seed 5
