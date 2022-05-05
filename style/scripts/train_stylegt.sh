# PRETRAIN

## General parameters
GPU=0 # 1. Set GPU


dataset='synthetic_lr_v2' # 2. Set dataset
f_envs='0.1l-0.1r-0.3l-0.3r-0.5l-0.5r'
train_len=9000
DATA="--dataset_name $dataset --filter_envs $f_envs --reduceall ${train_len}"

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
e='0-0-500-700-0-0'
irm=1.0 # 3. Set IRM weight

for seed in 1 #2 3 4
do
    for dbottle in 16 32 #8
    do
        for lr in  1e-3 #5e-4 # 1e-3 3e-4  
        do
            for enwidth in 8 #16 32 64
            do
                exp="gt_style_${enwidth}_${train_len}_${dbottle}_${lr}_emb_two_stage"
                echo $exp
                DIR="--tfdir runs/${dataset}/${exp}/${irm}"
                TRAINING="--num_epochs $e --batch_size $bs --counter false --irm $irm --exp $exp --lrstyle $lr --gt_style --gt_encoder $enwidth --decoder_bottle $dbottle --visualize_prediction" # 4. Set Counter
                echo $DIR
                CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL $USUAL --seed $seed &
            done 
            
        done
    done
done
#CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL $USUAL --seed 5
