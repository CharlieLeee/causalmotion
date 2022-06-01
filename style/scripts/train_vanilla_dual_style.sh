# Vanilla ERM with dual style dataset

## General parameters
GPU=0 # 1. Set GPU
exp='vanilla_style'

dataset='synthetic_lr_v2' # 2. Set dataset
f_envs='0.1r-0.1l-0.3r-0.3l-0.5r-0.5l'
DATA="--dataset_name $dataset --filter_envs $f_envs --reduceall 10000"
DIR="--tfdir new_runs/$dataset/$exp/$irm"
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
e='0-0-1000-0-0-0'
irm=1.0 # 3. Set IRM weight # without irm 0.06 0.1

dbottle=16
lr=1e-3

TRAINING="--num_epochs $e --batch_size $bs --exp $exp --irm $irm --counter false --decoder_bottle $dbottle --lrstgat $lr --visualize_prediction" # 4. Set Counter

for seed in 1 #2 3 4
do  
    CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL $USUAL --seed $seed 
done
# CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL $USUAL --seed 5
