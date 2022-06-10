# Style contrastive with dual styles
## General parameters
GPU=1 # 1. Set GPU

dataset='synthetic_lr_v2' # 2. Set dataset
f_envs='0.1l-0.1r-0.3l-0.3r-0.5l-0.5r'
DATA="--dataset_name $dataset --filter_envs $f_envs --reduceall 9000"

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

### Ours with IRM
USUAL="--contrastive 1 --classification 6" 
e='0-0-100-200-100-700' # Notice: if use sam, better cut the inv and decoder training to half to keep fairness
irm=1.0 # 3. Set IRM weight
dbottle=16
lr=1e-3
norm='none'


for seed in 1 2 3 4 5
do  
    for norm in 'none'  # 'layer' 'group' # 'none' # 'none' 
    do
        exp="contrast_dual_style_causaldecode_${norm}_OOD9_seed_${seed}_mored"
        DIR="--tfdir compare_runs/$dataset/$exp/$irm"
        TRAINING="--num_epochs $e --batch_size $bs --irm $irm --decoder_bottle $dbottle --lrstgat $lr --norm_type ${norm} --causal_decoder"
        CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL $USUAL --seed $seed --exp $exp --visualize_prediction & # --visualize_embedding
    done
done
#CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL $USUAL --seed 5 --visualize_embedding

