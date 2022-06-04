## Train Vanilla ERM with ground truth style
## General parameters
GPU=1 # 1. Set GPU

dataset='synthetic_lr_v3' # 2. Set dataset
f_envs='0.1l-0.1r-0.3l-0.3r-0.5l-0.5r-0.6l-0.6r-0.7l-0.7r-0.8l-0.8r'
train_len=10000
DATA="--dataset_name $dataset --filter_envs $f_envs --reduceall ${train_len}"

bs=64

## Method (uncomment the method of choice)

e='0-0-0-1000-0-0'
# irm=1.0 # 3. Set IRM weight
dbottle=16

for seed in 1 #2 3 4 5
do
    for irm in 0.0 #1.0 
    do
        for lr in  1e-3 #1e-3 5e-4 # 1e-3 3e-4  
        do
            for enwidth in 8 #16 32 64
            do
                exp="gt_style_${enwidth}_${train_len}_${dbottle}_${lr}"
                echo $exp
                DIR="--tfdir new_runs/${dataset}/${exp}/${irm}"
                TRAINING="--num_epochs $e --batch_size $bs --counter false --irm $irm --exp $exp --lrstyle $lr --gt_style --gt_encoder $enwidth --decoder_bottle $dbottle --lrstgat $lr --visualize_prediction" # if visualize prediction then add --visualize_prediction
                echo $DIR
                CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL $USUAL --seed $seed 
            done 
            
        done
    done
done
#CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL $USUAL --seed 5


------------------------------ Experiment script below ------------------------

### Training best learning rate
# e='0-0-0-50-0-0'
# # irm=1.0 # 3. Set IRM weight
# dbottle=16
# enwidth=8
# train_len=30
# DATA="--dataset_name $dataset --filter_envs $f_envs --reduceall ${train_len}"


# for seed in 1 #2 3 4
# do
#     for irm in 0.0 #1.0 #32 #8
#     do
#         for stylelr in  1e-3 5e-3 1e-2 5e-4 #1e-3 5e-4 # 1e-3 3e-4  
#         do
#             for stgatlr in 1e-3 5e-3 1e-2 5e-4 #16 32 64
#             do
#                 exp="gt_style_${enwidth}_${train_len}_${dbottle}_style_${stylelr}_stgat_${stgatlr}_tweaking"
#                 echo $exp
#                 DIR="--tfdir runs/${dataset}/${exp}/${irm}"
#                 TRAINING="--num_epochs $e --batch_size $bs --counter false --irm $irm --exp $exp --lrstyle $stylelr --gt_style --gt_encoder $enwidth --decoder_bottle $dbottle --lrstgat $stgatlr --visualize_prediction" # 4. Set Counter --visualize_prediction
#                 echo $DIR
#                 CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL $USUAL --seed $seed &
#             done 
            
#         done
#     done
# done