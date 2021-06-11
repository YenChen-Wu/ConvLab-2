#!/bin/bash

################ Benchmark ################
function run_benchmark {
# download best model under mle/save
python convlab2/policy/evaluate.py --model_name MLE --log_path_suffix mle
python convlab2/policy/evaluate.py --model_name GDPL --log_path_suffix gdpl
python convlab2/policy/evaluate.py --model_name PPO --log_path_suffix ppo
python convlab2/policy/evaluate.py --model_name PG --log_path_suffix pg
}

################ MLE ################

function run_mle {
    path=convlab2/policy/mle/multiwoz
    
    if [ -d $path/save_$1 ]
    then
        echo 'Directory of the model exists. Skip training phase'
    else
        cd $path & python train.py
        mv save save_$1 & cd ../../../..
    fi
    
    if [ -d log/mle/$1 ]
    then
        echo 'Directory of the log exists. Skip testing phase'
    else
        for i in {0..15}; do
        python convlab2/policy/evaluate.py --model_name MLE \
                                           --load_path $path/save_$1/${i} \
                                           --log_path_suffix mle_${i} \
                                           --log_dir_path log/mle/$1
        done

        python convlab2/policy/evaluate.py --model_name MLE \
                                           --load_path $path/save_$1/best \
                                           --log_path_suffix mle_best \
                                           --log_dir_path log/mle/$1
    fi
}

################ PPO / LCPO ################
function run {
    model=$1
    epoch=$2
    bsize=$3
    nproc=$4
    fname=$5
    seed=$6
    load=${7-../mle/multiwoz/save_2/best_mle}
    interval=$8
    
    path=convlab2/policy
    
    if [ -d $path/$model/model/${fname}/$seed ]
    then
        echo 'Directory of the model exists. Skip training phase'
    else
        cd $path/$model
        python train.py --load_path $load \
                        --epoch $epoch --batchsz $bsize --process_num $nproc
        log_file=$(ls log/log_20* | sort -V | tail -n1)
        mv $log_file log/log_${fname}_$seed.txt
        mkdir -p model/${fname}/
        mv save model/${fname}/$seed
        cd ../../..
    fi


    if [ -d log/$fname/$seed ]
    then
        echo 'Directory of the log exists. Skip testing phase'
    else
        for (( i=$interval-1 ; i<$epoch ; i+=$interval )); do
        python $path/evaluate.py --model_name ${model^^} \
                                 --load_path $path/$model/model/${fname}/${seed}/${i} \
                                 --log_path_suffix ${i} \
                                 --log_dir_path log/$fname/$seed
        done
    fi
}

function rm_exp {
    echo haha
}
################ Baselines ################
# run_benchmark
# for i in {3..4}; do run_mle $i; done
# run ppo 200 1024 8 ppo 0 ../mle/multiwoz/save_2/best_mle 5
# for i in {1..2}; do run ppo 200 1024 8 ppo_zero $i None 5; done
# run lcpo 200 1024 8 lcpo 1 ../mle/multiwoz/save_2/best_mle 5
# run lcpo 200 1024 8 lcpo_zero 0 None 5

################ EXP 1: update interval (batchsz) ################
### takes 20/30/50/15 min train + 10 min test
# run lcpo 100 128 8 lcpo_Q 0 ../mle/multiwoz/save_2/best_mle 10
# run lcpo 100 256 8 lcpo_Q 1 ../mle/multiwoz/save_2/best_mle 10
# run lcpo 100 512 8 lcpo_Q 2 ../mle/multiwoz/save_2/best_mle 10
# run lcpo 100 64 8 lcpo_Q 3 ../mle/multiwoz/save_2/best_mle 10

################ EXP 2: n process ################
### takes 16/18/26 min train + 10 min test
# run lcpo 100 128 4 lcpo_P 0 ../mle/multiwoz/save_2/best_mle 10
# run lcpo 100 128 2 lcpo_P 1 ../mle/multiwoz/save_2/best_mle 10
# run lcpo 100 128 1 lcpo_P 2 ../mle/multiwoz/save_2/best_mle 10

################ EXP 3: test for debugging ################
# run ppo  2 128 4 test 0 ../mle/multiwoz/save_2/best_mle 100
# run lcpo 2 128 4 test 0 ../mle/multiwoz/save_2/best_mle 100
# run ppo  1 128 4 test 0 None 100
# run lcpo 1 128 4 test 0 None 100
# run lcpo 1 128 4 test 0 model/lcpo_0526/0/104 100
# rm -r $path/lcpo/test_0

################ EXP 4: new baselines, x means light-pack settings ################
# for i in {3..3}; do run ppo  100 128 4 ppo_x $i ../mle/multiwoz/save_2/best_mle 5; done
# for i in {0..0}; do run ppo  100 128 4 ppo_zero_x $i None 5; done
# for i in {2..4}; do run lcpo 100 128 4 lcpo_x $i ../mle/multiwoz/save_2/best_mle 5; done
# for i in {0..0}; do run lcpo 100 128 4 lcpo_zero_x $i None 5; done

################ EXP 5: no domain rewards 5 ################
# for i in {0..2}; do run ppo  100 128 4 ppo_xr $i ../mle/multiwoz/save_2/best_mle 5; done
# for i in {0..2}; do run ppo  100 128 4 ppo_zero_xr $i None 5; done
# for i in {0..2}; do run lcpo 100 128 4 lcpo_xr $i ../mle/multiwoz/save_2/best_mle 5; done
# for i in {0..2}; do run lcpo 100 128 4 lcpo_zero_xr $i None 5; done

# for i in {0..0}; do run lcpo 200 1024 4 lcpo_r $i ../mle/multiwoz/save_2/best_mle 10; done

################ EXP 6: improve lcpo ################
# prev_v = 40
# for i in {0..4}; do run lcpo 200 128 4 lcpo_0525 $i ../mle/multiwoz/save_2/best_mle 10; done
# prev_v = v[t]+1
# for i in {0..1}; do run lcpo 200 128 4 lcpo_0526 $i ../mle/multiwoz/save_2/best_mle 10; done
# prev_v = v[t], v explode
# for i in {0..1}; do run lcpo 200 128 4 lcpo_0527 $i ../mle/multiwoz/save_2/best_mle 10; done
# 0:prev_v = prev_v 1:?? 2:bound=r-v[t]-v[t] (2 is good, bound is important for noisy env)
# for i in {2..2}; do run lcpo 200 1024 4 lcpo_0528 $i ../mle/multiwoz/save_2/best_mle 10; done
# int=128, soso
# for i in {0..0}; do run lcpo 200 128 4 lcpo_0529 $i ../mle/multiwoz/save_2/best_mle 10; done
# normalized reward, good but become bad in the end, 0.94
# for i in {0..1}; do run lcpo 200 1024 4 lcpo_0530 $i ../mle/multiwoz/save_2/best_mle 10; done
# sigmoid v + bce loss, i128, unstable, 0 is good
# for i in {0..4}; do run lcpo 200 128 4 lcpo_0532 $i ../mle/multiwoz/save_2/best_mle 10; done
# 0532 + prev_v=v[t], very bad, why? unstable?
# for i in {0..0}; do run lcpo 200 128 4 lcpo_0533 $i ../mle/multiwoz/save_2/best_mle 10; done
# my reward, unstable, v_lr too small?
# for i in {0..4}; do run lcpo 200 128 4 lcpo_0534 $i ../mle/multiwoz/save_2/best_mle 10; done
# lr*10, very unstable, p_lr too big
# for i in {0..4}; do run lcpo 200 128 4 lcpo_0535 $i ../mle/multiwoz/save_2/best_mle 10; done
# 256i prev_v=v[t], bound-, v_lr*10 good, can be baseline
# for i in {0..1}; do run lcpo 200 256 4 lcpo_0536 $i ../mle/multiwoz/save_2/best_mle 10; done
# bound=r-v
# for i in {0..7}; do run lcpo 200 256 4 lcpo_0537 $i ../mle/multiwoz/save_2/best_mle 10; done
# for i in {0..0}; do run lcpo 200 1024 4 lcpo_0540 $i ../mle/multiwoz/save_2/best_mle 10; done
# bound=r
# for i in {0..1}; do run lcpo 200 256 4 lcpo_0538 $i ../mle/multiwoz/save_2/best_mle 10; done
# for i in {0..1}; do run lcpo 200 1024 4 lcpo_0539 $i ../mle/multiwoz/save_2/best_mle 10; done
# adv update order
# zero
# for i in {0..0}; do run lcpo 200 1024 4 lcpo_0541 $i None 10; done
# no adv clipping
# for i in {0..2}; do run lcpo 200 128 4 lcpo_0542 $i ../mle/multiwoz/save_2/best_mle 10; done
# mse loss
# for i in {0..0}; do run lcpo 200 128 4 lcpo_0543 $i ../mle/multiwoz/save_2/best_mle 10; done
# PPO BCE loss
# for i in {0..1}; do run ppo 200 128 4 ppo_0544 $i ../mle/multiwoz/save_2/best_mle 10; done
# init 0
# for i in {0..0}; do run lcpo 200 128 4 lcpo_0545 $i ../mle/multiwoz/save_2/best_mle 10; done

################ baselines, xrs (light settings, reward in the end, sigmoid output) ################
# for i in {0..5}; do run ppo 200 128 4 ppo_xrs $i ../mle/multiwoz/save_2/best_mle 5; done
# for i in {0..5}; do run lcpo 200 128 4 lcpo_xrs $i ../mle/multiwoz/save_2/best_mle 5; done

################ EXP 7: advantage normalization ################
# advantage norm
# for i in {0..2}; do run ppo 200 128 4 ppo_0602 $i ../mle/multiwoz/save_2/best_mle 5; done
# no advantage norm
# for i in {0..2}; do run ppo 200 128 4 ppo_0601 $i ../mle/multiwoz/save_2/best_mle 5; done
# only mean norm
# for i in {0..2}; do run ppo 200 128 4 ppo_0603 $i ../mle/multiwoz/save_2/best_mle 5; done
# only scaling
# for i in {0..2}; do run ppo 200 128 4 ppo_0604 $i ../mle/multiwoz/save_2/best_mle 5; done

# adv.mean()+0.01
# for i in {0..2}; do run ppo 200 128 4 ppo_0605 $i ../mle/multiwoz/save_2/best_mle 5; done
# adv.mean()-0.01
# for i in {0..2}; do run ppo 200 128 4 ppo_0606 $i ../mle/multiwoz/save_2/best_mle 5; done

function ppo_config() {
    read_file=./convlab2/policy/ppo/config/config_best.json
    write_file=./convlab2/policy/ppo/config/config.json
    p_lr=$1
    v_lr=$2
    r_sc=$3
    ARG1=${1:-foo}
    cat ${read_file} | sed "/policy_lr/s/: .*$/: ${p_lr},/g" \
                     | sed "/value_lr/s/: .*$/: ${v_lr},/g"  \
                     | sed "/reward_scale/s/: .*$/: ${r_sc},/g"  > ${write_file}
}
function config () {
    model=$1
    name=$2
    value=$3
    read_file=./convlab2/policy/${model}/config/${4-config_best.json}
    write_file=./convlab2/policy/${model}/config/config.json
    
    if [${read_file} = ${write_file}]; then
        sed -i "/${name}/s/: .*$/: ${value},/" ${read_file}
    else
        sed "/${name}/s/: .*$/: ${value},/" ${read_file} > ${write_file}
    fi
}

################ EXP 8: policy learning rate ################
# the same as 0602
# for i in {0..2}; do run ppo 200 128 4 ppo_0607 $i ../mle/multiwoz/save_2/best_mle 5; done
# lr = 2e-4
# ppo_config 0.0002 0.00005
# for i in {0..5}; do run ppo 200 128 4 ppo_0608 $i ../mle/multiwoz/save_2/best_mle 5; done
# # lr = 5e-4
# ppo_config 0.0005 0.00005
# for i in {0..5}; do run ppo 200 128 4 ppo_0609 $i ../mle/multiwoz/save_2/best_mle 5; done
# #
# ppo_config 0.00005 0.00005
# for i in {0..5}; do run ppo 200 128 4 ppo_0610 $i ../mle/multiwoz/save_2/best_mle 5; done

################ EXP 9: reward scaling ################
# have to tune learning rate as well?
# no sigmoid in v_net, baseline, R=40
# ppo_config 0.0001 0.00005 1.0
# for i in {0..4}; do run ppo 200 128 4 ppo_0611 $i ../mle/multiwoz/save_2/best_mle 5; done
# # R=20
# ppo_config 0.0001 0.00005 2.0
# for i in {0..4}; do run ppo 200 128 4 ppo_0612 $i ../mle/multiwoz/save_2/best_mle 5; done
# # R=10 (optimal value in theory)
# ppo_config 0.0001 0.00005 4.0
# for i in {0..4}; do run ppo 200 128 4 ppo_0613 $i ../mle/multiwoz/save_2/best_mle 5; done
# # R=5
# ppo_config 0.0001 0.00005 8.0
# for i in {0..4}; do run ppo 200 128 4 ppo_0614 $i ../mle/multiwoz/save_2/best_mle 5; done
# # R=1
# ppo_config 0.0001 0.00005 40.0
# for i in {0..4}; do run ppo 200 128 4 ppo_0615 $i ../mle/multiwoz/save_2/best_mle 5; done
# R=2
# ppo_config 0.0001 0.00005 20.0
# for i in {0..4}; do run ppo 200 128 4 ppo_0616 $i ../mle/multiwoz/save_2/best_mle 5; done
# R=0.2
# ppo_config 0.0001 0.00005 200.0
# for i in {0..4}; do run ppo 200 128 4 ppo_0617 $i ../mle/multiwoz/save_2/best_mle 5; done
# # R = 200
# ppo_config 0.0001 0.00005 0.2
# for i in {0..4}; do run ppo 200 128 4 ppo_0618 $i ../mle/multiwoz/save_2/best_mle 5; done


################ EXP 10: GAE lambda (tau) ################
# config ppo tau 0.9
# for i in {0..4}; do run ppo 200 128 4 ppo_0619 $i ../mle/multiwoz/save_2/best_mle 5; done
# config ppo tau 0.8
# for i in {0..4}; do run ppo 200 128 4 ppo_0620 $i ../mle/multiwoz/save_2/best_mle 5; done
# config ppo tau 0.7
# for i in {0..4}; do run ppo 200 128 4 ppo_0621 $i ../mle/multiwoz/save_2/best_mle 5; done
# config ppo tau 0.6
# for i in {0..4}; do run ppo 200 128 4 ppo_0622 $i ../mle/multiwoz/save_2/best_mle 5; done
# config ppo tau 0.5
# for i in {0..4}; do run ppo 200 128 4 ppo_0623 $i ../mle/multiwoz/save_2/best_mle 5; done
# config ppo tau 0.4
# for i in {0..4}; do run ppo 200 128 4 ppo_0624 $i ../mle/multiwoz/save_2/best_mle 5; done
# config ppo tau 0.3
# for i in {0..4}; do run ppo 200 128 4 ppo_0625 $i ../mle/multiwoz/save_2/best_mle 5; done
# config ppo tau 0.2
# for i in {0..4}; do run ppo 200 128 4 ppo_0626 $i ../mle/multiwoz/save_2/best_mle 5; done
# config ppo tau 0.1
# for i in {0..4}; do run ppo 200 128 4 ppo_0627 $i ../mle/multiwoz/save_2/best_mle 5; done
# config ppo tau 0.0
# for i in {0..4}; do run ppo 200 128 4 ppo_0628 $i ../mle/multiwoz/save_2/best_mle 5; done


################ EXP 10: sanity check + lcpo with pseudo_v ################
# check reward scaling is equivalent to lr 
# config ppo value_lr 0.00025
# for i in {0..4}; do run ppo 200 128 4 ppo_0629 $i ../mle/multiwoz/save_2/best_mle 5; done
# # check settings is the same as ppo
# config lcpo adv_est \"ppo\"
# for i in {0..4}; do run lcpo 200 128 4 lcpo_0630 $i ../mle/multiwoz/save_2/best_mle 5; done
# # lcpo pseudo v
# config lcpo adv_est \"lcpo\"
# for i in {0..4}; do run lcpo 200 128 4 lcpo_0631 $i ../mle/multiwoz/save_2/best_mle 5; done
# do it again, collect stat
config lcpo adv_est \"lcpo\"
for i in {0..4}; do run lcpo 200 128 4 lcpo_0632 $i ../mle/multiwoz/save_2/best_mle 5; done
# tau=0.5
config lcpo tau 0.5
for i in {0..4}; do run lcpo 200 128 4 lcpo_0633 $i ../mle/multiwoz/save_2/best_mle 5; done
# tau=0
config lcpo tau 0.0
for i in {0..4}; do run lcpo 200 128 4 lcpo_0634 $i ../mle/multiwoz/save_2/best_mle 5; done
# rs = lr ? again
config ppo value_lr 0.00025
config ppo reward_scale 1 config.json
for i in {0..4}; do run ppo 200 128 4 ppo_0635 $i ../mle/multiwoz/save_2/best_mle 5; done

################ Decision Transformer ################ (6/11)
# cd ./convlab2/policy/dt
# python train.py --load_path ../mle/multiwoz/save_2/best_mle
# cd ../../..
################ EXP 11: move on to hpc and search best config ################ (6/12)
# NOT here

################ baselines, xrsa (single action) if pseodo_v works well ################ 
# single action from scratch

