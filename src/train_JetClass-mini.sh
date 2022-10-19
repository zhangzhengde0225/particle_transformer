#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_JetClass`
DATADIR=${DATADIR_JetClass_mini}
[[ -z $DATADIR ]] && DATADIR='./datasets/JetClass-mini'

# set a comment via `COMMENT`
suffix=${COMMENT}

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

epochs=3
samples_per_epoch=$((100 * 1024 / $NGPUS))  # 10,240,000
samples_per_epoch_val=$((100 * 128))  # 1,280,000
dataopts="--num-workers 1 --fetch-step 0.01"

# PN, PFN, PCNN, ParT
model=$1
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParticleTransformer.py --use-amp"
    batchopts="--batch-size 512 --start-lr 1e-3"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PFN" ]]; then
    modelopts="networks/example_PFN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="networks/example_PCNN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
else
    echo "Invalid model $model!"
    exit 1
fi

# "kin", "kinpid", "full"
FEATURE_TYPE=$2
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="full"

if ! [[ "${FEATURE_TYPE}" =~ ^(full|kin|kinpid)$ ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

# currently only Pythia
SAMPLE_TYPE=Pythia

$CMD \
    --data-train \
    "HToBB:${DATADIR}/${SAMPLE_TYPE}/train_1M/HToBB_*.root" \
    "HToCC:${DATADIR}/${SAMPLE_TYPE}/train_1M/HToCC_*.root" \
    "HToGG:${DATADIR}/${SAMPLE_TYPE}/train_1M/HToGG_*.root" \
    "HToWW2Q1L:${DATADIR}/${SAMPLE_TYPE}/train_1M/HToWW2Q1L_*.root" \
    "HToWW4Q:${DATADIR}/${SAMPLE_TYPE}/train_1M/HToWW4Q_*.root" \
    "TTBar:${DATADIR}/${SAMPLE_TYPE}/train_1M/TTBar_*.root" \
    "TTBarLep:${DATADIR}/${SAMPLE_TYPE}/train_1M/TTBarLep_*.root" \
    "WToQQ:${DATADIR}/${SAMPLE_TYPE}/train_1M/WToQQ_*.root" \
    "ZToQQ:${DATADIR}/${SAMPLE_TYPE}/train_1M/ZToQQ_*.root" \
    "ZJetsToNuNu:${DATADIR}/${SAMPLE_TYPE}/train_1M/ZJetsToNuNu_*.root" \
    --data-val "${DATADIR}/${SAMPLE_TYPE}/val_1M/*.root" \
    --data-test \
    "HToBB:${DATADIR}/${SAMPLE_TYPE}/test_1M/HToBB_*.root" \
    "HToCC:${DATADIR}/${SAMPLE_TYPE}/test_1M/HToCC_*.root" \
    "HToGG:${DATADIR}/${SAMPLE_TYPE}/test_1M/HToGG_*.root" \
    "HToWW2Q1L:${DATADIR}/${SAMPLE_TYPE}/test_1M/HToWW2Q1L_*.root" \
    "HToWW4Q:${DATADIR}/${SAMPLE_TYPE}/test_1M/HToWW4Q_*.root" \
    "TTBar:${DATADIR}/${SAMPLE_TYPE}/test_1M/TTBar_*.root" \
    "TTBarLep:${DATADIR}/${SAMPLE_TYPE}/test_1M/TTBarLep_*.root" \
    "WToQQ:${DATADIR}/${SAMPLE_TYPE}/test_1M/WToQQ_*.root" \
    "ZToQQ:${DATADIR}/${SAMPLE_TYPE}/test_1M/ZToQQ_*.root" \
    "ZJetsToNuNu:${DATADIR}/${SAMPLE_TYPE}/test_1M/ZJetsToNuNu_*.root" \
    --data-config data/JetClass/JetClass_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/JetClass/${SAMPLE_TYPE}/${FEATURE_TYPE}/${model}/{auto}${suffix}/net \
    $dataopts $batchopts \
    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} --num-epochs $epochs --gpus 0 \
    --optimizer ranger --log logs/JetClass_${SAMPLE_TYPE}_${FEATURE_TYPE}_${model}_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard JetClass_${SAMPLE_TYPE}_${FEATURE_TYPE}_${model}${suffix} \
    "${@:3}"
