#avhubert모델 30h 모델 
PRETRAINED_MODEL_PATH=/home/dan/projects/av_hubert/lrs3_vox_noise_pt_iter5.pt
result=/data/results/avhubert/base_noise_ft_30h
config_name=base_noise_pt_noise_ft_30h.yaml
ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
AV_HUBERT=${ROOT}

export PYTHONPATH="/home/dan/projects/av_hubert/fairseq:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
data=/data/DB/lrs3/30h_data
bpe_model=/data/DB/lrs3/spm1000/spm_unigram1000.model

if [ -f "${result}/finish.txt" ]; then
    echo "=== [SKIP train] already finished ==="
else
        fairseq-hydra-train \
        --config-dir ${AV_HUBERT}/conf/av-finetune \
        --config-name $config_name \
        task.data=$data \
        task.label_dir=$data \
        task.tokenizer_bpe_model=$bpe_model \
        model.w2v_path=${PRETRAINED_MODEL_PATH} \
        common.user_dir=${PWD} \
        task.noise_wav=/data/DB/musan/tsv/all \
        hydra.run.dir=${result} \
        dataset.num_workers=6 \
        distributed_training.distributed_world_size=1 \
        distributed_training.nprocs_per_node=1 \
        optimization.update_freq=[4] \
        dataset.max_tokens=2000 \
        task.noise_prob=0.0 \
        checkpoint.save_interval=1 &&
    echo "finished : $(date '+%Y-%m-%d %H:%M:%S')" > ${result}/finish.txt
fi

if [ -f "${result}/s2s/total_wer.txt" ]; then
    echo "=== [SKIP infer] already has total_wer.txt ==="
else
    ${AV_HUBERT}/infer_all.sh ${result} mix
fi

