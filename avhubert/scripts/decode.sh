GROUP=test
MODALITIES="audio,video"
result=/data/results/avhubert/base_noise_ft_30h
MODEL_PATH="${result}/checkpoints/checkpoint_best.pt"
DATA_PATH=/data/DB/lrs3/433h_data
OUT_PATH="${result}/s2s/decode"

# set paths
AV_HUBERT=$(dirname "$(dirname "$(readlink -fm "$0")")")
ROOT=$(dirname "${AV_HUBERT}")
export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"

# start decoding
python -B ${AV_HUBERT}/infer_s2s.py \
    --config-dir ${AV_HUBERT}/conf \
    --config-name s2s_decode \
        common.user_dir=${AV_HUBERT} \
        override.modalities=[${MODALITIES}] \
        dataset.gen_subset=${GROUP} \
        override.data=${DATA_PATH} \
        override.label_dir=${DATA_PATH} \
        common_eval.path=${MODEL_PATH} \
        common_eval.results_path=${OUT_PATH} \
        override.noise_prob=0.0 \
        distributed_training.distributed_world_size=1