#!/bin/bash

# 선택 기준 : valid accuracy
# mix or best : 전체 구간에서 best
# last        : last 사용
# cl_loss     : 150k 이후 valid_loss 최저
# cl or cbcl  : 150k 이후 valid_accuracy 최고

conf_name=s2s_decode
result=${1:?"Usage: $0 <result_path> <noise_mode>"}
noise_mode=${2:?"Usage: $0 <result_path> <noise_mode>"}

# 터미널 출력과 동시에 로그 파일 저장
mkdir -p "${result}/s2s"
exec > >(tee "${result}/s2s/infer.log") 2>&1

# Parse hydra_train.log for global best step (no step filter)
_get_best_step() {
    local result_dir="$1"
    python3 - "${result_dir}" <<'PYEOF'
import json, sys, os
result = sys.argv[1]
log_path = os.path.join(result, "hydra_train.log")
best_loss = float('inf')
best_update = -1
with open(log_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if ' - ' in line:
            line = line.split(' - ', 1)[1]
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        num_updates = int(d.get("valid_num_updates", 0))
        loss = d.get("valid_loss")
        if loss is None:
            continue
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_update = num_updates
if best_update < 0:
    print("ERROR: no valid_loss found in hydra_train.log", file=sys.stderr)
    sys.exit(1)
print(f"{best_update} {best_loss:.4f}")
PYEOF
}

# Select checkpoint based on noise_mode
if [[ "${noise_mode}" =~ ^[0-9]+$ ]]; then
    matches=(${result}/checkpoints/checkpoint_*_${noise_mode}.pt)
    if [ ${#matches[@]} -eq 0 ] || [ ! -f "${matches[0]}" ]; then
        echo "ERROR: checkpoint for update ${noise_mode} not found in ${result}/checkpoints/" >&2
        exit 1
    fi
    ckpt="${matches[0]}"
    echo "=== [step=${noise_mode}] Using checkpoint: ${ckpt} ==="
elif [ "${noise_mode}" == "mix" ] || [ "${noise_mode}" == "best" ]; then
    ckpt="${result}/checkpoints/checkpoint_best.pt"
    best_info=$(_get_best_step "${result}")
    if [ $? -eq 0 ]; then
        best_step=$(echo "${best_info}" | awk '{print $1}')
        best_loss_val=$(echo "${best_info}" | awk '{print $2}')
        echo "=== [${noise_mode}] Using checkpoint_best.pt (step ${best_step}, valid_loss=${best_loss_val}) ==="
    else
        echo "=== [${noise_mode}] Using checkpoint_best.pt ==="
    fi
elif [ "${noise_mode}" == "last" ]; then
    ckpt="${result}/checkpoints/checkpoint_last.pt"
    last_step=$(ls "${result}/checkpoints/" 2>/dev/null \
        | grep -E '^checkpoint_[0-9]+_[0-9]+\.pt$' \
        | sed 's/.*_\([0-9]*\)\.pt/\1/' \
        | sort -n | tail -1)
    if [ -n "${last_step}" ]; then
        echo "=== [last] Using checkpoint_last.pt (step ${last_step}) ==="
    else
        echo "=== [last] Using checkpoint_last.pt ==="
    fi
elif [ "${noise_mode}" == "cl_loss" ]; then
    # 150k 이후 valid_loss 최저 checkpoint 선택
    ckpt=$(python3 - "${result}" <<'EOF'
import json, sys, os, glob
result = sys.argv[1]
log_path = os.path.join(result, "hydra_train.log")
ckpt_dir = os.path.join(result, "checkpoints")

best_loss = float('inf')
best_update = -1

with open(log_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if ' - ' in line:
            line = line.split(' - ', 1)[1]
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        num_updates = int(d.get("valid_num_updates", 0))
        if num_updates <= 150000:
            continue
        loss = d.get("valid_loss")
        if loss is None:
            continue
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_update = num_updates

if best_update < 0:
    print("ERROR: no valid_loss found after 150k updates in hydra_train.log", file=sys.stderr)
    sys.exit(1)

matches = glob.glob(os.path.join(ckpt_dir, f"checkpoint_*_{best_update}.pt"))
if not matches:
    print(f"ERROR: checkpoint for update {best_update} not found", file=sys.stderr)
    sys.exit(1)
ckpt = matches[0]

print(f"[cl_loss] valid_loss={best_loss:.4f}  update={best_update}  ckpt={ckpt}", file=sys.stderr)
print(ckpt)
EOF
    )
    [ $? -ne 0 ] && exit 1
    echo "=== [${noise_mode}] Using checkpoint: ${ckpt} ==="
else
    # CL / CBCL: 150k 이후 valid_accuracy 최고 checkpoint 선택
    ckpt=$(python3 - "${result}" <<'EOF'
import json, sys, os, glob

result = sys.argv[1]
log_path = os.path.join(result, "hydra_train.log")
ckpt_dir = os.path.join(result, "checkpoints")

best_acc = -1
best_update = -1

with open(log_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if ' - ' in line:
            line = line.split(' - ', 1)[1]
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        num_updates = int(d.get("valid_num_updates", 0))
        if num_updates <= 150000:
            continue
        acc = d.get("valid_accuracy")
        if acc is None:
            continue
        if float(acc) > best_acc:
            best_acc = float(acc)
            best_update = num_updates

if best_update < 0:
    print("ERROR: no valid_accuracy found after 150k updates in hydra_train.log", file=sys.stderr)
    sys.exit(1)

matches = glob.glob(os.path.join(ckpt_dir, f"checkpoint_*_{best_update}.pt"))
if not matches:
    print(f"ERROR: checkpoint for update {best_update} not found", file=sys.stderr)
    sys.exit(1)
ckpt = matches[0]

print(f"[best] valid_accuracy={best_acc:.4f}  update={best_update}  ckpt={ckpt}", file=sys.stderr)
print(ckpt)
EOF
    )
    [ $? -ne 0 ] && exit 1
    echo "=== [${noise_mode}] Using checkpoint: ${ckpt} ==="
fi

subsets=("test" "test1" "test2" "test3" "test4")

for subset in "${subsets[@]}"; do
    echo "=== Inferring: ${subset} ==="
    python -B infer_s2s.py \
        --config-dir ./conf/ \
        --config-name ${conf_name} \
        dataset.gen_subset=${subset} \
        common_eval.path=${ckpt} \
        common_eval.results_path=${result}/s2s/${subset} \
        override.modalities=['audio','video'] \
        common.user_dir=`pwd` \
        override.noise_prob=0.0 \
        distributed_training.distributed_world_size=1 \

done

# 각 subset의 WER / Accuracy / MoE E[X] 수집 → total_wer.txt
total_wer_file="${result}/s2s/total_wer.txt"
> "${total_wer_file}"

sum=0
count=0

for subset in "${subsets[@]}"; do
    wer_file=$(ls "${result}/s2s/${subset}/wer."* 2>/dev/null | head -1)
    if [ -n "${wer_file}" ]; then
        wer_value=$(grep "^WER:" "${wer_file}" | awk '{print $2}')
        acc=$(echo "scale=2; 100 - ${wer_value}" | bc)
        moe_ex=$(grep "MoE E\[X\]" "${wer_file}" | awk '{print $NF}')
        line="${subset}: WER: ${wer_value}  Acc: ${acc}%"
        [ -n "${moe_ex}" ] && line="${line}  MoE_EX: ${moe_ex}"
        echo "${line}" >> "${total_wer_file}"
        sum=$(echo "${sum} + ${wer_value}" | bc)
        count=$((count + 1))
    else
        echo "${subset}: WER file not found" >> "${total_wer_file}"
    fi
done

if [ ${count} -gt 0 ]; then
    avg_wer=$(echo "scale=2; ${sum} / ${count}" | bc)
    avg_acc=$(echo "scale=2; 100 - ${avg_wer}" | bc)
    echo "------------------------" >> "${total_wer_file}"
    echo "Average WER: ${avg_wer}  Average Acc: ${avg_acc}%" >> "${total_wer_file}"
fi

echo ""
echo "=== Total WER ==="
cat "${total_wer_file}"

# 실행 요약 저장 (매 실행마다 새 파일)
ckpt_tag=$(basename "${ckpt}" .pt | sed 's/checkpoint_[0-9]*_//' | sed 's/checkpoint_//')
summary_file="${result}/s2s/${noise_mode}_${ckpt_tag}.txt"

{
    echo "noise_mode : ${noise_mode}"
    echo "finished   : $(date "+%Y-%m-%d %H:%M:%S")"
    echo "checkpoint : ${ckpt}"
    echo "------------------------"
    cat "${total_wer_file}"
} > "${summary_file}"

echo ""
echo "=== Summary saved: ${summary_file} ==="
ㅁ