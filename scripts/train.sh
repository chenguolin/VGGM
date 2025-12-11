ROOT_LIST=(
    "/apdcephfs_fsgm/share_303967936/cglin"
    "/apdcephfs/share_gz/apdcephfs_fsgm/share_303967936/cglin"
)
ROOT=""
for path in "${ROOT_LIST[@]}"; do
    if [ -d "$path" ]; then
        ROOT="$path"
        break
    fi
done
if [ -z "$ROOT" ]; then
    echo "None of the following roots exist:"
    printf '%s\n' "${ROOT_LIST[@]}"
    exit 1
fi
echo "ROOT = $ROOT"

FILE=$1
CONFIG_FILE=$2
TAG=$3
shift 3  # remove $1~$3 for $@

# export HF_ENDPOINT=https://hf-mirror.com
# export WANDB_BASE_URL=https://api.bandw.top
export HF_HOME=$ROOT/.cache/huggingface
export TORCH_HOME=$ROOT/.cache/torch
export NCCL_DEBUG=VERSION

accelerate launch \
    --num_machines $TAIJI_HOST_NUM \
    --num_processes $TAIJI_NODE_NUM \
    --machine_rank $INDEX \
    --main_process_ip $CHIEF_IP \
    --main_process_port 8081 \
    ${FILE} \
        --config_file ${CONFIG_FILE} \
        --tag ${TAG} \
        --pin_memory \
        --allow_tf32 \
$@
