export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"

ROOT_LIST=(
    "/apdcephfs_sgfd/share_303967936/cglin"
    "/apdcephfs/share_sg/apdcephfs_sgfd/share_303967936/cglin"
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
export HF_HOME=$ROOT/.cache/huggingface
export TORCH_HOME=$ROOT/.cache/torch

cd $ROOT/projects/VGGM

SETUP_FLAG=/tmp/.setup_done
if [ ! -f "$SETUP_FLAG" ]; then
    bash settings/setup.sh
    touch $SETUP_FLAG
fi

FILE=$1
CONFIG_FILE=$2
TAG=$3
shift 3  # remove $1~$3 for $@

# NCCL: use RoCE (mlx5) instead of TCP socket
# export NCCL_NET=Socket
# export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8
export NCCL_SOCKET_IFNAME=bond1  # fallback for bootstrap connection only
# Not sure if these are useful:
# export NCCL_IB_GID_INDEX=3  # force RoCE v2 for better congestion control
# export NCCL_NET_GDR_LEVEL=5  # enable GPU Direct RDMA (skip CPU memory copy)
# export NCCL_BUFFSIZE=16777216  # 16MB buffer for large all-gather
# export NCCL_IB_QPS_PER_CONNECTION=4  # more RDMA parallelism per connection
# export NCCL_MIN_NCHANNELS=32  # use more channels to saturate 8 NICs
# export NCCL_CROSS_NIC=0  # keep GPU-NIC affinity, no cross-NIC traffic

# export HF_ENDPOINT=https://hf-mirror.com
# export WANDB_BASE_URL=https://api.bandw.top
export HF_HOME=$ROOT/.cache/huggingface
export TORCH_HOME=$ROOT/.cache/torch
export NCCL_DEBUG=ERROR

torchrun \
  --nnodes $TAIJI_HOST_NUM \
  --nproc_per_node $(( $TAIJI_NODE_NUM / $TAIJI_HOST_NUM )) \
  --node_rank $INDEX \
  --master_addr $CHIEF_IP \
  --master_port 8081 \
    ${FILE} \
        --config_file ${CONFIG_FILE} \
        --tag ${TAG} \
        --allow_tf32 \
$@
