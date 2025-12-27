export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"

ROOT_LIST=(
    "/apdcephfs_fsgm/share_303967936/cglin"
    "/apdcephfs_sgfd/share_303967936/cglin"
    "/apdcephfs/share_gz/apdcephfs_fsgm/share_303967936/cglin"
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

cd $ROOT/projects/VGGM

SETUP_FLAG=/tmp/.setup_done
if [ ! -f "$SETUP_FLAG" ]; then
    bash settings/setup.sh
    touch $SETUP_FLAG
fi

# 使用 Socket 网络
export NCCL_NET=Socket # 数据传输协议，如果使用IB网卡协议，则不需要配置
export NCCL_SOCKET_IFNAME=bond1  # 指定的socket协议网口，默认是eth0

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
    src/generate_ode_pairs.py
$@
