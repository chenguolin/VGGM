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

# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
# pip install -U xformers==0.0.29.post1
# pip install nvidia-cublas-cu12==12.4.5.8  # https://github.com/InternLM/lmdeploy/issues/3297

cd $ROOT/projects/VGGM
pip3 install -r settings/requirements.txt && pip3 install -U peft transformers
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
cd extensions/Depth-Anything-3 && pip3 install -e . && cd ../..

sudo yum install -y mesa-libGL tmux

touch /tmp/.setup_done
