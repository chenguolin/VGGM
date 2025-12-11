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

cd $ROOT/projects/VGGM
pip3 install -r settings/requirements.txt && pip3 install -U peft transformers
cd extensions/Depth-Anything-3 && pip3 install -e . && cd ../..

sudo yum install -y mesa-libGL
