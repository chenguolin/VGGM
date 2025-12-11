conda activate qa

cd /apdcephfs/share_gz/apdcephfs_fsgm/share_303967936/cglin/projects/VGGM
pip3 install -r settings/requirements.txt
cd extensions/Depth-Anything-3 && pip3 install -e . && cd ../..

sudo yum install mesa-libGL
