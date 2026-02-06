import os
import shlex
import subprocess
import time
from functools import partial
from tqdm import tqdm
import argparse
from multiprocessing import Pool

import sys; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # for src modules
from src.options import ROOT


def _run_on_node(node, command, session_name, node_rank, chief_ip, host_num, dry_run=False):
    ip = node.split(":")[0]
    project_dir = f"{ROOT}/projects/VGGM"
    command_with_env = (
        f"export INDEX={node_rank}; "
        f"export CHIEF_IP={shlex.quote(chief_ip)}; "
        f"export TAIJI_HOST_NUM={host_num}; "
        "if [ -z \"$TAIJI_NODE_NUM\" ]; then "
        "  if [ -n \"$NPROC_PER_NODE\" ]; then "
        "    __LOCAL_NPROC=\"$NPROC_PER_NODE\"; "
        "  elif [ -n \"$CUDA_VISIBLE_DEVICES\" ]; then "
        "    __LOCAL_NPROC=$(echo \"$CUDA_VISIBLE_DEVICES\" | awk -F',' '{print NF}'); "
        "  elif command -v nvidia-smi >/dev/null 2>&1; then "
        "    __LOCAL_NPROC=$(nvidia-smi -L | wc -l); "
        "  else "
        "    __LOCAL_NPROC=1; "
        "  fi; "
        "  export TAIJI_NODE_NUM=$((TAIJI_HOST_NUM * __LOCAL_NPROC)); "
        "fi; "
        "echo \"[train_all] ip=$(hostname -I | awk '{print $1}') INDEX=$INDEX "
        "CHIEF_IP=$CHIEF_IP TAIJI_HOST_NUM=$TAIJI_HOST_NUM TAIJI_NODE_NUM=$TAIJI_NODE_NUM\"; "
        f"{command}"
    )
    remote_script = f"""
set -e
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
fi
conda activate qa
cd {shlex.quote(project_dir)}
"""
    if dry_run:
        remote_script += f"""
echo "[dry-run] node={ip} session={session_name}"
echo "[dry-run] command: bash -lc {shlex.quote(command_with_env)}"
"""
    else:
        remote_script += f"""
if tmux has-session -t {shlex.quote(session_name)} 2>/dev/null; then
    tmux kill-session -t {shlex.quote(session_name)}
fi
tmux new-session -d -s {shlex.quote(session_name)} "bash -lc {shlex.quote(command_with_env)}"
tmux set-option -t {shlex.quote(session_name)} mouse on
"""
    remote_cmd = f"bash -lc {shlex.quote(remote_script)}"
    proc = subprocess.run(["ssh", ip, remote_cmd], capture_output=True, text=True)
    return {
        "ip": ip,
        "code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "session": session_name,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run on all nodes, e.g., bash scripts/train.sh src/train_wan_cc.py ...",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=f"train_{time.strftime('%Y%m%d_%H%M%S')}",
        help="Tmux session name used on every node",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print per-node launch commands without creating tmux sessions or starting training.",
    )
    args = parser.parse_args()
    if not args.command:
        parser.error("Please provide the command to run on all nodes.")

    command = shlex.join(args.command)
    session_name = args.session

    ips = [x.strip() for x in os.environ["NODE_IP_LIST"].split(",") if x.strip()]
    if not ips:
        raise SystemExit("NODE_IP_LIST is empty.")
    chief_ip = ips[0].split(":")[0]
    host_num = len(ips)

    task_args = [
        (node, command, session_name, idx, chief_ip, host_num, args.dry_run)
        for idx, node in enumerate(ips)
    ]
    with Pool(len(ips)) as pool:
        results = list(
            tqdm(
                pool.starmap(_run_on_node, task_args),
                total=len(ips),
            )
        )

    failed = [x for x in results if x["code"] != 0]
    for item in results:
        if item["code"] == 0:
            if args.dry_run:
                print(f"[OK] {item['ip']} -> dry run finished")
                if item["stdout"]:
                    print(item["stdout"])
            else:
                print(f"[OK] {item['ip']} -> tmux session '{item['session']}' started")
        else:
            print(f"[FAIL] {item['ip']} (exit={item['code']})")
            if item["stdout"]:
                print(item["stdout"])
            if item["stderr"]:
                print(item["stderr"])

    if failed:
        raise SystemExit(1)
