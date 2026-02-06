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


def _run_on_node(node, command, session_name):
    ip = node.split(":")[0]
    project_dir = f"{ROOT}/projects/VGGM"
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
if tmux has-session -t {shlex.quote(session_name)} 2>/dev/null; then
    tmux kill-session -t {shlex.quote(session_name)}
fi
tmux new-session -d -s {shlex.quote(session_name)} {shlex.quote(command)}
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
    args = parser.parse_args()
    if not args.command:
        parser.error("Please provide the command to run on all nodes.")

    command = shlex.join(args.command)
    session_name = args.session

    ips = os.environ["NODE_IP_LIST"].split(",")

    with Pool(len(ips)) as pool:
        run_fn = partial(_run_on_node, command=command, session_name=session_name)
        results = list(
            tqdm(
                pool.imap(run_fn, ips),
                total=len(ips),
            )
        )

    failed = [x for x in results if x["code"] != 0]
    for item in results:
        if item["code"] == 0:
            print(f"[OK] {item['ip']} -> tmux session '{item['session']}' started")
        else:
            print(f"[FAIL] {item['ip']} (exit={item['code']})")
            if item["stderr"]:
                print(item["stderr"])

    if failed:
        raise SystemExit(1)
