import os
import shlex
import argparse
import time
from tqdm import tqdm
from multiprocessing import Pool

import sys; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # for src modules
from src.options import ROOT


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
        "--tmux-log-dir",
        type=str,
        default=None,
        help="If set, enable tmux pane logging to this directory on each node",
    )
    args = parser.parse_args()
    if not args.command:
        parser.error("Please provide the command to run on all nodes.")

    command = shlex.join(args.command)
    session_name = args.session
    tmux_log_dir = args.tmux_log_dir
    ips = os.environ["NODE_IP_LIST"].split(",")

    def _foo(the_ip):
        ip = the_ip.split(":")[0]
        remote_cmd = (
            "conda activate qa && "
            f"cd {shlex.quote(ROOT)}/projects/VGGM && "
            f"tmux kill-session -t {shlex.quote(session_name)} 2>/dev/null; "
            f"tmux new-session -d -s {shlex.quote(session_name)} {shlex.quote(command)}"
        )
        if tmux_log_dir:
            remote_log_dir = shlex.quote(tmux_log_dir)
            remote_log_file = shlex.quote(f"{tmux_log_dir}/{session_name}_{ip}.log")
            pipe_cmd = shlex.quote(f"cat >> {remote_log_file}")
            remote_cmd += (
                f" && mkdir -p {remote_log_dir}"
                f" && tmux pipe-pane -t {shlex.quote(session_name)}:0.0 -o {pipe_cmd}"
            )
        os.system(
            f"ssh {shlex.quote(ip)} {shlex.quote(remote_cmd)}"
        )

    with Pool(len(ips)) as pool:
        r = list(tqdm(pool.imap(_foo, ips), total=len(ips)))
