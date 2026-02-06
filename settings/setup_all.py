import os
from tqdm import tqdm
from multiprocessing import Pool

import sys; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # for src modules
from src.options import ROOT


if __name__ == "__main__":
    ips = os.environ["NODE_IP_LIST"].split(",")
    def _foo(the_ip):
        ip = the_ip.split(":")[0]
        os.system(
            f"ssh {ip} '" +
            f"conda activate qa && cd {ROOT}/projects/VGGM && bash settings/setup.sh" +
            "'"
        )

    with Pool(len(ips)) as pool:
        r = list(tqdm(pool.imap(_foo, ips), total=len(ips)))
