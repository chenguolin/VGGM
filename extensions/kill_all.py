import os
from tqdm import tqdm
from multiprocessing import Pool


if __name__ == "__main__":
    ips = os.environ["NODE_IP_LIST"].split(",")

    def _kill(the_ip):
        ip = the_ip.split(":")[0]
        os.system(
            f"ssh {ip} " +
            "\"bash -lc 'pkill -9 -f python || true'\""
        )

    with Pool(len(ips)) as pool:
        list(tqdm(pool.imap(_kill, ips), total=len(ips)))
