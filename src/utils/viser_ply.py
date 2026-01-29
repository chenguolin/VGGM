import argparse
import time
import numpy as np
from plyfile import PlyData
import viser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_file", type=str, help="Path to the input PLY file.")
    args = parser.parse_args()

    ply = PlyData.read(args.ply_file)
    v = ply["vertex"]

    points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    colors = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32) / 255.

    server = viser.ViserServer()
    server.scene.background_color = (1, 1, 1)

    server.add_point_cloud(
        "pc",
        points=points,
        colors=colors,
        point_size=0.003,
    )

    print("Viser running at http://localhost:8080")
    while True:
        time.sleep(1)
