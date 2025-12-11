import os, sys
import safetensors
import safetensors.torch


def main():
    """
    Merge multiple safetensors files into one.
    """
    assert len(sys.argv) == 2, "Usage: python3 merge_safetensors.py <ROOT>"
    ROOT = sys.argv[1]

    safetensor_file_paths = [os.path.join(ROOT, f) for f in os.listdir(ROOT) if f.endswith(".safetensors")]
    if len(safetensor_file_paths) == 1:
        return

    tensors = {}
    for path in safetensor_file_paths:
        with safetensors.safe_open(path, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    safetensors.torch.save_file(tensors, os.path.join(ROOT, os.path.basename(safetensor_file_paths[0]).split("-")[0]) + ".safetensors")

    for f in os.listdir(ROOT):
        path = os.path.join(ROOT, f)
        if path.endswith(".index.json"):
            os.remove(path)
        if path in safetensor_file_paths:
            os.remove(path)


if __name__ == "__main__":
    main()
