import os
import re
import json
import ast
import pandas as pd
from tqdm import tqdm

from src.options import DATAROOT


################################ Helper Functions Start ################################


CONTROL_CHARS = re.compile(r"[\x00-\x1f]")  # includes \n \r \t etc.

def escape_control_chars(s: str) -> str:
    # 把真正的控制字符替换成 JSON 允许的转义
    # 注意：这里会把真实换行变成 \\n（两个字符），而不是换行
    def repl(m):
        ch = m.group(0)
        if ch == "\n": return r"\n"
        if ch == "\r": return r"\r"
        if ch == "\t": return r"\t"
        if ch == "\b": return r"\b"
        if ch == "\f": return r"\f"
        # 其他控制字符用 \u00XX
        return "\\u%04x" % ord(ch)
    return CONTROL_CHARS.sub(repl, s)


def repair_jsonish(s: str, max_iters=10):
    s = s.strip()
    # 先修控制字符（必须先做）
    s = escape_control_chars(s)

    # 常见 literal 统一
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)

    # 迭代：遇到报错就把附近最可疑的裸引号转义
    for _ in range(max_iters):
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            msg = e.msg
            pos = e.pos

            if "Invalid control character" in msg:
                # 理论上已经处理过；再保险：只替换这一个字符
                bad = s[pos]
                rep = "\\u%04x" % ord(bad)
                s = s[:pos] + rep + s[pos+1:]
                continue

            # 常见：裸引号导致的各种 delimiter/property name 错
            # 在 pos 左右找一个 " 且不是 \\" 的，转义掉
            left = max(0, pos - 200)
            right = min(len(s), pos + 200)
            window = s[left:right]

            # 找离 pos 最近的未转义 "
            candidates = [left + m.start() for m in re.finditer(r'(?<!\\)"', window)]
            if not candidates:
                raise  # 没找到就放弃，让你看到原始报错

            cand = min(candidates, key=lambda x: abs(x - pos))
            s = s[:cand] + r'\"' + s[cand+1:]
            continue

    raise ValueError("repair_jsonish: exceeded max_iters; string likely truncated or severely malformed.")


################################ Helper Functions End ################################


if __name__ == "__main__":
    metadata = pd.read_csv(f"{DATAROOT}/世界模型实验数据.csv")
    metadata = metadata.dropna(subset=["caption_result"])
    metadata = metadata.drop(columns=["video_cos_url", "vipe_url"])
    rows = metadata[["org_raw_id", "caption_result"]].to_dict(orient="records")
    os.makedirs(os.path.join(DATAROOT, "valid_captions"), exist_ok=True)

    num_valid = 0
    for row in tqdm(rows, desc="Processing captions", ncols=125):
        uid = row["org_raw_id"]
        all_captions = ast.literal_eval(row["caption_result"].replace("\\\\\"", "\\\""))
        try:
            all_captions_dict = {}
            for i in range(len(all_captions)):
                clip_idx = int(float(all_captions[i]["index_idx"]))  # start from 1
                caption_result = repair_jsonish(all_captions[i]["caption_result"])
                caption_dict = caption_result[0]["caption"]  # `0`: EN, `1`: ZH
                long_caption = caption_dict["long_caption"]  # use long caption only
                all_captions_dict[clip_idx] = long_caption
            with open(os.path.join(DATAROOT, "valid_captions", f"{uid}.json"), "w", encoding="utf-8") as f:
                json.dump(all_captions_dict, f, ensure_ascii=False, indent=2)
            num_valid += 1
        except:
            continue
    print(f"Successfully processed {num_valid} / {len(rows)} captions.")  # 419344 / 462401
