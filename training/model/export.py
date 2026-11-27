"""Turn a trained .npz into the files of the Hugging Face model repo (hindsight-ai/hs-date-N).

    python -m training.model.export text2date/weights/text2date-d3p.npz hf/hs-date-1

Writes model.safetensors (the weights, plus the config as an int64 tensor), config.json and vocab.json
(the output words, in id order). text2date.runtime reads model.safetensors with numpy alone.
"""

import json
import sys
from pathlib import Path

import numpy as np

from text2date.runtime import MAX_BYTES, MAX_OUT, VOCAB

DTYPE = {np.dtype(np.float16): "F16", np.dtype(np.float32): "F32", np.dtype(np.int64): "I64"}


def write_safetensors(arrays: dict[str, np.ndarray], path: Path) -> None:
    head, blobs, off = {}, [], 0
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        head[k] = {"dtype": DTYPE[a.dtype], "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        blobs.append(a.tobytes())
        off += a.nbytes
    h = json.dumps(head, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)  # the format wants the data to start 8-byte aligned
    path.write_bytes(len(h).to_bytes(8, "little") + h + b"".join(blobs))


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(src)
    arrays = {k: z[k] for k in z.files}
    write_safetensors(arrays, out / "model.safetensors")
    heads, n_enc, n_dec, stride = (int(x) for x in z["config"])
    config = {"model_type": "text2date", "heads": heads, "encoder_layers": n_enc, "decoder_layers": n_dec,
              "stride": stride, "d_model": int(z["tok_emb"].shape[1]), "vocab_size": len(VOCAB),
              "max_input_bytes": MAX_BYTES, "max_output_tokens": MAX_OUT,
              "parameters": int(sum(a.size for k, a in arrays.items() if k != "config"))}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "vocab.json").write_text(json.dumps(VOCAB, ensure_ascii=False) + "\n")
    print(f"{out}: {config['parameters']:,} weights")


if __name__ == "__main__":
    main()
