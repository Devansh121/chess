#!/usr/bin/env python3
"""Export a trained state dict (nets/NAME.pt) to the NumPy weights the engine loads (nets/NAME.npz).

usage: scripts/export_net.py nets/NAME.pt [OUT.npz | OUT.nnue]

OUT.npz is read by the Python engine; OUT.nnue is the binary format for the C++ engine:
  "CHNN", u32 version (1), u32 arch (1 or 2), u32 inputs (769), u32 hidden1, u32 hidden2, f32 cp_scale,
  then little-endian float32 arrays: w1[inputs][hidden1], b1[hidden1], w2[hidden2][arch * hidden1],
  b2[hidden2], w3[hidden2], b3.

The training scale (--cp-scale, from nets/NAME.json) is stored with the weights, so the engine converts
the network's logit back to centipawns with the same constant.
"""

import json
import os
import struct
import sys

import numpy as np
import torch


def write_binary(path, sd, arch, cp_scale):
  w1 = sd["inputs.weight"].numpy().astype("<f4")
  w2 = sd["l2.weight"].numpy().astype("<f4")
  inputs, hidden1 = w1.shape
  hidden2 = w2.shape[0]
  assert w2.shape[1] == arch * hidden1, (w2.shape, arch, hidden1)
  with open(path, "wb") as f:
    f.write(b"CHNN")
    f.write(struct.pack("<5If", 1, arch, inputs, hidden1, hidden2, cp_scale))
    for _name, arr in (
      ("w1", w1),
      ("b1", sd["input_bias"]),
      ("w2", w2),
      ("b2", sd["l2.bias"]),
      ("w3", sd["out.weight"][0]),
      ("b3", sd["out.bias"]),
    ):
      f.write(np.ascontiguousarray(np.asarray(arr, dtype="<f4")).tobytes())


def main():
  src = sys.argv[1]
  out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".npz"
  sd = torch.load(src, map_location="cpu")
  meta = os.path.splitext(src)[0] + ".json"
  cp_scale, arch = 400.0, "v1"
  if os.path.exists(meta):
    with open(meta) as f:
      run_args = json.load(f)["args"]
    cp_scale = run_args.get("cp_scale", 400.0)
    arch = run_args.get("arch", "v1")
  if out.endswith(".nnue"):
    write_binary(out, sd, int(arch[1:]), cp_scale)
    print(out)
    return
  np.savez(
    out,
    w1=sd["inputs.weight"].numpy(),
    b1=sd["input_bias"].numpy(),
    w2=sd["l2.weight"].numpy(),
    b2=sd["l2.bias"].numpy(),
    w3=sd["out.weight"].numpy()[0],
    b3=sd["out.bias"].numpy()[0],
    cp_scale=cp_scale,
  )
  print(out)


if __name__ == "__main__":
  main()
