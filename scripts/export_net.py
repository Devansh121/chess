#!/usr/bin/env python3
"""Export a trained state dict (nets/NAME.pt) to the NumPy weights the engine loads (nets/NAME.npz).

usage: scripts/export_net.py nets/NAME.pt [OUT.npz]

The training scale (--cp-scale, from nets/NAME.json) is stored with the weights, so the engine converts
the network's logit back to centipawns with the same constant.
"""

import json
import os
import sys

import numpy as np
import torch


def main():
  src = sys.argv[1]
  out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".npz"
  sd = torch.load(src, map_location="cpu")
  meta = os.path.splitext(src)[0] + ".json"
  cp_scale = 400.0
  if os.path.exists(meta):
    with open(meta) as f:
      cp_scale = json.load(f)["args"].get("cp_scale", 400.0)
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
