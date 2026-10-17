#!/usr/bin/env python3
"""Export a trained state dict (nets/NAME.pt) to the NumPy weights the engine loads (nets/NAME.npz).

usage: scripts/export_net.py nets/NAME.pt [OUT.npz]
"""

import os
import sys

import numpy as np
import torch


def main():
  src = sys.argv[1]
  out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".npz"
  sd = torch.load(src, map_location="cpu")
  np.savez(
    out,
    w1=sd["inputs.weight"].numpy(),
    b1=sd["input_bias"].numpy(),
    w2=sd["l2.weight"].numpy(),
    b2=sd["l2.bias"].numpy(),
    w3=sd["out.weight"].numpy()[0],
    b3=sd["out.bias"].numpy()[0],
  )
  print(out)


if __name__ == "__main__":
  main()
