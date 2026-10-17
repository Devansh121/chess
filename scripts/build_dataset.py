#!/usr/bin/env python3
"""Turn data/evals.tsv.zst (from fetch_evals.py) into training shards.

Each shard data/processed/{train,val}_NNN.npz holds up to SHARD_SIZE positions:
  x:  uint16 [N, 32]  active feature indices (nn/features.py), padded with 0xFFFF
  y:  float16 [N]     win probability for the side to move
  cp: int16 [N]       clipped centipawn score for the side to move (mates as +/-CP_CLIP)

About 1% of positions go to validation, chosen by a hash of the FEN so the split is stable.

usage: scripts/build_dataset.py [--input data/evals.tsv.zst] [--out data/processed] [--limit N] [--workers N]
"""

import argparse
import os
import subprocess
import sys
import zlib
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nn.features import CP_CLIP, MAX_PIECES, PAD, fen_features, win_probability  # noqa: E402

SHARD_SIZE = 1_000_000
BATCH = 50_000
VAL_PERCENT = 1


def convert_row(line):
  """One TSV row (fen, cp, mate, depth; cp/mate from White's view) -> (is_val, indices, y, cp_stm), or None."""
  fen, cp, mate, _depth = line.rstrip("\n").split("\t")
  white = fen.split()[1] == "w"
  sign = 1 if white else -1
  if mate:
    m = int(mate) * sign
    y, cp_stm = win_probability(mate=m), CP_CLIP if m > 0 else -CP_CLIP
  elif cp:
    c = int(cp) * sign
    y, cp_stm = win_probability(cp=c), max(-CP_CLIP, min(CP_CLIP, c))
  else:
    return None
  indices = fen_features(fen)
  if len(indices) > MAX_PIECES:
    return None
  is_val = zlib.crc32(fen.encode()) % 100 < VAL_PERCENT
  return is_val, indices, y, cp_stm


def convert_batch(lines):
  out = {False: ([], [], []), True: ([], [], [])}
  for line in lines:
    row = convert_row(line)
    if row is None:
      continue
    is_val, indices, y, cp = row
    x = np.full(MAX_PIECES, PAD, dtype=np.uint16)
    x[: len(indices)] = indices
    xs, ys, cps = out[is_val]
    xs.append(x)
    ys.append(y)
    cps.append(cp)
  return {
    k: (np.stack(xs) if xs else np.empty((0, MAX_PIECES), np.uint16), np.array(ys, np.float16), np.array(cps, np.int16))
    for k, (xs, ys, cps) in out.items()
  }


def batches(path, limit):
  proc = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
  batch, n = [], 0
  for line in proc.stdout:
    batch.append(line)
    n += 1
    if len(batch) == BATCH:
      yield batch
      batch = []
    if limit and n >= limit:
      break
  if batch:
    yield batch
  proc.kill()


class ShardWriter:
  def __init__(self, out, name):
    self.out, self.name, self.index = out, name, 0
    self.parts, self.size, self.total = [], 0, 0

  def add(self, x, y, cp):
    if len(y):
      self.parts.append((x, y, cp))
      self.size += len(y)
    while self.size >= SHARD_SIZE:
      self.flush(SHARD_SIZE)

  def flush(self, count=None):
    if not self.size:
      return
    x = np.concatenate([p[0] for p in self.parts])
    y = np.concatenate([p[1] for p in self.parts])
    cp = np.concatenate([p[2] for p in self.parts])
    count = count or len(y)
    np.savez(os.path.join(self.out, f"{self.name}_{self.index:03d}.npz"), x=x[:count], y=y[:count], cp=cp[:count])
    self.index += 1
    self.total += count
    self.parts = [(x[count:], y[count:], cp[count:])] if count < len(y) else []
    self.size = len(y) - count


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--input", default="data/evals.tsv.zst")
  ap.add_argument("--out", default="data/processed")
  ap.add_argument("--limit", type=int, default=0, help="stop after this many input rows (0 = all)")
  ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  train, val = ShardWriter(args.out, "train"), ShardWriter(args.out, "val")
  done = 0
  with Pool(args.workers) as pool:
    for result in pool.imap(convert_batch, batches(args.input, args.limit), chunksize=1):
      train.add(*result[False])
      val.add(*result[True])
      done += len(result[False][1]) + len(result[True][1])
      if done % 1_000_000 < BATCH:
        print(f"{done:,} positions", file=sys.stderr, flush=True)
  train.flush()
  val.flush()
  print(f"done: {train.total:,} train in {train.index} shards, {val.total:,} val in {val.index} shards", flush=True)


if __name__ == "__main__":
  main()
