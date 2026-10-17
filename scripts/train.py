#!/usr/bin/env python3
"""Train the value network on the shards from build_dataset.py.

Streams one 1M-position shard at a time (shard order and positions reshuffled every epoch), so memory
stays at a few hundred MB regardless of dataset size. Loss is MSE between the predicted and target win
probability. Writes nets/<name>.pt (state dict) and nets/<name>.json (settings + per-epoch losses).

usage: scripts/train.py NAME [--data data/processed] [--shards K] [--epochs N] [--decided keep|weight|drop]

For a data-size curve use --shards: it trains on a fixed subset (the first K shards, ~K million positions).
--positions instead caps each epoch, drawing a different sample from all shards every epoch.
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nn.features import CP_CLIP, CP_SCALE  # noqa: E402
from nn.model import ValueNet, to_model_input  # noqa: E402

DECIDED_WEIGHT = 0.1  # loss weight for decided positions with --decided weight


def load_shard(path, decided, cp_scale=CP_SCALE):
  d = np.load(path)
  x, y, cp = d["x"], d["y"].astype(np.float32), d["cp"]
  if cp_scale != CP_SCALE:
    # Rebuild the target from the stored score; a wider scale keeps large advantages apart in probability space.
    y = (1.0 / (1.0 + 10.0 ** (-cp.astype(np.float32) / cp_scale))).astype(np.float32)
  w = np.ones(len(y), np.float32)
  is_decided = np.abs(cp) >= CP_CLIP
  if decided == "drop":
    keep = ~is_decided
    x, y, w = x[keep], y[keep], w[keep]
  elif decided == "weight":
    w[is_decided] = DECIDED_WEIGHT
  return x, y, w


def batches(shards, decided, batch_size, limit, device, shuffle, cp_scale=CP_SCALE):
  """Yield (x, y, w) batches on `device`, stopping after `limit` positions (0 = all)."""
  seen = 0
  order = list(shards)
  if shuffle:
    random.shuffle(order)
  for path in order:
    x, y, w = load_shard(path, decided, cp_scale)
    idx = np.random.permutation(len(y)) if shuffle else np.arange(len(y))
    if limit:
      idx = idx[: max(0, limit - seen)]
    for start in range(0, len(idx), batch_size):
      b = idx[start : start + batch_size]
      yield (
        to_model_input(x[b]).to(device, non_blocking=True),
        torch.from_numpy(y[b]).to(device, non_blocking=True),
        torch.from_numpy(w[b]).to(device, non_blocking=True),
      )
    seen += len(idx)
    if limit and seen >= limit:
      return


def weighted_mse(logits, y, w):
  return (w * (torch.sigmoid(logits) - y) ** 2).sum() / w.sum()


@torch.no_grad()
def evaluate(model, shards, device, batch_size, cp_scale=CP_SCALE):
  """Unweighted MSE on every validation position (decided ones included), with the run's own target scale."""
  model.eval()
  total, count = 0.0, 0
  for x, y, _ in batches(shards, "keep", batch_size, 0, device, shuffle=False, cp_scale=cp_scale):
    total += ((torch.sigmoid(model(x)) - y) ** 2).sum().item()
    count += len(y)
  model.train()
  return total / max(count, 1)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("name")
  ap.add_argument("--data", default="data/processed")
  ap.add_argument("--out", default="nets")
  ap.add_argument("--shards", type=int, default=0, help="train on the first K shards only (0 = all)")
  ap.add_argument("--positions", type=int, default=0, help="training positions per epoch (0 = all)")
  ap.add_argument("--epochs", type=int, default=10)
  ap.add_argument("--batch-size", type=int, default=16384)
  ap.add_argument("--lr", type=float, default=1e-3)
  ap.add_argument("--decided", choices=["keep", "weight", "drop"], default="keep")
  ap.add_argument("--cp-scale", type=float, default=CP_SCALE, help="target = 1 / (1 + 10^(-cp / scale))")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  train_shards = sorted(glob.glob(os.path.join(args.data, "train_*.npz")))
  if args.shards:
    train_shards = train_shards[: args.shards]
  val_shards = sorted(glob.glob(os.path.join(args.data, "val_*.npz")))
  if not train_shards or not val_shards:
    sys.exit(f"no shards in {args.data}; run scripts/build_dataset.py first")

  model = ValueNet().to(device)
  opt = torch.optim.Adam(model.parameters(), lr=args.lr)
  sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 0.5 * (1 + math.cos(math.pi * e / args.epochs)))
  os.makedirs(args.out, exist_ok=True)
  log = {"args": vars(args), "device": device, "params": sum(p.numel() for p in model.parameters()), "epochs": []}
  print(f"{log['params']:,} params on {device}; {len(train_shards)} train shards, {len(val_shards)} val", flush=True)

  best = float("inf")
  for epoch in range(args.epochs):
    start, total, count = time.time(), 0.0, 0
    for x, y, w in batches(
      train_shards, args.decided, args.batch_size, args.positions, device, shuffle=True, cp_scale=args.cp_scale
    ):
      loss = weighted_mse(model(x), y, w)
      opt.zero_grad(set_to_none=True)
      loss.backward()
      opt.step()
      total += loss.item() * len(y)
      count += len(y)
    sched.step()
    val = evaluate(model, val_shards, device, args.batch_size, args.cp_scale)
    secs = time.time() - start
    log["epochs"].append(
      {"epoch": epoch + 1, "train_mse": total / count, "val_mse": val, "positions": count, "secs": secs}
    )
    print(
      f"epoch {epoch + 1}/{args.epochs}: train {total / count:.5f}  val {val:.5f}  "
      f"{count:,} positions in {secs:.0f}s ({count / secs:,.0f}/s)",
      flush=True,
    )
    if val < best:
      best = val
      torch.save(model.state_dict(), os.path.join(args.out, f"{args.name}.pt"))
    log["best_val_mse"] = best
    with open(os.path.join(args.out, f"{args.name}.json"), "w") as f:
      json.dump(log, f, indent=2)


if __name__ == "__main__":
  main()
