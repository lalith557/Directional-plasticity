"""Dependency-free readers for MNIST IDX files + Split-MNIST protocol.

We deliberately avoid torchvision.datasets so the data path is fully auditable
and works offline.
"""
from __future__ import annotations
import os
import struct
import numpy as np
import torch
from typing import List, Tuple


def _read_idx_images(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad IDX image magic in {path}: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, h, w)


def _read_idx_labels(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad IDX label magic in {path}: {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8)


def _find(dirpath: str, options: List[str]) -> str:
    for name in options:
        p = os.path.join(dirpath, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"none of {options} found in {dirpath}")


def load_mnist(dirpath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = _read_idx_images(_find(dirpath, [
        "train-images-idx3-ubyte", "train-images.idx3-ubyte",
    ]))
    train_y = _read_idx_labels(_find(dirpath, [
        "train-labels-idx1-ubyte", "train-labels.idx1-ubyte",
    ]))
    test_x = _read_idx_images(_find(dirpath, [
        "t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte",
    ]))
    test_y = _read_idx_labels(_find(dirpath, [
        "t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte",
    ]))
    return train_x, train_y, test_x, test_y


# =============================================================================
# CIFAR-10 (Kaggle-style: PNG files + trainLabels.csv)
# =============================================================================

def _read_cifar10_png_dir(png_dir: str, id_to_label: dict, class_to_idx: dict,
                          which_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 PNGs into (N, 32, 32, 3) uint8 + (N,) int64 array."""
    from PIL import Image
    xs = np.empty((len(which_ids), 32, 32, 3), dtype=np.uint8)
    ys = np.empty(len(which_ids), dtype=np.int64)
    for i, img_id in enumerate(which_ids):
        p = os.path.join(png_dir, f"{img_id}.png")
        with Image.open(p) as im:
            xs[i] = np.asarray(im.convert("RGB"))
        ys[i] = class_to_idx[id_to_label[img_id]]
    return xs, ys


def load_cifar10(cifar10_dir: str, cache_npz: bool = True
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 from the Kaggle folder layout (train/ dir + trainLabels.csv).

    Since Kaggle's test.7z does not include labels, we split the labeled train
    set 45k/5k for train/test (deterministic seed) so we have a labeled test.

    Caches to cifar-10/_cifar10_cache.npz to avoid re-reading 50k PNGs.
    """
    cache = os.path.join(cifar10_dir, "_cifar10_cache.npz")
    if cache_npz and os.path.isfile(cache):
        with np.load(cache) as z:
            return z["tx"], z["ty"], z["vx"], z["vy"]

    labels_csv = os.path.join(cifar10_dir, "trainLabels.csv")
    id_to_label: dict = {}
    with open(labels_csv, "r") as f:
        next(f)  # header
        for line in f:
            row = line.strip().split(",")
            id_to_label[int(row[0])] = row[1]
    classes = sorted(set(id_to_label.values()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    all_ids = sorted(id_to_label.keys())
    rng = np.random.default_rng(20260824)
    perm = rng.permutation(len(all_ids))
    train_ids = [all_ids[i] for i in perm[:45000]]
    test_ids = [all_ids[i] for i in perm[45000:]]

    train_dir = os.path.join(cifar10_dir, "train")
    tx, ty = _read_cifar10_png_dir(train_dir, id_to_label, class_to_idx, train_ids)
    vx, vy = _read_cifar10_png_dir(train_dir, id_to_label, class_to_idx, test_ids)
    if cache_npz:
        np.savez_compressed(cache, tx=tx, ty=ty, vx=vx, vy=vy)
    return tx, ty, vx, vy


def split_cifar10_tasks(
    tx, ty, vx, vy,
    task_class_pairs: Tuple[Tuple[int, int], ...],
) -> List[dict]:
    """Split-CIFAR-10 in the standard 5x2 format. Uses 10-way head across tasks."""
    import torch
    tasks: List[dict] = []
    for pair in task_class_pairs:
        mask_tr = np.isin(ty, pair); mask_te = np.isin(vy, pair)
        xt = tx[mask_tr].astype(np.float32) / 255.0
        yt = ty[mask_tr].astype(np.int64)
        xv = vx[mask_te].astype(np.float32) / 255.0
        yv = vy[mask_te].astype(np.int64)
        # HWC -> CHW
        xt = np.transpose(xt, (0, 3, 1, 2))
        xv = np.transpose(xv, (0, 3, 1, 2))
        tasks.append({
            "train_x": torch.from_numpy(xt),
            "train_y": torch.from_numpy(yt),
            "test_x": torch.from_numpy(xv),
            "test_y": torch.from_numpy(yv),
            "pair": pair,
        })
    return tasks


# =============================================================================
# CIFAR-100 (folder-per-class layout: train/<class>/*.png)
# =============================================================================

def load_cifar100(cifar100_dir: str, cache_npz: bool = True
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-100 with classes in the order given by label_names.txt.

    Caches to _cifar100_cache.npz.
    """
    cache = os.path.join(cifar100_dir, "_cifar100_cache.npz")
    if cache_npz and os.path.isfile(cache):
        with np.load(cache) as z:
            return z["tx"], z["ty"], z["vx"], z["vy"]

    from PIL import Image
    labels_path = os.path.join(cifar100_dir, "label_names.txt")
    with open(labels_path, "r") as f:
        classes = [ln.strip() for ln in f if ln.strip()]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    def _read(split: str):
        split_dir = os.path.join(cifar100_dir, split)
        xs, ys = [], []
        for cls in classes:
            d = os.path.join(split_dir, cls)
            for fn in os.listdir(d):
                if not fn.endswith(".png"): continue
                with Image.open(os.path.join(d, fn)) as im:
                    xs.append(np.asarray(im.convert("RGB")))
                ys.append(class_to_idx[cls])
        return np.stack(xs, axis=0), np.array(ys, dtype=np.int64)

    tx, ty = _read("train")
    vx, vy = _read("test")
    if cache_npz:
        np.savez_compressed(cache, tx=tx, ty=ty, vx=vx, vy=vy)
    return tx, ty, vx, vy


def split_cifar100_tasks(
    tx, ty, vx, vy,
    task_size: int = 10,
) -> List[dict]:
    """Split-CIFAR-100 in `task_size`-class chunks over class index."""
    import torch
    n_classes = int(max(ty.max(), vy.max())) + 1
    tasks: List[dict] = []
    for start in range(0, n_classes, task_size):
        classes = tuple(range(start, min(start + task_size, n_classes)))
        mask_tr = np.isin(ty, classes); mask_te = np.isin(vy, classes)
        xt = tx[mask_tr].astype(np.float32) / 255.0
        yt = ty[mask_tr].astype(np.int64)
        xv = vx[mask_te].astype(np.float32) / 255.0
        yv = vy[mask_te].astype(np.int64)
        xt = np.transpose(xt, (0, 3, 1, 2))
        xv = np.transpose(xv, (0, 3, 1, 2))
        tasks.append({
            "train_x": torch.from_numpy(xt),
            "train_y": torch.from_numpy(yt),
            "test_x": torch.from_numpy(xv),
            "test_y": torch.from_numpy(yv),
            "classes": classes,
        })
    return tasks


def split_mnist_tasks(
    train_x, train_y, test_x, test_y,
    task_class_pairs: Tuple[Tuple[int, int], ...],
) -> List[dict]:
    """Return list of task dicts: {'train_x','train_y','test_x','test_y'} per pair.

    Labels are remapped to 0/1 within each task so the shared 2-way head is stable.
    """
    tasks: List[dict] = []
    for pair in task_class_pairs:
        mask_train = np.isin(train_y, pair)
        mask_test = np.isin(test_y, pair)
        tx = train_x[mask_train].astype(np.float32) / 255.0
        ty = np.where(train_y[mask_train] == pair[0], 0, 1).astype(np.int64)
        vx = test_x[mask_test].astype(np.float32) / 255.0
        vy = np.where(test_y[mask_test] == pair[0], 0, 1).astype(np.int64)
        tasks.append({
            "train_x": torch.from_numpy(tx.reshape(len(tx), -1)),
            "train_y": torch.from_numpy(ty),
            "test_x": torch.from_numpy(vx.reshape(len(vx), -1)),
            "test_y": torch.from_numpy(vy),
            "pair": pair,
        })
    return tasks
