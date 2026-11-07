"""Purged Group Time-Series Split (walk-forward) dengan purge + embargo.

- Group = timestamp (open_time). Semua simbol pada menit yang sama selalu berada
  di fold yang sama, sehingga tidak ada kebocoran lintas simbol.
- Purge: sampel train yang labelnya "melihat" ke dalam periode test dibuang
  (label memakai HORIZON menit ke depan -> buang HORIZON grup terakhir sebelum test).
- Embargo: jeda tambahan setelah purge untuk meredam autokorelasi fitur.
- Tidak pernah ada data masa depan di set train (bukan random K-Fold).
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class PurgedGroupTimeSeriesSplit:
    def __init__(self, n_splits: int = 5, purge: int = 30, embargo: int = 60, min_train_groups: int | None = None):
        if n_splits < 2:
            raise ValueError("n_splits minimal 2")
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo
        self.min_train_groups = min_train_groups

    def split(self, groups: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        groups = np.asarray(groups)
        unique = np.unique(groups)  # terurut naik (waktu)
        n = len(unique)
        fold_size = n // (self.n_splits + 1)
        gap = self.purge + self.embargo
        for k in range(1, self.n_splits + 1):
            test_start = k * fold_size
            test_end = (k + 1) * fold_size if k < self.n_splits else n
            train_end = test_start - gap
            if train_end <= (self.min_train_groups or 1):
                continue
            train_groups = unique[:train_end]
            test_groups = unique[test_start:test_end]
            train_idx = np.flatnonzero(np.isin(groups, train_groups))
            test_idx = np.flatnonzero(np.isin(groups, test_groups))
            yield train_idx, test_idx
