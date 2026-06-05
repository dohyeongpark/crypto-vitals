"""
PyTorch Dataset and DataLoader factory for volatility prediction.

VolatilityDataset keeps only the underlying numpy arrays in memory and
generates each (X, y) sample on demand in __getitem__, avoiding the
~6 GB peak that pre-materializing all windows would require.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .features import FEATURE_COLS
from .windows  import normalize


class VolatilityDataset(Dataset):
    """
    On-demand sliding-window dataset.

    Each call to __getitem__(idx) :
      1. Slices the raw feature array for the window at indices[idx].
      2. Normalizes the window in-place (per-window, no data leakage).
      3. Computes realized volatility over the next `horizon` bars as the label.

    Args:
        values:   float32 array (n_rows, n_features) from the feature DataFrame.
        closes:   float64 array (n_rows,) of close prices for label computation.
        indices:  int64 array of valid window-endpoint positions.
        lookback: Input window length (minutes).
        horizon:  Future window length for realized-vol label (minutes).
    """

    def __init__(
        self,
        values:   np.ndarray,
        closes:   np.ndarray,
        indices:  np.ndarray,
        lookback: int = 360,
        horizon:  int = 30,
    ) -> None:
        self.values   = values
        self.closes   = closes
        self.indices  = indices
        self.lookback = lookback
        self.horizon  = horizon

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i      = int(self.indices[idx])
        window = self.values[i - self.lookback : i]          # (lookback, features)
        X      = torch.from_numpy(normalize(window))         # float32

        # Realized volatility = std of log-returns over the next horizon bars.
        # Log-transformed so the target is closer to normally distributed,
        # which stabilizes MSE training (raw volatility is right-skewed).
        # exp() during evaluation converts predictions back to original scale.
        fut_closes      = self.closes[i - 1 : i + self.horizon]
        fut_log_returns = np.log(fut_closes[1:] / fut_closes[:-1])
        log_vol = np.log(np.std(fut_log_returns) + 1e-8)
        y = torch.tensor(float(log_vol), dtype=torch.float32)

        return X, y


def make_loaders(
    df:          "pd.DataFrame",                             # noqa: F821
    splits:      dict[str, tuple[np.ndarray, np.ndarray]],
    lookback:    int   = 360,
    horizon:     int   = 30,
    batch_size:  int   = 64,
    num_workers: int   = 0,
) -> dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from a feature DataFrame and split indices.

    Args:
        df:          Feature DataFrame from build_features().
        splits:      Output of time_split() — {"train": (indices, ts), ...}.
        lookback:    Window length in minutes.
        horizon:     Label horizon in minutes.
        batch_size:  Samples per batch.
        num_workers: DataLoader worker processes (0 = main process).

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    from .features import FEATURE_COLS          # local import avoids circulars

    values = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    closes = df["close"].to_numpy(dtype=np.float64)

    loaders = {}
    for name, (indices, _) in splits.items():
        dataset = VolatilityDataset(values, closes, indices, lookback, horizon)
        loaders[name] = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = (name == "train"),
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
        )
    return loaders
