import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, IterableDataset
from constants import MIN_BATCH_MULTIPLE, SEQ_LEN


def _check_batch_size(batch_size: int) -> None:
    if batch_size <= 0 or batch_size % MIN_BATCH_MULTIPLE != 0:
        raise ValueError(
            f"batch_size должен быть положительным и кратным {MIN_BATCH_MULTIPLE}, получено: {batch_size}"
        )


class ParquetDataset(IterableDataset):

    def __init__(
        self, file_path: str, batch_size: int = 1000, columns=None, dtype=torch.float32
    ):
        _check_batch_size(batch_size)
        self.file_path = file_path
        self.batch_size = batch_size
        self.columns = columns
        self.dtype = dtype
        self._metadata = None

    def _get_metadata(self):
        if self._metadata is None:
            self._metadata = pq.ParquetFile(self.file_path).metadata
        return self._metadata

    def __iter__(self):
        pf = pq.ParquetFile(self.file_path)
        for rb in pf.iter_batches(batch_size=self.batch_size, columns=self.columns):
            arrays = [rb.column(i).to_numpy() for i in range(rb.num_columns)]
            yield torch.tensor(np.column_stack(arrays), dtype=self.dtype)

    def __len__(self):
        total_rows = self._get_metadata().num_rows
        return (total_rows + self.batch_size - 1) // self.batch_size


class ParquetSequenceDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        feature_columns: list,
        target_columns: list,
        seq_len: int = SEQ_LEN,
        dtype=torch.float32,
    ):
        self.dtype = dtype
        self.seq_len = seq_len
        self.feature_columns = list(feature_columns)
        self.target_columns = list(target_columns)
        columns = self.feature_columns + self.target_columns
        self._table = pq.read_table(file_path, columns=columns)
        self._n_sequences = self._table.num_rows // seq_len

    def __len__(self):
        return self._n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self._table.slice(start, self.seq_len)
        n_f = len(self.feature_columns)
        np_chunk = np.column_stack(
            [chunk.column(i).to_numpy() for i in range(chunk.num_columns)]
        )
        x = torch.from_numpy(np_chunk[:, :n_f].copy()).to(dtype=self.dtype)
        y = torch.from_numpy(np_chunk[:, n_f:].copy()).to(dtype=self.dtype)
        return (x, y)
