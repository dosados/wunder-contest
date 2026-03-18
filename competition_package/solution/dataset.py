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
    """
    Датасет для чтения векторов из parquet файла по батчам (без загрузки всего файла в память).
    Размер батча при чтении обязательно кратен 1000.

    Args:
        file_path (str): путь к parquet файлу
        batch_size (int): размер батча при чтении (кратен 1000)
        columns (list[str], optional): список колонок для загрузки
        dtype (torch.dtype): тип данных для тензора
    """

    def __init__(
        self,
        file_path: str,
        batch_size: int = 1000,
        columns=None,
        dtype=torch.float32,
    ):
        _check_batch_size(batch_size)
        self.file_path = file_path
        self.batch_size = batch_size
        self.columns = columns
        self.dtype = dtype
        self._metadata = None

    def _get_metadata(self):
        if self._metadata is None:
            pf = pq.ParquetFile(self.file_path)
            self._metadata = pf.metadata
        return self._metadata

    def __iter__(self):
        pf = pq.ParquetFile(self.file_path)
        for record_batch in pf.iter_batches(batch_size=self.batch_size, columns=self.columns):
            arrays = [
                record_batch.column(i).to_numpy() for i in range(record_batch.num_columns)
            ]
            data_np = np.column_stack(arrays)
            yield torch.tensor(data_np, dtype=self.dtype)

    def __len__(self):
        """Приблизительное число батчей (по метаданным parquet)."""
        meta = self._get_metadata()
        total_rows = meta.num_rows
        return (total_rows + self.batch_size - 1) // self.batch_size


class ParquetSequenceDataset(Dataset):
    """
    Датасет, отдающий целые последовательности.
    В parquet последовательности уже сгруппированы (одинаковый seq_ix идёт подряд),
    шаги упорядочены — одна последовательность = 1000 подряд идущих строк.
    Чтение через PyArrow, без pandas.
    __getitem__ возвращает (x, y): x (SEQ_LEN, n_features), y (SEQ_LEN, n_targets).
    """

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
        table = pq.read_table(file_path, columns=columns)
        self._table = table
        self._n_sequences = table.num_rows // seq_len

    def __len__(self):
        return self._n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self._table.slice(start, self.seq_len)
        n_f = len(self.feature_columns)
        # chunk has columns in order: feature_columns + target_columns
        np_chunk = np.column_stack([chunk.column(i).to_numpy() for i in range(chunk.num_columns)])
        x = torch.from_numpy(np_chunk[:, :n_f].copy()).to(dtype=self.dtype)
        y = torch.from_numpy(np_chunk[:, n_f:].copy()).to(dtype=self.dtype)
        return x, y


if __name__ == "__main__":
    # Пример: чтение по батчам по 1000 строк
    dataset = ParquetDataset(
        "/home/timofey/Documents/own/wunder-contest/competition_package/datasets/train.parquet", 
        batch_size=100000
        )
    K = 0

    for batch in dataset:
        # batch — тензор [batch_size, feature_dim]
        K += 1
        print(K, "size: ", batch.shape)

