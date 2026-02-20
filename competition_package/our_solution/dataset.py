import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

# Размер батча при чтении parquet должен быть кратен 1000
MIN_BATCH_MULTIPLE = 1000


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


if __name__ == "__main__":
    # Пример: чтение по батчам по 1000 строк
    dataset = ParquetDataset("data.parquet", batch_size=1000)
    for batch in dataset:
        # batch — тензор [batch_size, feature_dim]
        print(batch.shape)
