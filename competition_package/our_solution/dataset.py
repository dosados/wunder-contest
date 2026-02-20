import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

class ParquetDataset(Dataset):
    def __init__(self, file_path, columns=None, dtype=torch.float32):
        """
        Dataset для чтения векторов из parquet файла.
        
        Args:
            file_path (str): путь к parquet файлу
            columns (list[str], optional): список колонок, которые нужно загрузить
            dtype (torch.dtype): тип данных для тензора
        """
        self.file_path = file_path
        self.columns = columns
        self.dtype = dtype
        
        # Используем pyarrow для чтения метаданных (чтобы получить количество строк)
        self.table = pq.read_table(file_path, columns=columns)
        self.data = self.table.to_pandas().values  # конвертируем сразу в numpy array
        self.data = torch.tensor(self.data, dtype=self.dtype)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



if __name__ == "__main__":
    # Пример использования
    dataset = ParquetDataset("data.parquet")
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    for batch in loader:
        # batch — это тензор [batch_size, feature_dim]
        print(batch.shape)