import sys

sys.path.append("../")

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple, Union
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transforms.time_series import DataTransform_FD, DataTransform_TD
import torch.fft as fft
from sklearn.model_selection import train_test_split
import gdown


class HarDataset(Dataset):
    def __init__(
        self,
        csv_file_path: Union[Path, str],
        feature_column_prefixes: list = (
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ),
        target_column: str = "standard activity code",
        flatten: bool = False,
        frac: int = 1,
    ):
        """Define the dataloaders for train, validation and test splits for
        HAR datasets. This datasets assumes that data is stored as a single CSV
        file with time series data. Each row is a single sample that can be
        composed of multiple modalities (series). Each column is a feature of
        some series with the prefix indicating the series. The suffix may
        indicates the time step. For instance, if we have two series, accel-x
        and accel-y, the data will look something like:

        +-----------+-----------+-----------+-----------+--------+
        | accel-x-0 | accel-x-1 | accel-y-0 | accel-y-1 |  class |
        +-----------+-----------+-----------+-----------+--------+
        | 0.502123  | 0.02123   | 0.502123  | 0.502123  |  0     |
        | 0.6820123 | 0.02123   | 0.502123  | 0.502123  |  1     |
        | 0.498217  | 0.00001   | 1.414141  | 3.141592  |  2     |
        +-----------+-----------+-----------+-----------+--------+

        The ``feature_column_prefixes`` parameter is used to select the columns
        that will be used as features. For instance, if we want to use only the
        accel-x series, we can set ``feature_prefixes=["accel-x"]``. If we want
        to use both accel-x and accel-y, we can set
        ``feature_prefixes=["accel-x", "accel-y"]``.
        The label column is specified by the ``target_column`` parameter.

        The dataset will return a 2-element tuple with the data and the label.

        If ``flatten`` is ``False``, the data will be returned as a
        vector of shape `(C, T)`, where C is the number of channels (features)
        and `T` is the number of time steps. Else, the data will be returned as
        a vector of shape  T*C (a single vector with all the features).

        Parameters
        ----------
        data_path : PathLike
            The path to the folder with "train.csv", "validation.csv" and
            "test.csv" files inside it.
        feature_column_prefixes : Union[str, List[str]], optional
            The prefix of the column names in the dataframe that will be used
            to become features.
        target_column : str, optional
            The name of the column that will be used as label
        flatten : bool, optional
            If False, the data will be returned as a vector of shape (C, T),
            else the data will be returned as a vector of shape  T*C.

        Examples
        --------
        # supposing our data.csv file contains the following data:

        +-----------+-----------+-----------+-----------+--------+
        | accel-x-0 | accel-x-1 | accel-y-0 | accel-y-1 |  class |
        +-----------+-----------+-----------+-----------+--------+
        | 0.502123  | 0.02123   | 0.502123  | 0.502123  |  0     |
        | 0.6820123 | 0.02123   | 0.502123  | 0.502123  |  1     |
        | 0.498217  | 0.00001   | 1.414141  | 3.141592  |  2     |
        +-----------+-----------+-----------+-----------+--------+

        # Using the data from data.csv, and flatten=True
        >>> data_path = "data.csv"
        >>> dataset = HarDataset(
                data_path,
                feature_prefixes=["accel-x", "accel-y"],
                label="class",
                flatten=True
            )
        >>> data, label = dataset[0]
        >>> data.shape
        (4, )

         # Using the data from data.csv, and flatten=False
        >>> dataset = HarDataset(
                data_path,
                feature_prefixes=["accel-x", "accel-y"],
                label="class",
                flatten=False
            )
        >>> data, label = dataset[0]
        >>> data.shape
        (2, 2)

        # And the dataset length
        >>> len(dataset)
        3
        """
        self.csv_file_path = Path(csv_file_path)
        self.feature_column_prefixes = feature_column_prefixes
        self.target_column = target_column
        self.flatten = flatten
        self.frac = frac
        # Read data
        self.data = pd.read_csv(self.csv_file_path)

        if frac < 1:
            # select a fraction of samples but keep the same distribution of classes
            self.data, _ = train_test_split(self.data, test_size=1 - frac, stratify=self.data[target_column])
        # List of list of columns to select, based on their prefixes
        # Something like [[accel-x-0, accel-x-1], [accel-y-0, accel-y-1], ...]
        self.columns_to_select = [
            self._list_columns_starting_with(self.data, prefix)
            for prefix in self.feature_column_prefixes
        ]

    @staticmethod
    def _list_columns_starting_with(df: pd.DataFrame, prefix: str) -> List[str]:
        return [column for column in df.columns if column.startswith(prefix)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Retuns a 2-element tuple with the data and the label.

        Parameters
        ----------
        idx : int
            index of the sample to be returned

        Returns
        -------
        Tuple[np.ndarray, int]
            The data and the label of the sample at index `idx`
        """

        data = np.zeros(
            shape=(len(self.columns_to_select), len(self.columns_to_select[0])),
            dtype=np.float32,
        )
        for channel in range(len(self.columns_to_select)):
            data[channel, :] = self.data.iloc[idx][self.columns_to_select[channel]]
        if self.flatten:
            data = data.flatten()

        target = self.data.iloc[idx][self.target_column]
        return data, target


class HarDataModule(L.LightningDataModule):
    def __init__(
        self,
        # General DataModule parameters
        root_data_dir: Union[Path, str],
        # Dataset parameters
        feature_column_prefixes: list = (
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ),
        target_column: str = "standard activity code",
        flatten: bool = False,
        # DataLoader parameters
        batch_size: int = 32,
        frac: int = 1,
    ):
        """Encapsulates the data loading and processing for the HAR dataset.
        This class is a subclass of `LightningDataModule` and implements the
        `train_dataloader`, `val_dataloader` and `test_dataloader` methods.

        The dataset is assumed to be stored in a folder with the following
        structure:

            ```
            root_data_dir
            ├── train.csv
            ├── validation.csv
            └── test.csv
            ```

        The data is assumed to be stored in CSV files with the following
        structure:

        +-----------+-----------+-----------+-----------+--------+
        | accel-x-0 | accel-x-1 | accel-y-0 | accel-y-1 |  class |
        +-----------+-----------+-----------+-----------+--------+
        | 0.502123  | 0.02123   | 0.502123  | 0.502123  |  0     |
        | 0.6820123 | 0.02123   | 0.502123  | 0.502123  |  1     |
        | 0.498217  | 0.00001   | 1.414141  | 3.141592  |  2     |
        +-----------+-----------+-----------+-----------+--------+

        `train_dataloader` method will create a `HarDataset` object with the
        `train.csv` file, shuffle the data and return a DataLoader object.

        `val_dataloader` method will create a `HarDataset` object with the
        `validation.csv` file, do not shuffle the data and return a DataLoader

        `test_dataloader` method will create a `HarDataset` object with the
        `test.csv` file, do not shuffle the data and return a DataLoader

        Parameters
        ----------
        root_data_dir : Union[Path, str]
            The path to the folder with "train.csv", "validation.csv" and
            "test.csv" files inside it.
        feature_column_prefixes : list, optional
            The prefix of the column names in the dataframe that will be used
            to become features. This parameter is used for instantiating the
            `HarDataset` object.
        target_column : str, optional
            The name of the column that will be used as label. This parameter
            is used for instantiating the `HarDataset` object.
        flatten : bool, optional
            If False, the data will be returned as a vector of shape (C, T),
            else the data will be returned as a vector of shape  T*C. This
            parameter is used for instantiating the `HarDataset` object.
        batch_size : int, optional
            Number of samples per batch. This parameter is used for
            instantiating the DataLoader objects.
        frac : int, optional
            Fraction of the dataset to be used. If frac < 1, a fraction of the
            samples will be selected but keeping the same distribution of
            classes. This parameter is used for instantiating the DataLoader
            objects.
        """
        super().__init__()
        self.root_data_dir = Path(root_data_dir)
        self.feature_column_prefixes = feature_column_prefixes
        self.target_column = target_column
        self.flatten = flatten
        self.batch_size = batch_size
        self.frac = frac
        self.zip_file = os.path.join(self.root_data_dir, "har.zip")
        self.csv_files = {
            "train": os.path.join(self.root_data_dir, "train.csv"),
            "validation": os.path.join(self.root_data_dir, "validation.csv"),
            "test": os.path.join(self.root_data_dir, "test.csv"),
        }
        self.setup()

    def setup(self, stage: str = None) -> None:
        # Verify that the data is available. If not, fectch and unzip dataset
        for k, v in self.csv_files.items():
            if not os.path.exists(v):
                print(v, "file is missing")
                self.fetch_and_unzip_dataset()

    def fetch_and_unzip_dataset(self) -> None:

        if not os.path.exists(self.root_data_dir):
            print(f"Creating the root data directory: [{self.root_data_dir}]")
            os.makedirs(self.root_data_dir)

        if not os.path.exists(self.zip_file):
            print(f"Could not find the zip file [{self.zip_file}]")
            print(f"Trying to download it.")
            url = "https://www.ic.unicamp.br/~edson/disciplinas/mo436/2024-1s/data/har.zip"
            urllib.request.urlretrieve(url, self.zip_file)

        # extract data
        with zipfile.ZipFile(self.zip_file, "r") as zip_ref:
            zip_ref.extractall(self.root_data_dir)
        print("Data downloaded and extracted")

    def _get_dataset_dataloader(
        self, path: Path, shuffle: bool, frac: int = 1
    ) -> DataLoader[HarDataset]:
        """Create a HarDataset object and return a DataLoader object.

        Parameters
        ----------
        path : Path
            Path to CSV file with the data
        shuffle : bool
            If True, samples will be shuffled by the DataLoader
        frac : int, optional
            Fraction of the dataset to be used. If frac < 1, a fraction of the
            samples will be selected but keeping the same distribution of
            classes.

        Returns
        -------
        DataLoader[HarDataset]
            DataLoader object with the HarDataset of the given CSV file
        """
        dataset = HarDataset(
            path,
            feature_column_prefixes=self.feature_column_prefixes,
            target_column=self.target_column,
            flatten=self.flatten,
            frac = frac,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=1,
        )
        return dataloader

    def train_dataloader(self):
        dataloader = self._get_dataset_dataloader(
            self.root_data_dir / "train.csv", shuffle=True, frac=self.frac
        )
        return dataloader

    def val_dataloader(self):
        dataloader = self._get_dataset_dataloader(
            self.root_data_dir / "validation.csv", shuffle=False
        )
        return dataloader

    def test_dataloader(self):
        dataloader = self._get_dataset_dataloader(
            self.root_data_dir / "test.csv", shuffle=False
        )
        return dataloader


class TFCDataModule(L.LightningDataModule):
    def __init__(self, root_data_dir: Union[Path, str], batch_size: int = 32, drop_last: bool = False, frac: int = 1):
        super().__init__()
        self.root_data_dir = Path(root_data_dir)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.frac = frac
        self.csv_files = {
            "train": os.path.join(self.root_data_dir, "train.csv"),
            "validation": os.path.join(self.root_data_dir, "validation.csv"),
            "test": os.path.join(self.root_data_dir, "test.csv"),
        }
        self.setup()

    def setup(self, stage: str = None) -> None:
        # Verify that the data is available. If not, fectch and unzip dataset
        for k, v in self.csv_files.items():
            if not os.path.exists(v):
                print(v, "file is missing")
                self.fetch_and_unzip_dataset()
    
    def fetch_and_unzip_dataset(self) -> None:

        if not os.path.exists(self.root_data_dir):
            print(f"Creating the root data directory: [{self.root_data_dir}]")
            os.makedirs(self.root_data_dir)

        if not os.path.exists(str(self.root_data_dir) + ".zip"):
            print(f"Could not find the zip file")
            print(f"Trying to download it.")
            url = "https://drive.google.com/file/d/1DCqPFlA27bY19zKXGaZP9Jsu_EnBa2ko/view?usp=sharing"
            url = "https://drive.google.com/uc?id=1DCqPFlA27bY19zKXGaZP9Jsu_EnBa2ko"
            gdown.download(url, str(self.root_data_dir) + ".zip")

        # extract data
        with zipfile.ZipFile(str(self.root_data_dir) + ".zip", "r") as zip_ref:
            zip_ref.extractall(self.root_data_dir)
        print("Data downloaded and extracted")
        
    def _get_dataset_dataloader(self, path: Path, shuffle: bool, frac: int = 1):
        dataset = TFCHarDataset(
            path,
            frac = frac
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=1,
            drop_last=self.drop_last,
        )
        return dataloader

    def train_dataloader(self):
        dataloader = self._get_dataset_dataloader(
            self.root_data_dir / "train.csv", shuffle=True, frac = self.frac
        )
        return dataloader

    def val_dataloader(self):
        dataloader = self._get_dataset_dataloader(
            self.root_data_dir / "validation.csv", shuffle=False
        )
        return dataloader

    def test_dataloader(self):
        dataloader = self._get_dataset_dataloader(
            self.root_data_dir / "test.csv", shuffle=False
        )
        return dataloader


class TFCHarDataset(Dataset):
    def __init__(
        self,
        csv_file_path: Union[Path, str],
        feature_column_prefixes: list = (
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ),
        target_column: str = "standard activity code",
        training_mode: str = "pre_train",
        frac: int = 1,
    ):
        super(TFCHarDataset, self).__init__()
        self.csv_file_path = Path(csv_file_path)
        data_df = pd.read_csv(self.csv_file_path)

        if frac < 1:
            # select a fraction of samples but keep the same distribution of classes
            data_df, _ = train_test_split(data_df, test_size=1 - frac, stratify=data_df[target_column])
        
        self.len = len(data_df)
        self.feature_column_prefixes = feature_column_prefixes
        self.training_mode = training_mode
        self.columns_to_select = [
            self._list_columns_starting_with(data_df, prefix)
            for prefix in self.feature_column_prefixes
        ]

        all_columns = sum(self.columns_to_select, [])
        self.data = data_df[all_columns].values
        self.data = self.data.reshape(
            -1, len(self.columns_to_select), len(self.columns_to_select[0])
        )

        self.x_data = torch.tensor(self.data[:, :, :], dtype=torch.float32)
        self.x_data_f = fft.fft(self.x_data).abs()
        self.y_data = torch.tensor(data_df[target_column].values)

        if training_mode == "pre_train":
            self.aug1 = DataTransform_TD(self.x_data).float()
            self.aug1_f = DataTransform_FD(self.x_data_f)

    @staticmethod
    def _list_columns_starting_with(df: pd.DataFrame, prefix: str) -> List[str]:
        return [column for column in df.columns if column.startswith(prefix)]

    def __getitem__(self, index):
        if self.training_mode == "pre_train":
            return (
                self.x_data[index],
                self.y_data[index],
                self.aug1[index],
                self.x_data_f[index],
                self.aug1_f[index],
            )
        else:
            return (
                self.x_data[index],
                self.y_data[index],
                self.x_data[index],
                self.x_data_f[index],
                self.x_data_f[index],
            )

    def __len__(self):
        return self.len
