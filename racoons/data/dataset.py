import pandas as pd
from racoons.data import utils
from pathlib import Path


class DataSet:
    def __init__(self, dataset_path, features, targets, scale_levels=None):
        self.dataset_path = Path(dataset_path)
        self.features = features
        self.targets = targets
        self._read_dataframe()
        if not scale_levels:
            self.scale_levels = self.dataset_path.parent / "scale_levels.csv"

    def _read_dataframe(self):
        """Reads the dataframe at the specified path."""
        if self.dataset_path.suffix == ".csv":
            self._raw_dataframe = pd.read_csv(self.dataset_path)
        if self.dataset_path.suffix == ".xlsx":
            self._raw_dataframe = pd.read_excel(self.dataset_path)

    def _read_scale_levels(self):
        """
        Extracts the scale levels from the lookup file.
        If the file is not found, it creates one to be filled manually.
        """
        try:
            scale_levels = pd.read_csv(self.scale_levels)
        except FileNotFoundError:
            template = utils.create_scale_level_template(self._raw_dataframe)
