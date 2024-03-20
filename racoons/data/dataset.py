import warnings
import pandas as pd
from racoons.data import utils
from pathlib import Path
from typing import List


class DataSet:
    def __init__(self, dataset_path: Path or str, feature_cols: List[str], target_cols: List[str], scale_levels=None):
        self.dataset_path = Path(dataset_path)
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self._read_dataframe()
        self.ordinal_lookup = {}
        if not scale_levels:
            self.scale_levels_path = self.dataset_path.parent / "scale_levels.csv"
            self.scale_levels = self._read_scale_levels()
        self.dataframe = self._apply_scale_levels()
        self.features = self.dataframe.copy().loc[:, self.feature_cols]
        self.targets = self.dataframe.copy().loc[:, self.target_cols]

    def _read_dataframe(self):
        """Reads the dataframe at the specified path."""
        if self.dataset_path.suffix == ".csv":
            self._raw_dataframe = pd.read_csv(self.dataset_path)
        if self.dataset_path.suffix == ".xlsx":
            self._raw_dataframe = pd.read_excel(self.dataset_path)

    def _read_scale_levels(self):
        """
        Extracts the scale levels from the lookup file.
        If the file is not found, it creates one to be filled manually and exits
        """
        try:
            # load the lookup file and convert it to a dict
            result_dict = {}
            scale_levels = pd.read_csv(self.scale_levels_path)
            for index, row in scale_levels.iterrows():
                if not isinstance(row["Level order (for ordinal values)"], float):
                    result_dict[row["Column"]] = {"scale_level": row["Scale Level"], "level_order": row["Level order (for ordinal values)"].split(" ")}
                else:
                    result_dict[row["Column"]] = {"scale_level": row["Scale Level"],
                                                  "level_order": None}

            return result_dict
        except FileNotFoundError:
            # Create the template and exit.
            template = utils.create_scale_level_template(self._raw_dataframe, self.feature_cols+self.target_cols)
            warnings.warn(f"No lookup file found, creating one to be filled.\n \
            Please fill out the file created at {self.scale_levels_path} and start the module again.")
            template.to_csv(self.scale_levels_path, index=False)
            raise SystemExit(0)

    def _apply_scale_levels(self) -> pd.DataFrame:
        """Applies the scale levels from the lookup dict to the dataframe.
        This results in a new dataframe for actual use.
        """
        df = self._raw_dataframe.copy()
        for column, details in self.scale_levels.items():
            if column in self.target_cols+self.feature_cols:
                if details["scale_level"] == "numerical":
                    df[column] = df[column].astype("float")
                elif details["scale_level"] == "ordinal":
                    if not pd.api.types.is_numeric_dtype(df[column].dtype):
                        df[column] = df[column].apply(lambda x: details["level_order"].index(x))
                    df[column] = df[column].astype("int64")
                elif details["scale_level"] == "categorical":
                    df[column] = df[column].astype("category")

        return df

