import pytest
import pandas as pd

from racoons.data import DataSet

def test_data_set(tmp_path, classification_data):
    classification_data[0].to_csv(tmp_path / "test_set.csv", index=False)
    dataset = DataSet(tmp_path/"test_set.csv", classification_data[1], classification_data[2])
