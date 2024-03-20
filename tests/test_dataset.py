import numpy as np
import pytest
import pandas as pd

from racoons.data import DataSet


@pytest.fixture
def mock_dataset(tmp_path):
    # Create a mock dataset
    df = pd.DataFrame({
        'Feature1': [1, 2, 3],
        'Feature2': [4, 5, 6],
        'Feature3': [7.94, 8.55, 4.22],
        'Target': [0, 1, 0],
        'Target2': ["Small", "Large", "Medium"],
        'Target3': ['Berlin', 'Frankfurt', 'Paris']
    })
    dataset_path_csv = tmp_path / "dataset.csv"
    df.to_csv(dataset_path_csv, index=False)
    dataset_path_xlsx = tmp_path / "dataset.xlsx"
    df.to_excel(dataset_path_xlsx, index=False)
    return tmp_path


@pytest.fixture
def mock_scale_levels(tmp_path, mock_dataset):
    # Create a mock scale levels CSV file
    df = pd.DataFrame({
        'Column': ['Feature1', 'Feature2', 'Feature3', 'Target', 'Target2', 'Target3'],
        'Scale Level': ['ordinal', 'numerical', 'numerical', 'categorical', 'ordinal', 'categorical'],
        'Level order (for ordinal values)': ["1 2 3", None, None, None, "Small Medium Large", None]
    })
    scale_levels_path = tmp_path / "scale_levels.csv"
    df.to_csv(scale_levels_path, index=False)
    return scale_levels_path


def test_dataset_initialization_csv(mock_dataset, mock_scale_levels):
    # Initialize the dataset
    dataset_csv = DataSet(
        dataset_path=mock_dataset / 'dataset.csv',
        feature_cols=['Feature1', 'Feature3'],
        target_cols=['Target', 'Target2']
    )
    dataset_excel = DataSet(
        dataset_path=mock_dataset / 'dataset.xlsx',
        feature_cols=['Feature1', 'Feature3'],
        target_cols=['Target', 'Target2']
    )
    for dataset in [dataset_csv, dataset_excel]:
        # Assertions to ensure everything was loaded correctly
        assert dataset.scale_levels_path == mock_scale_levels
        assert 'Feature1' in dataset.dataframe.columns
        assert dataset.dataframe['Feature1'].dtype == np.int64
        assert dataset.dataframe['Feature3'].dtype == float
        assert dataset.dataframe['Target'].dtype.name == 'category'
        assert dataset.dataframe['Target2'].dtype == np.int64
        assert dataset.scale_levels['Target2']['scale_level'] == 'ordinal'
        assert dataset.scale_levels['Target2']['level_order'][1] == 'Medium'
        assert dataset.targets.columns.tolist() == ['Target', 'Target2']
        assert dataset.features.columns.tolist() == ['Feature1', 'Feature3']


def test_missing_scale_levels_creates_template(mock_dataset, tmp_path):
    # Make sure the scale_levels.csv file does not exist
    scale_levels_path = tmp_path / "scale_levels.csv"
    if scale_levels_path.exists():
        scale_levels_path.unlink()

    with pytest.raises(SystemExit):
        DataSet(
            dataset_path=mock_dataset / "dataset.csv",
            feature_cols=['Feature1', 'Feature2', 'Feature3'],
            target_cols=['Target', 'Target2', 'Target3'],
            scale_levels=None
        )

    # Check if the template was created
    assert scale_levels_path.exists()

