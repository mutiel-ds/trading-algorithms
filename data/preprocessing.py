import pandas as pd

from typing import List, Optional

from utils.logging import get_logger, Logger

logger: Logger = get_logger()

class Preprocessor:
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def set_datetime_index(self) -> None:
        """
        Ensure Pandas DataFrame has a Datetime Index.
        """
        if not isinstance(self.df.index, pd.DatetimeIndex):
            logger.info(msg="Setting Datetime index...")
            self.df.index = pd.to_datetime(self.df.index)

    def flatten_columns(self, level: int = 0) -> None:
        """
        Flatten columns for Multi Index DataFrames'.

        Args:
            level: Level at which columns are located.
        """
        if isinstance(self.df.columns, pd.MultiIndex):
            logger.info(msg=f"Flattening level {level} columns for MultiIndex columns...")
            self.df.columns = self.df.columns.get_level_values(level=level)

    def fill_missing_values(
            self,
            value: int = 0,
            columns: List[str] = ["Volume"],
            ffill: bool = False
        ) -> None:
        """
        Fills missing values in given columns with the given value.

        Args:
            value: The value to be used for replacing the missing values.
            columns: The list of columns over which the filling is to be done.
            ffill: If True, fills missing values with previous valid observation value. Otherwise, removes the row.
        """
        if ffill:
            logger.info(msg=f"Filling missing values for {', '.join(columns)} columns with previous valid observation value...")
            self.df[columns].ffill()
        else:
            logger.info(msg=f"Filling missing values for {', '.join(columns)} columns with {value}s...")
            self.df[columns] = self.df[columns].fillna(value=value)

    def remove_missing_values(self) -> None:
        """
        Removes rows with missing values.
        """
        initial_n_rows: int = len(self.df)
        self.df.dropna(inplace=True)
        if len(self.df) < initial_n_rows:
            logger.warning(msg=f"Removed {initial_n_rows - len(self.df)} rows with missing values!")

    def remove_negative_values(self) -> None:
        """
        Remove rows with negative values (i.e., Closing price is smaller than 0).
        """
        initial_n_rows: int = len(self.df)
        self.df = pd.DataFrame(data=self.df[self.df >= 0])
        if len(self.df) < initial_n_rows:
            logger.warning(msg=f"Removed {initial_n_rows - len(self.df)} rows with negative values!")

    def preprocess_stock_data(self, **kwargs) -> None:
        """
        Full preprocessing for loaded stock data.
        """
        logger.info(msg="Initiating Preprocessing for Stock Data!")
        self.set_datetime_index()
        self.flatten_columns(level=kwargs.get("level", 0))
        self.fill_missing_values(
            value=kwargs.get("value", 0),
            columns=kwargs.get("columns", ["Volume"]),
            ffill=kwargs.get("ffill", False)
        )
        self.remove_missing_values()
        self.remove_negative_values()
        self.df.sort_index(inplace=True)
        logger.info(msg="Finished Preprocessing for Stock Data!")