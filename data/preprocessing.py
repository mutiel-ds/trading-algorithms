from numpy import ndarray
from pandas import to_datetime, DataFrame, Series, DatetimeIndex, MultiIndex

from typing import List, Literal, Dict, Any, Optional, cast

from utils.logging import get_logger, Logger

logger: Logger = get_logger()

class Preprocessor:
    df: DataFrame

    def __init__(self, df: DataFrame) -> None:
        self.df = df.copy()

    def set_datetime_index(self) -> None:
        """
        Ensure Pandas DataFrame has a Datetime Index.
        """
        if not isinstance(self.df.index, DatetimeIndex):
            logger.info(msg="Setting Datetime index...")
            self.df.index = to_datetime(arg=self.df.index)

    def flatten_columns(self, level: int = 0) -> None:
        """
        Flatten columns for Multi Index DataFrames'.

        Args:
            level: Level at which columns are located.
        """
        if isinstance(self.df.columns, MultiIndex):
            logger.info(msg=f"Flattening level {level} columns for MultiIndex columns...")
            self.df.columns = self.df.columns.get_level_values(level=level)

    def validate_ohlc_data(self) -> None:
        """
        Validates OHLC (Open, High, Low, Close) data for financial consistency.
        """
        logger.info(msg="Validating OHLC data consistency...")
        
        # Check if required columns exist
        ohlc_columns: List[str] = ['Open', 'High', 'Low', 'Close']
        missing_columns: List[str] = [col for col in ohlc_columns if col not in self.df.columns]
        
        if missing_columns:
            logger.warning(msg=f"Missing OHLC columns: {missing_columns}. Skipping OHLC validation.")
            return
        
        initial_rows: int = len(self.df)
        
        # Validate OHLC relationships
        invalid_mask: ndarray = (
            (self.df['High'] < self.df['Low']) |  # High should be >= Low
            (self.df['High'] < self.df['Open']) |  # High should be >= Open
            (self.df['High'] < self.df['Close']) |  # High should be >= Close
            (self.df['Low'] > self.df['Open']) |   # Low should be <= Open
            (self.df['Low'] > self.df['Close'])    # Low should be <= Close
        )
        
        invalid_rows: int = invalid_mask.sum()
        if invalid_rows > 0:
            logger.warning(msg=f"Found {invalid_rows} rows with invalid OHLC relationships. Removing them.")
            self.df = cast(DataFrame, self.df[~invalid_mask].copy())
        
        logger.info(msg=f"OHLC validation complete. Removed {initial_rows - len(self.df)} invalid rows.")

    def validate_volume_data(self) -> None:
        """
        Validates volume data for financial consistency.
        """
        logger.info(msg="Validating volume data...")
        
        if 'Volume' not in self.df.columns:
            logger.warning(msg="Volume column not found. Skipping volume validation.")
            return
        
        initial_rows: int = len(self.df)
        
        # Remove rows with negative volume
        invalid_volume_mask: ndarray = self.df['Volume'] < 0
        invalid_rows: int = invalid_volume_mask.sum()
        
        if invalid_rows > 0:
            logger.warning(msg=f"Found {invalid_rows} rows with negative volume. Removing them.")
            self.df = cast(DataFrame, self.df[~invalid_volume_mask].copy())
        
        logger.info(msg=f"Volume validation complete. Removed {initial_rows - len(self.df)} invalid rows.")

    def detect_outliers(self, threshold: float = 0.3) -> Dict[str, int]:
        """
        Detects outliers in price data based on percentage change threshold.
        
        Args:
            threshold: Maximum allowed percentage change in a single day (default: 30%)
            
        Returns:
            Dictionary with outlier counts per column
        """
        logger.info(msg=f"Detecting outliers with {threshold*100}% threshold...")
        
        outlier_counts: Dict[str, int] = {}
        price_columns: List[str] = ['Open', 'High', 'Low', 'Close']
        
        for col in price_columns:
            if col not in self.df.columns:
                continue
                
            # Calculate percentage change
            pct_change: DataFrame | Series = self.df[col].pct_change().abs()
            
            # Find outliers
            outliers: ndarray = pct_change > threshold
            outlier_count: int = outliers.sum()
            outlier_counts[col] = outlier_count
            
            if outlier_count > 0:
                logger.warning(msg=f"Found {outlier_count} outliers in {col} column")
        
        return outlier_counts

    def remove_outliers(self, threshold: float = 0.3, method: str = 'remove') -> None:
        """
        Removes or handles outliers in price data.
        
        Args:
            threshold: Maximum allowed percentage change in a single day
            method: 'remove' to remove outliers, 'cap' to cap at threshold
        """
        logger.info(msg=f"Handling outliers with {method} method...")
        
        initial_rows: int = len(self.df)
        price_columns: List[str] = ['Open', 'High', 'Low', 'Close']
        
        for col in price_columns:
            if col not in self.df.columns:
                continue
                
            # Calculate percentage change
            pct_change: DataFrame | Series = self.df[col].pct_change().abs()
            outliers: ndarray = pct_change > threshold
            
            if method == 'remove':
                # Remove rows with outliers
                self.df = cast(DataFrame, self.df[~outliers].copy())
            elif method == 'cap':
                # Cap outliers at threshold
                prev_values: DataFrame | Series = self.df[col].shift(1)
                max_change: ndarray = prev_values * threshold
                
                # Cap upward outliers
                upward_outliers: ndarray = (self.df[col] > prev_values + max_change) & outliers
                self.df.loc[upward_outliers, col] = prev_values[upward_outliers] + max_change[upward_outliers]
                
                # Cap downward outliers
                downward_outliers: ndarray = (self.df[col] < prev_values - max_change) & outliers
                self.df.loc[downward_outliers, col] = prev_values[downward_outliers] - max_change[downward_outliers]
        
        removed_rows: int = initial_rows - len(self.df)
        if removed_rows > 0:
            logger.info(msg=f"Outlier handling complete. Removed {removed_rows} rows.")

    def fill_missing_values(
            self,
            value: float = 0.0,
            columns: List[str] = ["Volume"],
            ffill: bool = False,
            strategy: str = 'auto'
        ) -> None:
        """
        Fills missing values in given columns with appropriate strategy.

        Args:
            value: The value to be used for replacing the missing values.
            columns: The list of columns over which the filling is to be done.
            ffill: If True, fills missing values with previous valid observation value.
            strategy: 'auto' uses ffill for prices and value for volume, 'ffill' or 'value'
        """
        if strategy == 'auto':
            # Auto strategy: ffill for prices, value for volume
            price_columns: List[str] = [col for col in columns if col in ['Open', 'High', 'Low', 'Close']]
            volume_columns: List[str] = [col for col in columns if col == 'Volume']
            
            if price_columns:
                logger.info(msg=f"Filling missing values for price columns {price_columns} with forward fill...")
                self.df[price_columns] = self.df[price_columns].ffill()
            
            if volume_columns:
                logger.info(msg=f"Filling missing values for volume columns {volume_columns} with {value}...")
                self.df[volume_columns] = self.df[volume_columns].fillna(value=value)
                
        elif ffill or strategy == 'ffill':
            logger.info(msg=f"Filling missing values for {', '.join(columns)} columns with previous valid observation value...")
            self.df[columns] = self.df[columns].ffill()
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
        Remove rows with negative values in price columns.
        """
        initial_n_rows: int = len(self.df)
        
        # Only check price columns for negative values
        price_columns: List[str] = ['Open', 'High', 'Low', 'Close']
        existing_price_columns: List[str] = [col for col in price_columns if col in self.df.columns]
        
        if existing_price_columns:
            # Create mask for rows with any negative price values
            negative_mask: ndarray = (self.df[existing_price_columns] < 0).any(axis=1)
            self.df = cast(DataFrame, self.df[~negative_mask].copy())
            
            if len(self.df) < initial_n_rows:
                logger.warning(msg=f"Removed {initial_n_rows - len(self.df)} rows with negative price values!")

    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive data quality report.
        
        Returns:
            Dictionary containing data quality metrics
        """
        report: Dict[str, Any] = {
            'total_rows': len(self.df),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'date_range': {
                'start': self.df.index.min() if len(self.df) > 0 else None,
                'end': self.df.index.max() if len(self.df) > 0 else None
            }
        }
        
        # Add price statistics if OHLC columns exist
        price_columns: List[str] = ['Open', 'High', 'Low', 'Close']
        existing_price_columns: List[str] = [col for col in price_columns if col in self.df.columns]
        
        if existing_price_columns:
            report['price_statistics'] = self.df[existing_price_columns].describe().to_dict()
            
            # Check for potential data quality issues
            if 'High' in self.df.columns and 'Low' in self.df.columns:
                invalid_ohlc: int = (self.df['High'] < self.df['Low']).sum()
                report['invalid_ohlc_rows'] = invalid_ohlc
        
        # Add volume statistics if Volume column exists
        if 'Volume' in self.df.columns:
            report['volume_statistics'] = {
                'mean': self.df['Volume'].mean(),
                'median': self.df['Volume'].median(),
                'min': self.df['Volume'].min(),
                'max': self.df['Volume'].max(),
                'zero_volume_days': (self.df['Volume'] == 0).sum()
            }
        
        return report

    def preprocess_stock_data(
        self,
        level: int = 0,
        value: float = 0.0,
        columns: List[str] = ["Volume"],
        ffill: bool = False,
        strategy: Literal["auto", "ffill", "value"] = "auto",
        outlier_threshold: float = 0.3,
        outlier_method: Literal["remove", "cap"] = "remove",
        validate_data: bool = True,
        handle_outliers: bool = True,
        return_df: bool = False 
    ) -> Optional[DataFrame]:
        """
        Full preprocessing for loaded stock data with enhanced validation.

        Args:
            level: Level at which columns are located.
            value: The value to be used for replacing the missing values.
            columns: The list of columns over which the filling is to be done.
            ffill: If True, fills missing values with previous valid observation value.
            strategy: 'auto' uses ffill for prices and value for volume, 'ffill' or 'value'
            outlier_threshold: Maximum allowed percentage change in a single day
            outlier_method: 'remove' to remove outliers, 'cap' to cap at threshold
            validate_data: Whether to perform OHLC and volume validation.
            handle_outliers: Whether to detect and handle outliers.
            return_df: Whether to return the processed DataFrame.
        """
        logger.info(msg="Initiating Enhanced Preprocessing for Stock Data!")
        
        # Basic preprocessing
        self.set_datetime_index()
        self.flatten_columns(level=level)
        
        # Enhanced data validation
        if validate_data:
            self.validate_ohlc_data()
            self.validate_volume_data()
        
        # Handle missing values with improved strategy
        self.fill_missing_values(
            value=value,
            columns=columns,
            ffill=ffill,
            strategy=strategy
        )
        
        # Remove remaining missing values
        self.remove_missing_values()
        
        # Remove negative values (improved)
        self.remove_negative_values()
        
        # Handle outliers
        if handle_outliers:
            self.remove_outliers(
                threshold=outlier_threshold,
                method=outlier_method
            )
        
        # Final sorting
        self.df.sort_index(inplace=True)
        
        # Generate quality report
        quality_report: Dict[str, Any] = self.get_data_quality_report()
        logger.info(msg=f"Data quality report: {quality_report['total_rows']} rows processed")
        
        logger.info(msg="Finished Enhanced Preprocessing for Stock Data!")
        
        if return_df:
            return self.df