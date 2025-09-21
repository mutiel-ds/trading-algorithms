import numpy as np
from pandas import DataFrame, Series

from typing import List, Dict, Any, Optional, cast

from utils.logging import get_logger, Logger

logger: Logger = get_logger()

class FeatureEngineer:
    """
    Feature engineering class for stock price prediction using machine learning.
    
    This class provides comprehensive feature engineering capabilities including:
    - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
    - Volatility and return metrics
    - Momentum and trend indicators
    - Volume-based features
    - Statistical features
    - Candlestick patterns
    """
    
    df: DataFrame
    
    def __init__(self, df: DataFrame) -> None:
        """
        Initialize FeatureEngineer with stock data.
        
        Args:
            df: DataFrame with OHLCV data (Open, High, Low, Close, Volume)
        """
        self.df = df.copy()
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate that required columns exist."""
        required_columns: List[str] = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns: List[str] = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(msg="FeatureEngineer initialized successfully")
    
    # ==================== BASIC TECHNICAL INDICATORS ====================
    
    def add_simple_moving_averages(self, windows: List[int] = [5, 10, 20, 50, 200]) -> None:
        """
        Add Simple Moving Averages (SMA) for different time windows.
        
        Args:
            windows: List of window sizes for SMA calculation
        """
        logger.info(msg=f"Adding SMA indicators for windows: {windows}")
        
        for window in windows:
            self.df[f'SMA_{window}'] = self.df['Close'].rolling(window=window).mean()
            
            # SMA-based signals
            self.df[f'SMA_{window}_signal'] = np.where(
                self.df['Close'] > self.df[f'SMA_{window}'], 1, 0
            )
            
            # Price relative to SMA
            self.df[f'Close_SMA_{window}_ratio'] = self.df['Close'] / self.df[f'SMA_{window}']
    
    def add_exponential_moving_averages(self, windows: List[int] = [12, 26, 50]) -> None:
        """
        Add Exponential Moving Averages (EMA) for different time windows.
        
        Args:
            windows: List of window sizes for EMA calculation
        """
        logger.info(msg=f"Adding EMA indicators for windows: {windows}")
        
        for window in windows:
            self.df[f'EMA_{window}'] = self.df['Close'].ewm(span=window).mean()
            
            # EMA-based signals
            self.df[f'EMA_{window}_signal'] = np.where(
                self.df['Close'] > self.df[f'EMA_{window}'], 1, 0
            )
            
            # Price relative to EMA
            self.df[f'Close_EMA_{window}_ratio'] = self.df['Close'] / self.df[f'EMA_{window}']
    
    def add_rsi(self, window: int = 14) -> None:
        """
        Add Relative Strength Index (RSI) indicator.
        
        Args:
            window: Period for RSI calculation (default: 14)
        """
        logger.info(msg=f"Adding RSI indicator with window: {window}")
        
        # Calculate price changes
        delta: Series = cast(Series, self.df['Close'].diff())
        
        # Separate gains and losses
        gains: Series = cast(Series, delta.where(delta > 0, 0))
        losses: Series = cast(Series, -delta.where(delta < 0, 0))
        
        # Calculate average gains and losses
        avg_gains: Series = cast(Series, gains.rolling(window=window).mean())
        avg_losses: Series = cast(Series, losses.rolling(window=window).mean())
        
        # Calculate RSI
        rs: Series = cast(Series, avg_gains / avg_losses)
        rsi: Series = cast(Series, 100 - (100 / (1 + rs)))
        
        self.df[f'RSI_{window}'] = rsi
        
        # RSI-based signals
        self.df[f'RSI_{window}_oversold'] = np.where(rsi < 30, 1, 0)  # Oversold
        self.df[f'RSI_{window}_overbought'] = np.where(rsi > 70, 1, 0)  # Overbought
        self.df[f'RSI_{window}_neutral'] = np.where((rsi >= 30) & (rsi <= 70), 1, 0)  # Neutral
    
    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        """
        Add MACD (Moving Average Convergence Divergence) indicator.
        
        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
        """
        logger.info(msg=f"Adding MACD indicator: fast={fast}, slow={slow}, signal={signal}")
        
        # Calculate EMAs
        ema_fast: Series = cast(Series, self.df['Close'].ewm(span=fast).mean())
        ema_slow: Series = cast(Series, self.df['Close'].ewm(span=slow).mean())
        
        # MACD line
        macd_line: Series = cast(Series, ema_fast - ema_slow)
        
        # Signal line
        signal_line: Series = cast(Series, macd_line.ewm(span=signal).mean())
        
        # Histogram
        histogram: Series = macd_line - signal_line
        
        self.df[f'MACD_{fast}_{slow}'] = macd_line
        self.df[f'MACD_signal_{signal}'] = signal_line
        self.df[f'MACD_histogram'] = histogram
        
        # MACD signals
        self.df[f'MACD_bullish'] = np.where(macd_line > signal_line, 1, 0)
        self.df[f'MACD_bearish'] = np.where(macd_line < signal_line, 1, 0)
    
    def add_bollinger_bands(self, window: int = 20, std_dev: float = 2.0) -> None:
        """
        Add Bollinger Bands indicator.
        
        Args:
            window: Period for moving average calculation
            std_dev: Standard deviation multiplier
        """
        logger.info(msg=f"Adding Bollinger Bands: window={window}, std_dev={std_dev}")
        
        # Calculate SMA and standard deviation
        sma: Series = cast(Series, self.df['Close'].rolling(window=window).mean())
        std: Series = cast(Series, self.df['Close'].rolling(window=window).std())
        
        # Calculate bands
        upper_band: Series = cast(Series, sma + (std * std_dev))
        lower_band: Series = cast(Series, sma - (std * std_dev))
        
        self.df[f'BB_upper_{window}'] = upper_band
        self.df[f'BB_middle_{window}'] = sma
        self.df[f'BB_lower_{window}'] = lower_band
        
        # Bollinger Band features
        self.df[f'BB_width_{window}'] = (upper_band - lower_band) / sma
        self.df[f'BB_position_{window}'] = (self.df['Close'] - lower_band) / (upper_band - lower_band)
        
        # Bollinger Band signals
        self.df[f'BB_upper_touch_{window}'] = np.where(self.df['Close'] >= upper_band, 1, 0)
        self.df[f'BB_lower_touch_{window}'] = np.where(self.df['Close'] <= lower_band, 1, 0)
        self.df[f'BB_squeeze_{window}'] = np.where(self.df[f'BB_width_{window}'] < 0.1, 1, 0)
    
    # ==================== VOLATILITY AND RETURNS ====================
    
    def add_returns(self, windows: List[int] = [1, 5, 10, 20]) -> None:
        """
        Add various return calculations.
        
        Args:
            windows: List of periods for return calculation
        """
        logger.info(msg=f"Adding returns for windows: {windows}")
        
        for window in windows:
            # Simple returns
            self.df[f'return_{window}d'] = self.df['Close'].pct_change(periods=window)
            
            # Log returns
            self.df[f'log_return_{window}d'] = np.log(self.df['Close'] / self.df['Close'].shift(window))
            
            # Cumulative returns
            self.df[f'cumulative_return_{window}d'] = (1 + self.df[f'return_{window}d']).cumprod() - 1
    
    def add_volatility(self, windows: List[int] = [5, 10, 20, 30]) -> None:
        """
        Add volatility measures.
        
        Args:
            windows: List of periods for volatility calculation
        """
        logger.info(msg=f"Adding volatility measures for windows: {windows}")
        
        for window in windows:
            # Rolling volatility (standard deviation of returns)
            returns: Series = cast(Series, self.df['Close'].pct_change())
            self.df[f'volatility_{window}d'] = cast(Series, returns.rolling(window=window).std())
            
            # Annualized volatility
            self.df[f'volatility_annualized_{window}d'] = self.df[f'volatility_{window}d'] * np.sqrt(252)
            
            # Volatility ratio (current vs historical)
            if window > 5:
                self.df[f'volatility_ratio_{window}d'] = (
                    self.df[f'volatility_{window}d'] / 
                    self.df[f'volatility_{window}d'].rolling(window=window*2).mean()
                )
    
    def add_price_features(self) -> None:
        """Add basic price-based features."""
        logger.info(msg="Adding basic price features")
        
        # Price ranges
        self.df['daily_range'] = self.df['High'] - self.df['Low']
        self.df['daily_range_pct'] = self.df['daily_range'] / self.df['Close']
        
        # Price position within daily range
        self.df['close_position'] = (self.df['Close'] - self.df['Low']) / self.df['daily_range']
        
        # Gap analysis
        self.df['gap_up'] = np.where(self.df['Open'] > self.df['Close'].shift(1), 1, 0)
        self.df['gap_down'] = np.where(self.df['Open'] < self.df['Close'].shift(1), 1, 0)
        self.df['gap_size'] = (self.df['Open'] - self.df['Close'].shift(1)) / self.df['Close'].shift(1)
        
        # Intraday momentum
        self.df['intraday_momentum'] = (self.df['Close'] - self.df['Open']) / self.df['Open']
    
    # ==================== MOMENTUM AND TREND INDICATORS ====================
    
    def add_momentum_indicators(self, windows: List[int] = [5, 10, 20]) -> None:
        """
        Add momentum indicators.
        
        Args:
            windows: List of periods for momentum calculation
        """
        logger.info(msg=f"Adding momentum indicators for windows: {windows}")
        
        for window in windows:
            # Rate of Change (ROC)
            self.df[f'ROC_{window}d'] = (
                (self.df['Close'] - self.df['Close'].shift(window)) / 
                self.df['Close'].shift(window) * 100
            )
            
            # Momentum
            self.df[f'momentum_{window}d'] = self.df['Close'] - self.df['Close'].shift(window)
            
            # Price acceleration
            if window > 1:
                self.df[f'acceleration_{window}d'] = (
                    self.df[f'momentum_{window}d'] - 
                    self.df[f'momentum_{window}d'].shift(1)
                )
    
    def add_trend_indicators(self) -> None:
        """Add trend identification indicators."""
        logger.info(msg="Adding trend indicators")
        
        # Trend strength (ADX-like calculation)
        high_low: Series = cast(Series, self.df['High'] - self.df['Low'])
        high_close: Series = cast(Series, np.abs(self.df['High'] - self.df['Close'].shift(1)))
        low_close: Series = cast(Series, np.abs(self.df['Low'] - self.df['Close'].shift(1)))
        
        true_range: Series = cast(Series, np.maximum(high_low, np.maximum(high_close, low_close)))
        self.df['true_range'] = true_range
        
        # Average True Range (ATR)
        self.df['ATR_14'] = cast(Series, true_range.rolling(window=14).mean())
        
        # Trend direction
        self.df['trend_up'] = np.where(self.df['Close'] > self.df['Close'].shift(1), 1, 0)
        self.df['trend_down'] = np.where(self.df['Close'] < self.df['Close'].shift(1), 1, 0)
        
        # Consecutive trend days
        self.df['consecutive_up_days'] = (
            self.df['trend_up'].groupby((self.df['trend_up'] != self.df['trend_up'].shift()).cumsum()).cumsum()
        )
        self.df['consecutive_down_days'] = (
            self.df['trend_down'].groupby((self.df['trend_down'] != self.df['trend_down'].shift()).cumsum()).cumsum()
        )
    
    # ==================== VOLUME-BASED FEATURES ====================
    
    def add_volume_features(self, windows: List[int] = [5, 10, 20]) -> None:
        """
        Add volume-based features.
        
        Args:
            windows: List of periods for volume calculations
        """
        logger.info(msg=f"Adding volume features for windows: {windows}")
        
        # Volume moving averages
        for window in windows:
            self.df[f'volume_sma_{window}'] = self.df['Volume'].rolling(window=window).mean()
            self.df[f'volume_ratio_{window}'] = self.df['Volume'] / self.df[f'volume_sma_{window}']
        
        # Volume-price relationship
        self.df['volume_price_trend'] = self.df['Volume'] * self.df['Close'].pct_change()
        
        # On-Balance Volume (OBV)
        obv: Series = cast(Series, np.where(
            self.df['Close'] > self.df['Close'].shift(1),
            self.df['Volume'],
            np.where(
                self.df['Close'] < self.df['Close'].shift(1),
                -self.df['Volume'],
                0
            )
        ).cumsum())
        self.df['OBV'] = obv
        
        # Volume-weighted average price (VWAP)
        self.df['VWAP'] = cast(Series, (self.df['Volume'] * (self.df['High'] + self.df['Low'] + self.df['Close']) / 3).cumsum() / self.df['Volume'].cumsum())
        
        # Price relative to VWAP
        self.df['price_vwap_ratio'] = self.df['Close'] / self.df['VWAP']
    
    # ==================== STATISTICAL FEATURES ====================
    
    def add_statistical_features(self, windows: List[int] = [5, 10, 20]) -> None:
        """
        Add statistical features.
        
        Args:
            windows: List of periods for statistical calculations
        """
        logger.info(msg=f"Adding statistical features for windows: {windows}")
        
        for window in windows:
            # Skewness and Kurtosis
            self.df[f'skewness_{window}d'] = self.df['Close'].rolling(window=window).skew()
            self.df[f'kurtosis_{window}d'] = self.df['Close'].rolling(window=window).kurt()
            
            # Percentile ranks
            self.df[f'close_percentile_{window}d'] = (
                self.df['Close'].rolling(window=window).rank(pct=True)
            )
            
            # Z-score
            rolling_mean: Series = cast(Series, self.df['Close'].rolling(window=window).mean())
            rolling_std: Series = cast(Series, self.df['Close'].rolling(window=window).std())
            self.df[f'z_score_{window}d'] = cast(Series, (self.df['Close'] - rolling_mean) / rolling_std)
    
    # ==================== CANDLESTICK PATTERNS ====================
    
    def add_candlestick_features(self) -> None:
        """Add basic candlestick pattern features."""
        logger.info(msg="Adding candlestick pattern features")
        
        # Body and shadow sizes
        body_size: Series = cast(Series, np.abs(self.df['Close'] - self.df['Open']))
        upper_shadow: Series = cast(Series, self.df['High'] - np.maximum(self.df['Open'], self.df['Close']))
        lower_shadow: Series = cast(Series, np.minimum(self.df['Open'], self.df['Close']) - self.df['Low'])
        
        self.df['body_size'] = body_size
        self.df['upper_shadow'] = upper_shadow
        self.df['lower_shadow'] = lower_shadow
        
        # Candlestick patterns
        self.df['doji'] = np.where(body_size < (self.df['High'] - self.df['Low']) * 0.1, 1, 0)
        self.df['hammer'] = np.where(
            (lower_shadow > body_size * 2) & (upper_shadow < body_size * 0.5), 1, 0
        )
        self.df['shooting_star'] = np.where(
            (upper_shadow > body_size * 2) & (lower_shadow < body_size * 0.5), 1, 0
        )
        
        # Bullish/Bearish candles
        self.df['bullish_candle'] = np.where(self.df['Close'] > self.df['Open'], 1, 0)
        self.df['bearish_candle'] = np.where(self.df['Close'] < self.df['Open'], 1, 0)
    
    # ==================== AUTOMATIC FEATURE GENERATION ====================
    
    def generate_all_features(
        self,
        sma_windows: List[int] = [5, 10, 20, 50],
        ema_windows: List[int] = [12, 26, 50],
        return_windows: List[int] = [1, 5, 10, 20],
        volatility_windows: List[int] = [5, 10, 20, 30],
        volume_windows: List[int] = [5, 10, 20],
        momentum_windows: List[int] = [5, 10, 20],
        statistical_windows: List[int] = [5, 10, 20],
        include_candlestick: bool = True,
        return_df: bool = False, 
    ) -> Optional[DataFrame]:
        """
        Generate all available features automatically.
        
        Args:
            sma_windows: Windows for SMA calculations
            ema_windows: Windows for EMA calculations
            return_windows: Windows for return calculations
            volatility_windows: Windows for volatility calculations
            volume_windows: Windows for volume calculations
            momentum_windows: Windows for momentum calculations
            statistical_windows: Windows for statistical calculations
            include_candlestick: Whether to include candlestick patterns
            return_df: Whether to return the processed DataFrame.
            
        Returns:
            DataFrame with all features added
        """
        logger.info(msg="Generating all features automatically")
        
        # Basic technical indicators
        self.add_simple_moving_averages(windows=sma_windows)
        self.add_exponential_moving_averages(windows=ema_windows)
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        
        # Volatility and returns
        self.add_returns(windows=return_windows)
        self.add_volatility(windows=volatility_windows)
        self.add_price_features()
        
        # Momentum and trend
        self.add_momentum_indicators(windows=momentum_windows)
        self.add_trend_indicators()
        
        # Volume features
        self.add_volume_features(windows=volume_windows)
        
        # Statistical features
        self.add_statistical_features(windows=statistical_windows)
        
        # Candlestick patterns
        if include_candlestick:
            self.add_candlestick_features()
        
        logger.info(msg=f"Feature generation complete. Total features: {len(self.df.columns)}")
        
        if return_df:
            return self.df
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all generated features.
        
        Returns:
            Dictionary with feature summary information
        """
        feature_categories: Dict[str, List[str]] = {
            'Technical Indicators': [],
            'Returns & Volatility': [],
            'Momentum & Trend': [],
            'Volume Features': [],
            'Statistical Features': [],
            'Candlestick Patterns': [],
            'Price Features': []
        }
        
        for col in self.df.columns:
            if col.startswith(('SMA_', 'EMA_', 'RSI_', 'MACD_', 'BB_')):
                feature_categories['Technical Indicators'].append(col)
            elif col.startswith(('return_', 'log_return_', 'volatility_', 'cumulative_return_')):
                feature_categories['Returns & Volatility'].append(col)
            elif col.startswith(('ROC_', 'momentum_', 'acceleration_', 'trend_', 'consecutive_', 'ATR_')):
                feature_categories['Momentum & Trend'].append(col)
            elif col.startswith(('volume_', 'OBV', 'VWAP')):
                feature_categories['Volume Features'].append(col)
            elif col.startswith(('skewness_', 'kurtosis_', 'percentile_', 'z_score_')):
                feature_categories['Statistical Features'].append(col)
            elif col in ['doji', 'hammer', 'shooting_star', 'bullish_candle', 'bearish_candle', 'body_size', 'upper_shadow', 'lower_shadow']:
                feature_categories['Candlestick Patterns'].append(col)
            elif col in ['daily_range', 'close_position', 'gap_', 'intraday_momentum']:
                feature_categories['Price Features'].append(col)
        
        return {
            'total_features': len(self.df.columns),
            'feature_categories': feature_categories,
            'data_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum()
        }
