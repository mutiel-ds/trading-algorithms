import numpy as np
from pandas import Series
from typing import List, Dict, Tuple, Any, Optional

from utils.logging import get_logger, Logger

logger: Logger = get_logger()

class FinancialMetrics:
    """
    Financial-specific metrics for evaluating trading strategies and models.
    
    This class provides comprehensive financial metrics including:
    - Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
    - Drawdown analysis
    - Hit rate and win/loss ratios
    - Volatility metrics
    - Performance attribution
    """
    
    def __init__(self, returns: Series, benchmark_returns: Optional[Series] = None) -> None:
        """
        Initialize FinancialMetrics.
        
        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns for comparison
        """
        self.returns: Series = returns.dropna()
        self.benchmark_returns: Optional[Series] = benchmark_returns.dropna() if benchmark_returns is not None else None
        
        logger.info(msg=f"FinancialMetrics initialized with {len(self.returns)} observations")
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02, annualization_factor: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            risk_free_rate: Annual risk-free rate
            annualization_factor: Factor to annualize returns (252 for daily)
            
        Returns:
            Sharpe ratio
        """
        excess_returns: Series = self.returns - (risk_free_rate / annualization_factor)
        sharpe_ratio = np.sqrt(annualization_factor) * excess_returns.mean() / excess_returns.std()
        
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02, annualization_factor: int = 252) -> float:
        """
        Calculate Sortino ratio (downside deviation).
        
        Args:
            risk_free_rate: Annual risk-free rate
            annualization_factor: Factor to annualize returns
            
        Returns:
            Sortino ratio
        """
        excess_returns: Series = self.returns - (risk_free_rate / annualization_factor)
        downside_returns: Series = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        sortino_ratio = np.sqrt(annualization_factor) * excess_returns.mean() / downside_deviation
        
        return sortino_ratio
    
    def calculate_calmar_ratio(self, annualization_factor: int = 252) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        
        Args:
            annualization_factor: Factor to annualize returns
            
        Returns:
            Calmar ratio
        """
        annual_return: float = self.returns.mean() * annualization_factor
        max_drawdown: float = self.calculate_max_drawdown()
        
        if max_drawdown == 0:
            return np.inf
        
        calmar_ratio: float = annual_return / abs(max_drawdown)
        
        return calmar_ratio
    
    def calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.
        
        Returns:
            Maximum drawdown as negative percentage
        """
        cumulative_returns = (1 + self.returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return drawdown.min()
    
    def calculate_drawdown_duration(self) -> Dict[str, Any]:
        """
        Calculate drawdown duration statistics.
        
        Returns:
            Dictionary with drawdown duration metrics
        """
        cumulative_returns = (1 + self.returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods: List = []
        current_period_start: Optional[int] = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and current_period_start is None:
                current_period_start = i
            elif not is_dd and current_period_start is not None:
                drawdown_periods.append(i - current_period_start)
                current_period_start = None
        
        # Handle case where drawdown period extends to end
        if current_period_start is not None:
            drawdown_periods.append(len(in_drawdown) - current_period_start)
        
        if not drawdown_periods:
            return {
                'max_drawdown_duration': 0,
                'avg_drawdown_duration': 0,
                'total_drawdown_periods': 0
            }
        
        return {
            'max_drawdown_duration': max(drawdown_periods),
            'avg_drawdown_duration': np.mean(drawdown_periods),
            'total_drawdown_periods': len(drawdown_periods)
        }
    
    def calculate_hit_rate(self) -> float:
        """
        Calculate hit rate (percentage of positive returns).
        
        Returns:
            Hit rate as percentage
        """
        positive_returns: int = (self.returns > 0).sum()
        total_returns: int = len(self.returns)
        
        return (positive_returns / total_returns) * 100
    
    def calculate_win_loss_ratio(self) -> Dict[str, Any]:
        """
        Calculate win/loss statistics.
        
        Returns:
            Dictionary with win/loss metrics
        """
        positive_returns: Series | np.ndarray = self.returns[self.returns > 0]
        negative_returns: Series | np.ndarray = self.returns[self.returns < 0]
        
        if len(negative_returns) == 0:
            return {
                'win_loss_ratio': np.inf,
                'avg_win': positive_returns.mean(),
                'avg_loss': 0,
                'largest_win': positive_returns.max(),
                'largest_loss': 0
            }
        
        return {
            'win_loss_ratio': abs(positive_returns.mean() / negative_returns.mean()),
            'avg_win': positive_returns.mean(),
            'avg_loss': negative_returns.mean(),
            'largest_win': positive_returns.max(),
            'largest_loss': negative_returns.min()
        }
    
    def calculate_volatility_metrics(self, annualization_factor: int = 252) -> Dict[str, float]:
        """
        Calculate volatility metrics.
        
        Args:
            annualization_factor: Factor to annualize volatility
            
        Returns:
            Dictionary with volatility metrics
        """
        volatility = self.returns.std() * np.sqrt(annualization_factor)
        
        # Calculate rolling volatility
        rolling_vol = self.returns.rolling(window=min(30, len(self.returns))).std() * np.sqrt(annualization_factor)
        
        return {
            'annualized_volatility': volatility,
            'volatility_of_volatility': rolling_vol.std(),
            'min_rolling_volatility': rolling_vol.min(),
            'max_rolling_volatility': rolling_vol.max()
        }
    
    def calculate_skewness_kurtosis(self) -> Dict[str, float]:
        """
        Calculate skewness and kurtosis of returns.
        
        Returns:
            Dictionary with skewness and kurtosis
        """
        from scipy.stats import skew, kurtosis
        
        return {
            'skewness': skew(a=self.returns),
            'kurtosis': kurtosis(a=self.returns),
            'excess_kurtosis': kurtosis(a=self.returns, fisher=True)
        }
    
    def calculate_information_ratio(self) -> Optional[float]:
        """
        Calculate information ratio (active return / tracking error).
        
        Returns:
            Information ratio or None if no benchmark
        """
        if self.benchmark_returns is None:
            return None
        
        # Align returns
        aligned_returns: Tuple[Series, Series] = self.returns.align(other=self.benchmark_returns, join='inner')
        active_returns: Series = aligned_returns[0] - aligned_returns[1]
        
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return np.inf
        
        information_ratio = active_returns.mean() / tracking_error
        
        return information_ratio
    
    def calculate_beta(self) -> Optional[float]:
        """
        Calculate beta relative to benchmark.
        
        Returns:
            Beta coefficient or None if no benchmark
        """
        if self.benchmark_returns is None:
            return None
        
        # Align returns
        aligned_returns: Tuple[Series, Series] = self.returns.align(other=self.benchmark_returns, join='inner')
        
        covariance = np.cov(aligned_returns[0], aligned_returns[1])[0, 1]
        benchmark_variance: np.floating = np.var(aligned_returns[1])
        
        if benchmark_variance == 0:
            return np.inf
        
        beta = covariance / benchmark_variance
        
        return beta
    
    def calculate_alpha(self, risk_free_rate: float = 0.02, annualization_factor: int = 252) -> Optional[float]:
        """
        Calculate alpha (excess return over CAPM).
        
        Args:
            risk_free_rate: Annual risk-free rate
            annualization_factor: Factor to annualize returns
            
        Returns:
            Alpha or None if no benchmark
        """
        if self.benchmark_returns is None:
            return None
        
        beta: Optional[float] = self.calculate_beta()
        if beta is None:
            return None
        
        # Align returns
        aligned_returns: Tuple[Series, Series] = self.returns.align(other=self.benchmark_returns, join='inner')
        
        portfolio_return: float = aligned_returns[0].mean() * annualization_factor
        benchmark_return: float = aligned_returns[1].mean() * annualization_factor
        
        expected_return: float = risk_free_rate + beta * (benchmark_return - risk_free_rate)
        alpha: float = portfolio_return - expected_return
        
        return alpha
    
    def calculate_comprehensive_metrics(self, risk_free_rate: float = 0.02, annualization_factor: int = 252) -> Dict[str, Any]:
        """
        Calculate comprehensive financial metrics.
        
        Args:
            risk_free_rate: Annual risk-free rate
            annualization_factor: Factor to annualize returns
            
        Returns:
            Dictionary with all financial metrics
        """
        logger.info(msg="Calculating comprehensive financial metrics...")
        
        metrics: Dict[str, Any] = {
            # Return metrics
            'total_return': (1 + self.returns).prod() - 1,
            'annualized_return': self.returns.mean() * annualization_factor,
            'hit_rate': self.calculate_hit_rate(),
            
            # Risk metrics
            'sharpe_ratio': self.calculate_sharpe_ratio(risk_free_rate, annualization_factor),
            'sortino_ratio': self.calculate_sortino_ratio(risk_free_rate, annualization_factor),
            'calmar_ratio': self.calculate_calmar_ratio(annualization_factor),
            'max_drawdown': self.calculate_max_drawdown(),
            
            # Drawdown analysis
            'drawdown_analysis': self.calculate_drawdown_duration(),
            
            # Win/Loss analysis
            'win_loss_analysis': self.calculate_win_loss_ratio(),
            
            # Volatility metrics
            'volatility_metrics': self.calculate_volatility_metrics(annualization_factor),
            
            # Distribution metrics
            'distribution_metrics': self.calculate_skewness_kurtosis(),
        }
        
        # Benchmark-relative metrics
        if self.benchmark_returns is not None:
            metrics['benchmark_metrics'] = {
                'information_ratio': self.calculate_information_ratio(),
                'beta': self.calculate_beta(),
                'alpha': self.calculate_alpha(risk_free_rate, annualization_factor)
            }
        
        logger.info(msg="Financial metrics calculation completed")
        
        return metrics
    
    def generate_performance_report(self, risk_free_rate: float = 0.02, annualization_factor: int = 252) -> str:
        """
        Generate comprehensive performance report.
        
        Args:
            risk_free_rate: Annual risk-free rate
            annualization_factor: Factor to annualize returns
            
        Returns:
            Formatted performance report
        """
        metrics: Dict[str, Any] = self.calculate_comprehensive_metrics(risk_free_rate, annualization_factor)
        
        report: List = []
        report.append("=" * 80)
        report.append("FINANCIAL PERFORMANCE REPORT")
        report.append("=" * 80)
        
        # Basic statistics
        report.append(f"\nBASIC STATISTICS:")
        report.append(f"Total observations: {len(self.returns)}")
        report.append(f"Total return: {metrics['total_return']:.4f} ({metrics['total_return']*100:.2f}%)")
        report.append(f"Annualized return: {metrics['annualized_return']:.4f} ({metrics['annualized_return']*100:.2f}%)")
        report.append(f"Hit rate: {metrics['hit_rate']:.2f}%")
        
        # Risk-adjusted returns
        report.append(f"\nRISK-ADJUSTED RETURNS:")
        report.append(f"Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
        report.append(f"Sortino ratio: {metrics['sortino_ratio']:.4f}")
        report.append(f"Calmar ratio: {metrics['calmar_ratio']:.4f}")
        
        # Risk metrics
        report.append(f"\nRISK METRICS:")
        report.append(f"Maximum drawdown: {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
        report.append(f"Annualized volatility: {metrics['volatility_metrics']['annualized_volatility']:.4f} ({metrics['volatility_metrics']['annualized_volatility']*100:.2f}%)")
        
        # Drawdown analysis
        dd_analysis = metrics['drawdown_analysis']
        report.append(f"\nDRAWDOWN ANALYSIS:")
        report.append(f"Maximum drawdown duration: {dd_analysis['max_drawdown_duration']} periods")
        report.append(f"Average drawdown duration: {dd_analysis['avg_drawdown_duration']:.2f} periods")
        report.append(f"Total drawdown periods: {dd_analysis['total_drawdown_periods']}")
        
        # Win/Loss analysis
        wl_analysis = metrics['win_loss_analysis']
        report.append(f"\nWIN/LOSS ANALYSIS:")
        report.append(f"Win/Loss ratio: {wl_analysis['win_loss_ratio']:.4f}")
        report.append(f"Average win: {wl_analysis['avg_win']:.4f} ({wl_analysis['avg_win']*100:.2f}%)")
        report.append(f"Average loss: {wl_analysis['avg_loss']:.4f} ({wl_analysis['avg_loss']*100:.2f}%)")
        report.append(f"Largest win: {wl_analysis['largest_win']:.4f} ({wl_analysis['largest_win']*100:.2f}%)")
        report.append(f"Largest loss: {wl_analysis['largest_loss']:.4f} ({wl_analysis['largest_loss']*100:.2f}%)")
        
        # Distribution metrics
        dist_metrics = metrics['distribution_metrics']
        report.append(f"\nDISTRIBUTION METRICS:")
        report.append(f"Skewness: {dist_metrics['skewness']:.4f}")
        report.append(f"Kurtosis: {dist_metrics['kurtosis']:.4f}")
        report.append(f"Excess kurtosis: {dist_metrics['excess_kurtosis']:.4f}")
        
        # Benchmark metrics
        if 'benchmark_metrics' in metrics:
            bench_metrics = metrics['benchmark_metrics']
            report.append(f"\nBENCHMARK RELATIVE METRICS:")
            if bench_metrics['information_ratio'] is not None:
                report.append(f"Information ratio: {bench_metrics['information_ratio']:.4f}")
            if bench_metrics['beta'] is not None:
                report.append(f"Beta: {bench_metrics['beta']:.4f}")
            if bench_metrics['alpha'] is not None:
                report.append(f"Alpha: {bench_metrics['alpha']:.4f} ({bench_metrics['alpha']*100:.2f}%)")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
