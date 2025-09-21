import numpy as np
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import (
    SelectKBest, RFE, RFECV,
    mutual_info_regression, f_regression
)
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from typing import List, Dict, Set, Any, Optional

from utils.logging import get_logger, Logger

logger: Logger = get_logger()

class FeatureAnalyzer:
    """
    Comprehensive feature analysis and selection for stock price prediction.
    
    This class provides:
    - Correlation analysis
    - Feature importance analysis
    - Univariate and multivariate feature selection
    - Temporal stability analysis
    - Feature ranking and recommendations
    """
    
    def __init__(self, df: DataFrame, target_column: str = 'target_1d') -> None:
        """
        Initialize FeatureAnalyzer.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of the target column
        """
        self.df: DataFrame = df.copy()
        self.target_column: str = target_column
        self.feature_columns: List[Any] = [col for col in df.columns if col != target_column]
        self.scaler: StandardScaler = StandardScaler()
        
        logger.info(msg=f"FeatureAnalyzer initialized with {len(self.feature_columns)} features")
    
    def analyze_correlations(self, threshold: float = 0.95) -> Dict[str, Any]:
        """
        Analyze correlations between features and target.
        
        Args:
            threshold: Correlation threshold for identifying highly correlated features
            
        Returns:
            Dictionary with correlation analysis results
        """
        logger.info(msg="Analyzing feature correlations...")
        
        # Calculate correlation matrix
        corr_matrix: DataFrame = self.df[self.feature_columns + [self.target_column]].corr() # type: ignore
        
        # Target correlations
        target_correlations: Series = corr_matrix[self.target_column].drop(self.target_column).abs().sort_values(ascending=False) # type: ignore
        
        # Find highly correlated feature pairs
        high_corr_pairs: List[Dict[str, Any]] = []
        for i in range(len(self.feature_columns)):
            for j in range(i+1, len(self.feature_columns)):
                corr_val: float = abs(corr_matrix.iloc[i, j])
                if corr_val > threshold:
                    high_corr_pairs.append({
                        'feature1': self.feature_columns[i],
                        'feature2': self.feature_columns[j],
                        'correlation': corr_val
                    })
        
        # Feature redundancy analysis
        redundant_features: Set = set()
        for pair in high_corr_pairs:
            # Keep the feature with higher correlation to target
            corr1: int = abs(target_correlations.get(key=pair['feature1'], default=0) or 0)
            corr2: int = abs(target_correlations.get(key=pair['feature2'], default=0) or 0)
            
            if corr1 < corr2:
                redundant_features.add(pair['feature1'])
            else:
                redundant_features.add(pair['feature2'])
        
        results: Dict[str, Any] = {
            'target_correlations': target_correlations,
            'high_correlation_pairs': high_corr_pairs,
            'redundant_features': list(redundant_features),
            'correlation_matrix': corr_matrix,
            'recommended_features': [f for f in self.feature_columns if f not in redundant_features]
        }
        
        logger.info(msg=f"Found {len(high_corr_pairs)} highly correlated pairs")
        logger.info(msg=f"Identified {len(redundant_features)} redundant features")
        
        return results
    
    def analyze_feature_importance(self, methods: List[str] = ['random_forest', 'lasso', 'mutual_info']) -> Dict[str, Any]:
        """
        Analyze feature importance using multiple methods.
        
        Args:
            methods: List of methods to use for importance analysis
            
        Returns:
            Dictionary with importance scores for each method
        """
        logger.info(msg=f"Analyzing feature importance using methods: {methods}")
        
        # Prepare data
        X: DataFrame | Series = self.df[self.feature_columns].fillna(0)
        y: DataFrame | Series = self.df[self.target_column].fillna(0)
        
        importance_scores: Dict = {}
        
        if 'random_forest' in methods:
            logger.info(msg="Calculating Random Forest importance...")
            rf: RandomForestRegressor = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importance_scores['random_forest'] = Series(
                data=rf.feature_importances_, 
                index=self.feature_columns
            ).sort_values(ascending=False)
        
        if 'lasso' in methods:
            logger.info(msg="Calculating Lasso importance...")
            lasso: LassoCV = LassoCV(cv=5, random_state=42)
            lasso.fit(X, y)
            importance_scores['lasso'] = Series(
                data=np.abs(lasso.coef_), 
                index=self.feature_columns
            ).sort_values(ascending=False)
        
        if 'mutual_info' in methods:
            logger.info(msg="Calculating Mutual Information...")
            mi_scores: np.ndarray = mutual_info_regression(X, y, random_state=42)
            importance_scores['mutual_info'] = Series(
                data=mi_scores, 
                index=self.feature_columns
            ).sort_values(ascending=False)
        
        if 'f_regression' in methods:
            logger.info(msg="Calculating F-regression scores...")
            f_scores, _ = f_regression(X, y)
            importance_scores['f_regression'] = Series(
                data=f_scores, 
                index=self.feature_columns
            ).sort_values(ascending=False)
        
        # Calculate combined importance score
        combined_scores: Series = Series(data=0.0, index=self.feature_columns)
        for method, scores in importance_scores.items():
            # Normalize scores to 0-1 range
            normalized_scores: Any = (scores - scores.min()) / (scores.max() - scores.min())
            combined_scores += normalized_scores
        
        combined_scores = combined_scores / len(importance_scores)
        importance_scores['combined'] = combined_scores.sort_values(ascending=False)
        
        logger.info(msg="Feature importance analysis completed")
        
        return importance_scores
    
    def univariate_feature_selection(self, k: int = 20, score_func: str = 'f_regression') -> Dict[str, Any]:
        """
        Perform univariate feature selection.
        
        Args:
            k: Number of top features to select
            score_func: Scoring function ('f_regression', 'mutual_info')
            
        Returns:
            Dictionary with selection results
        """
        logger.info(msg=f"Performing univariate feature selection with k={k}")
        
        X: DataFrame | Series = self.df[self.feature_columns].fillna(0)
        y: DataFrame | Series = self.df[self.target_column].fillna(0)
        
        # Choose scoring function
        if score_func == 'f_regression':
            selector: SelectKBest = SelectKBest(score_func=f_regression, k=k)
        elif score_func == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            raise ValueError(f"Unknown score function: {score_func}")
        
        selector.fit(X, y)
        
        # Get selected features
        support: Optional[np.ndarray] = selector.get_support(indices=True)
        selected_features: List = []
        if support is not None:
            for i in support:
                selected_features.append(self.feature_columns[i])
        scores: np.ndarray = selector.scores_
        
        results: Dict[str, Any] = {
            'selected_features': selected_features,
            'feature_scores': Series(data=scores, index=self.feature_columns).sort_values(ascending=False),
            'selector': selector
        }
        
        logger.info(msg=f"Selected {len(selected_features)} features using univariate selection")
        
        return results
    
    def multivariate_feature_selection(self, method: str = 'rfe', n_features: int = 20) -> Dict[str, Any]:
        """
        Perform multivariate feature selection.
        
        Args:
            method: Selection method ('rfe', 'rfecv')
            n_features: Number of features to select
            
        Returns:
            Dictionary with selection results
        """
        logger.info(msg=f"Performing multivariate feature selection using {method}")
        
        X: DataFrame | Series = self.df[self.feature_columns].fillna(0)
        y: DataFrame | Series = self.df[self.target_column].fillna(0)
        
        # Use Random Forest as base estimator
        base_estimator: RandomForestRegressor = RandomForestRegressor(n_estimators=50, random_state=42)
        
        selector: RFE | RFECV
        if method == 'rfe':
            selector = RFE(estimator=base_estimator, n_features_to_select=n_features)
        elif method == 'rfecv':
            # Use time series cross-validation
            tscv: TimeSeriesSplit = TimeSeriesSplit(n_splits=3)
            selector = RFECV(estimator=base_estimator, cv=tscv, scoring='neg_mean_squared_error')
        else:
            raise ValueError(f"Unknown method: {method}")
        
        selector.fit(X, y)
        
        # Get selected features
        support: Optional[np.ndarray] = selector.get_support(indices=True)
        selected_features: List = []
        if support is not None:
            for i in support:
                selected_features.append(self.feature_columns[i])
        
        results: Dict[str, Any] = {
            'selected_features': selected_features,
            'feature_ranking': Series(data=selector.ranking_, index=self.feature_columns).sort_values(),
            'selector': selector
        }
        
        if isinstance(selector, RFECV):
            results['optimal_n_features'] = selector.n_features_
            results['cv_scores'] = selector.cv_results_
        
        logger.info(msg=f"Selected {len(selected_features)} features using multivariate selection")
        
        return results
    
    def analyze_temporal_stability(self, window_size: int = 252) -> Dict[str, Any]:
        """
        Analyze temporal stability of features.
        
        Args:
            window_size: Size of rolling window for stability analysis
            
        Returns:
            Dictionary with stability analysis results
        """
        logger.info(msg=f"Analyzing temporal stability with window size {window_size}")
        
        stability_scores: Dict = {}
        
        for feature in self.feature_columns:
            if feature in self.df.columns:
                # Calculate rolling correlation with target
                rolling_corr: DataFrame | Series = self.df[feature].rolling(window=window_size).corr(
                    other=self.df[self.target_column]
                )
                
                # Calculate stability metrics
                mean_corr: Series | float | int = rolling_corr.mean()
                std_corr: Series | float | int = rolling_corr.std()
                stability_score: float | np.ndarray = mean_corr / (std_corr + 1e-8)  # Avoid division by zero
                
                stability_scores[feature] = {
                    'mean_correlation': mean_corr,
                    'correlation_std': std_corr,
                    'stability_score': stability_score,
                    'rolling_correlation': rolling_corr
                }
        
        # Sort by stability score
        sorted_stability: List = sorted(
            stability_scores.items(), 
            key=lambda x: x[1]['stability_score'], 
            reverse=True
        )
        
        results: Dict[str, Any] = {
            'stability_scores': dict(sorted_stability),
            'most_stable_features': [item[0] for item in sorted_stability[:10]],
            'least_stable_features': [item[0] for item in sorted_stability[-10:]]
        }
        
        logger.info(msg="Temporal stability analysis completed")
        
        return results
    
    def comprehensive_feature_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive feature analysis combining all methods.
        
        Returns:
            Dictionary with comprehensive analysis results
        """
        logger.info(msg="Performing comprehensive feature analysis...")
        
        # Correlation analysis
        corr_results: Dict[str, Any] = self.analyze_correlations()
        
        # Feature importance analysis
        importance_results: Dict[str, Any] = self.analyze_feature_importance()
        
        # Univariate selection
        univariate_results: Dict[str, Any] = self.univariate_feature_selection()
        
        # Multivariate selection
        multivariate_results: Dict[str, Any] = self.multivariate_feature_selection()
        
        # Temporal stability analysis
        stability_results: Dict[str, Any] = self.analyze_temporal_stability()
        
        # Combine results
        comprehensive_results: Dict[str, Dict[str, Any]] = {
            'correlation_analysis': corr_results,
            'importance_analysis': importance_results,
            'univariate_selection': univariate_results,
            'multivariate_selection': multivariate_results,
            'temporal_stability': stability_results
        }
        
        # Generate feature recommendations
        recommendations: Dict[str, Any] = self._generate_feature_recommendations(analysis_results=comprehensive_results)
        comprehensive_results['recommendations'] = recommendations
        
        logger.info(msg="Comprehensive feature analysis completed")
        
        return comprehensive_results
    
    def _generate_feature_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate feature recommendations based on analysis results.
        
        Args:
            analysis_results: Results from comprehensive analysis
            
        Returns:
            Dictionary with feature recommendations
        """
        # Get top features from different methods
        top_importance = analysis_results['importance_analysis']['combined'].head(20).index.tolist()
        top_univariate = analysis_results['univariate_selection']['selected_features']
        top_multivariate = analysis_results['multivariate_selection']['selected_features']
        most_stable = analysis_results['temporal_stability']['most_stable_features']
        
        # Find consensus features
        consensus_features: List = []
        for feature in self.feature_columns:
            count = 0
            if feature in top_importance:
                count += 1
            if feature in top_univariate:
                count += 1
            if feature in top_multivariate:
                count += 1
            if feature in most_stable:
                count += 1
            
            if count >= 2:  # Feature appears in at least 2 methods
                consensus_features.append(feature)
        
        # Categorize features
        technical_features: List = [f for f in consensus_features if any(x in f.lower() for x in ['sma', 'ema', 'rsi', 'macd', 'bb'])]
        volatility_features: List = [f for f in consensus_features if 'volatility' in f.lower()]
        return_features: List = [f for f in consensus_features if 'return' in f.lower()]
        volume_features: List = [f for f in consensus_features if 'volume' in f.lower()]
        
        recommendations: Dict[str, Any] = {
            'consensus_features': consensus_features,
            'technical_indicators': technical_features,
            'volatility_features': volatility_features,
            'return_features': return_features,
            'volume_features': volume_features,
            'total_recommended': len(consensus_features),
            'feature_categories': {
                'Technical Indicators': len(technical_features),
                'Volatility': len(volatility_features),
                'Returns': len(return_features),
                'Volume': len(volume_features)
            }
        }
        
        return recommendations
    
    def create_feature_visualizations(self, analysis_results: Dict[str, Any], save_plots: bool = False) -> None:
        """
        Create visualizations for feature analysis.
        
        Args:
            analysis_results: Results from comprehensive analysis
            save_plots: Whether to save plots to files
        """
        logger.info(msg="Creating feature analysis visualizations...")
        
        # Set up the plotting style
        plt.style.use(style='seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(t='Feature Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Top features by importance
        ax1 = axes[0, 0]
        top_features = analysis_results['importance_analysis']['combined'].head(15)
        top_features.plot(kind='barh', ax=ax1, color='skyblue')
        ax1.set_title('Top 15 Features by Combined Importance')
        ax1.set_xlabel('Importance Score')
        
        # 2. Correlation heatmap for top features
        ax2 = axes[0, 1]
        top_corr_features = analysis_results['correlation_analysis']['target_correlations'].head(10).index.tolist()
        corr_matrix = analysis_results['correlation_analysis']['correlation_matrix']
        top_corr_matrix = corr_matrix.loc[top_corr_features, top_corr_features]
        
        sns.heatmap(data=top_corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax2)
        ax2.set_title('Correlation Matrix - Top 10 Features')
        
        # 3. Feature stability scores
        ax3 = axes[1, 0]
        stability_data = analysis_results['temporal_stability']['stability_scores']
        stability_scores: List = [data['stability_score'] for data in stability_data.values()]
        feature_names: List = list(stability_data.keys())
        
        # Sort by stability score
        sorted_data: List = sorted(zip(feature_names, stability_scores), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_scores = zip(*sorted_data[:15])
        
        ax3.barh(range(len(sorted_names)), sorted_scores, color='lightgreen')
        ax3.set_yticks(range(len(sorted_names)))
        ax3.set_yticklabels(sorted_names)
        ax3.set_title('Top 15 Most Stable Features')
        ax3.set_xlabel('Stability Score')
        
        # 4. Feature selection comparison
        ax4 = axes[1, 1]
        methods: List[str] = ['Univariate', 'Multivariate', 'Importance', 'Stability']
        counts: List[int] = [
            len(analysis_results['univariate_selection']['selected_features']),
            len(analysis_results['multivariate_selection']['selected_features']),
            len(analysis_results['importance_analysis']['combined'].head(20)),
            len(analysis_results['temporal_stability']['most_stable_features'])
        ]
        
        bars = ax4.bar(methods, counts, color=['red', 'blue', 'green', 'orange'])
        ax4.set_title('Feature Selection Methods Comparison')
        ax4.set_ylabel('Number of Selected Features')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('feature_analysis_dashboard.png', dpi=300, bbox_inches='tight')
            logger.info(msg="Feature analysis plots saved to 'feature_analysis_dashboard.png'")
        
        plt.show()
    
    def get_optimal_feature_set(self, analysis_results: Dict[str, Any], max_features: int = 30) -> List[str]:
        """
        Get optimal feature set based on comprehensive analysis.
        
        Args:
            analysis_results: Results from comprehensive analysis
            max_features: Maximum number of features to select
            
        Returns:
            List of optimal features
        """
        logger.info(msg=f"Selecting optimal feature set (max {max_features} features)")
        
        # Get consensus features
        consensus_features = analysis_results['recommendations']['consensus_features']
        
        # If we have fewer consensus features than max_features, add more from importance
        if len(consensus_features) < max_features:
            importance_features = analysis_results['importance_analysis']['combined'].index.tolist()
            additional_features: List = [f for f in importance_features if f not in consensus_features]
            optimal_features = consensus_features + additional_features[:max_features - len(consensus_features)]
        else:
            optimal_features = consensus_features[:max_features]
        
        logger.info(msg=f"Selected {len(optimal_features)} optimal features")
        
        return optimal_features
    
    def generate_feature_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive feature analysis report.
        
        Args:
            analysis_results: Results from comprehensive analysis
            
        Returns:
            Formatted report string
        """
        report: List = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE FEATURE ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Dataset overview
        report.append(f"\nDATASET OVERVIEW:")
        report.append(f"Total features: {len(self.feature_columns)}")
        report.append(f"Total samples: {len(self.df)}")
        report.append(f"Target variable: {self.target_column}")
        
        # Correlation analysis
        corr_results = analysis_results['correlation_analysis']
        report.append(f"\nCORRELATION ANALYSIS:")
        report.append(f"Highly correlated pairs: {len(corr_results['high_correlation_pairs'])}")
        report.append(f"Redundant features identified: {len(corr_results['redundant_features'])}")
        report.append(f"Top 5 features by target correlation:")
        for i, (feature, corr) in enumerate(corr_results['target_correlations'].head(5).items()):
            report.append(f"  {i+1}. {feature}: {corr:.4f}")
        
        # Feature importance
        importance_results = analysis_results['importance_analysis']
        report.append(f"\nFEATURE IMPORTANCE ANALYSIS:")
        report.append(f"Top 5 features by combined importance:")
        for i, (feature, score) in enumerate(importance_results['combined'].head(5).items()):
            report.append(f"  {i+1}. {feature}: {score:.4f}")
        
        # Temporal stability
        stability_results = analysis_results['temporal_stability']
        report.append(f"\nTEMPORAL STABILITY ANALYSIS:")
        report.append(f"Most stable features:")
        for i, feature in enumerate(stability_results['most_stable_features'][:5]):
            stability_score = stability_results['stability_scores'][feature]['stability_score']
            report.append(f"  {i+1}. {feature}: {stability_score:.4f}")
        
        # Recommendations
        recommendations = analysis_results['recommendations']
        report.append(f"\nFEATURE RECOMMENDATIONS:")
        report.append(f"Consensus features: {len(recommendations['consensus_features'])}")
        report.append(f"Feature categories:")
        for category, count in recommendations['feature_categories'].items():
            report.append(f"  {category}: {count} features")
        
        report.append(f"\nOPTIMAL FEATURE SET:")
        optimal_features: List[str] = self.get_optimal_feature_set(analysis_results)
        for i, feature in enumerate(optimal_features[:10]):
            report.append(f"  {i+1}. {feature}")
        if len(optimal_features) > 10:
            report.append(f"  ... and {len(optimal_features) - 10} more")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
