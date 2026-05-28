# src/pipeline.py

from data_fetcher import MarketDataFetcher
from data_cleaner import DataCleaner
import pandas as pd
import logging
from typing import List, Dict
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataPipeline:
    """
    Complete market data processing pipeline
    """
    
    def __init__(self, config_path: str = '../config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.fetcher = MarketDataFetcher(self.config['data_dir'])
        self.cleaner = DataCleaner()
        
    def run_pipeline(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Run the complete data pipeline
        """
        # 1. Fetch data
        logger.info("Step 1: Fetching data...")
        raw_data = self.fetcher.fetch_yfinance_data(
            symbols, 
            start_date, 
            end_date,
            self.config['interval']
        )
        
        processed_data = {}
        
        # 2. Process each symbol
        for symbol, df in raw_data.items():
            logger.info(f"\nProcessing {symbol}...")
            
            # Clean timestamps
            df = self.cleaner.clean_timestamps(
                df, 
                self.config['timezone']
            )
            
            # Standardize columns
            df = self.cleaner.standardize_columns(df)
            
            # Align frequency
            df = self.cleaner.align_frequency(
                df,
                self.config['target_freq'],
                self.config['resample_method']
            )
            
            # Handle missing data
            df = self.cleaner.handle_missing_data(
                df,
                self.config['max_gap'],
                self.config['fill_method']
            )
            
            # Calculate returns
            df = self.cleaner.calculate_returns(
                df,
                method=self.config['return_method']
            )
            
            # Save processed data
            df.to_csv(f"{self.config['data_dir']}/processed/{symbol}_processed.csv")
            
            # Generate quality report
            self._generate_quality_report(df, symbol)
            
            processed_data[symbol] = df
            
        return processed_data
    
    def _generate_quality_report(self, df: pd.DataFrame, symbol: str):
        """Generate data quality metrics"""
        report = f"""
        Data Quality Report for {symbol}
        ================================
        Date Range: {df.index.min()} to {df.index.max()}
        Total Periods: {len(df)}
        
        Missing Data:
        {df.isnull().sum()}
        
        Data Statistics:
        {df.describe()}
        
        Returns Statistics:
        Skewness: {df['returns'].skew():.4f}
        Kurtosis: {df['returns'].kurtosis():.4f}
        """
        
        with open(f"{self.config['data_dir']}/processed/{symbol}_report.txt", 'w') as f:
            f.write(report)
        
        logger.info(f"Quality report saved for {symbol}")

if __name__ == "__main__":
    # Example usage
    pipeline = MarketDataPipeline()
    
    results = pipeline.run_pipeline(
        symbols=['AAPL', 'GOOGL', 'MSFT', 'SPY'],
        start_date='2020-01-01',
        end_date='2023-12-31'
    )