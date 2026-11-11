import duckdb
import pandas as pd
import numpy as np

from utils.data_store import connect_input

class FactorDataInterface:
    """
    数据接口类，用于获取股票数据、特征数据，计算未来收益率并构建因子挖掘用宽表
    """
    
    def __init__(self, db_path, *, data_bindings=None):
        """
        初始化数据接口，打开并保持到指定数据库的连接
        
        Args:
            db_path: DuckDB 数据库文件路径，或现有 DuckDB 连接
        """
        self._owns_connection = not isinstance(db_path, duckdb.DuckDBPyConnection)
        self.conn = connect_input(db_path, bindings=data_bindings) if self._owns_connection else db_path

    def _fetchdf(self, query: str) -> pd.DataFrame:
        return self.conn.execute(query).fetchdf()
    
    def get_filtered_symbols(self, filters):
        """
        根据日频特征字段的条件，查询并返回符合条件的股票代码列表
        
        Args:
            filters: 字典，键为特征列名，值为带比较符号的过滤表达式
                    例如: {"momentum": ">0.1", "volatility": "<0.05"}
        
        Returns:
            符合条件的股票代码列表
        """
        conditions = []
        for feature, condition in filters.items():
            conditions.append(f"{feature} {condition}")
        
        query = f"""
        SELECT DISTINCT symbol
        FROM daily_features
        WHERE {' AND '.join(conditions)}
        """
        
        symbols = self._fetchdf(query)['symbol'].tolist()
        return symbols
    
    def get_feature_data(self, symbols, start_date=None, end_date=None):
        """
        获取指定股票在给定日期区间内的所有日频特征
        
        Args:
            symbols: 待查询的股票代码列表
            start_date: 起始日期，格式'YYYY-MM-DD'，可选
            end_date: 结束日期，格式'YYYY-MM-DD'，可选
            
        Returns:
            DataFrame，索引为(symbol, date)，列为各特征字段
        """
        conditions = []
        if symbols:
            symbols_str = "', '".join(symbols)
            conditions.append(f"symbol IN ('{symbols_str}')")
        
        if start_date:
            conditions.append(f"date >= '{start_date}'")
        
        if end_date:
            conditions.append(f"date <= '{end_date}'")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT * 
        FROM daily_features
        WHERE {where_clause}
        ORDER BY symbol, date
        """
        
        df = self._fetchdf(query)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index(['symbol', 'date'])
        
        return df
    
    def get_price_data(self, symbols, start_datetime=None, end_datetime=None):
        """
        获取指定股票在给定时间区间内的5分钟级别价格数据
        
        Args:
            symbols: 待查询的股票代码列表
            start_datetime: 起始时间，格式'YYYY-MM-DD HH:MM:SS'，可选
            end_datetime: 结束时间，格式'YYYY-MM-DD HH:MM:SS'，可选
            
        Returns:
            DataFrame，索引为(symbol, datetime)，包含close收盘价列
        """
        conditions = []
        if symbols:
            symbols_str = "', '".join(symbols)
            conditions.append(f"symbol IN ('{symbols_str}')")
        
        if start_datetime:
            conditions.append(f"datetime >= '{start_datetime}'")
        
        if end_datetime:
            conditions.append(f"datetime <= '{end_datetime}'")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT symbol, datetime, close
        FROM prices_5min
        WHERE {where_clause}
        ORDER BY symbol, datetime
        """
        
        df = self._fetchdf(query)
        
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index(['symbol', 'datetime'])
        
        return df
    
    def compute_future_returns(self, price_df, horizons):
        """
        基于5分钟级价格计算多种未来收益率
        
        Args:
            price_df: DataFrame，索引为(symbol, datetime)，包含close列
            horizons: 整数列表，以交易日为单位的收益期列表
            
        Returns:
            DataFrame，索引同price_df，列名为ret_Xd（X为对应周期）
        """
        if price_df.empty:
            return pd.DataFrame()
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=price_df.index)
        
        # 重置索引以便处理
        price_df = price_df.reset_index()
        price_df['date'] = price_df['datetime'].dt.date
        
        # 对每只股票分别处理
        for symbol in price_df['symbol'].unique():
            symbol_data = price_df[price_df['symbol'] == symbol].copy()
            
            # 获取交易日列表
            daily_closes = symbol_data.groupby('date')['close'].last()
            trading_dates = daily_closes.index.tolist()
            
            # 为每个周期计算未来收益
            for horizon in horizons:
                col_name = f'ret_{horizon}d'
                result[col_name] = np.nan  # 初始化收益率列
                
                for i, row in symbol_data.iterrows():
                    current_date = row['date']
                    current_price = row['close']
                    
                    # 找到当前日期在交易日列表中的位置
                    try:
                        date_idx = trading_dates.index(current_date)
                        if date_idx + horizon < len(trading_dates):
                            future_date = trading_dates[date_idx + horizon]
                            future_price = daily_closes.loc[future_date]
                            
                            # 计算收益率: (future_price / current_price) - 1
                            ret = future_price / current_price - 1
                            result.loc[(symbol, row['datetime']), col_name] = ret
                    except ValueError:
                        continue
        
        return result
    
    def build_factor_dataframe(self, filters, price_range, feature_range, horizons):
        """
        串联所有步骤，输出日频特征与未来收益对齐后的宽表
        
        Args:
            filters: 特征筛选条件
            price_range: {'start': ..., 'end': ...}，5分钟价格时间区间
            feature_range: {'start': ..., 'end': ...}，日频特征日期区间
            horizons: 未来收益周期列表
            
        Returns:
            DataFrame，每行对应一只股票在某个交易日，包含特征和收益率
        """
        # 1. 获取符合条件的股票
        filtered_symbols = self.get_filtered_symbols(filters)
        
        if not filtered_symbols:
            return pd.DataFrame()
        
        # 2. 获取特征数据
        feature_data = self.get_feature_data(
            filtered_symbols,
            feature_range.get('start'),
            feature_range.get('end')
        )
        
        # 3. 获取价格数据
        price_data = self.get_price_data(
            filtered_symbols,
            price_range.get('start'),
            price_range.get('end')
        )
        
        # 4. 计算未来收益率
        returns_data = self.compute_future_returns(price_data, horizons)
        
        # 5. 合并数据前的准备
        # 将价格和收益率数据聚合到日级别
        price_df_reset = price_data.reset_index()
        price_df_reset['date'] = price_df_reset['datetime'].dt.date
        
        # 获取每个交易日最后一个价格bar的索引
        last_bars_idx = price_df_reset.groupby(['symbol', 'date'])['datetime'].idxmax()
        last_bars = price_df_reset.loc[last_bars_idx]
        
        # 提取对应时间点的收益率
        returns_for_merge = returns_data.loc[list(zip(last_bars['symbol'], last_bars['datetime']))]
        returns_for_merge = returns_for_merge.reset_index()
        returns_for_merge['date'] = returns_for_merge['datetime'].dt.date
        returns_for_merge = returns_for_merge.drop('datetime', axis=1)
        
        # 设置索引用于合并
        returns_for_merge = returns_for_merge.set_index(['symbol', 'date'])
        
        # 6. 合并特征数据和收益率
        merged_data = pd.concat([feature_data, returns_for_merge], axis=1)
        
        # 重置索引，转为宽表格式
        result = merged_data.reset_index()
        
        return result
    
    def close(self):
        """关闭数据库连接"""
        if self._owns_connection and hasattr(self, 'conn') and self.conn:
            self.conn.close()
def test_interface():
    # 初始化接口
    factor_interface = FactorDataInterface("path_to_database.duckdb")

    # 定义筛选条件
    filters = {
        "momentum": ">0.05",
        "volatility": "<0.2"
    }

    # 定义时间范围
    price_range = {
        "start": "2023-01-01 09:30:00",
        "end": "2023-03-31 15:00:00"
    }

    feature_range = {
        "start": "2023-01-01",
        "end": "2023-03-31"
    }

    # 定义收益周期
    horizons = [5, 10, 20]

    # 构建因子数据宽表
    factor_data = factor_interface.build_factor_dataframe(filters, price_range, feature_range, horizons)

    # 使用完成后关闭连接
    factor_interface.close()
    pass
