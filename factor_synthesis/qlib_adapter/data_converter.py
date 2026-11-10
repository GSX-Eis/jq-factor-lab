# -*- coding: utf-8 -*-
"""
数据格式转换器

将现有的pd.DataFrame/Series格式转换为Qlib DatasetH格式，
以及将Qlib预测结果转换回原始格式。
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging

try:
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    DatasetH = None
    DataHandlerLP = None


class DataConverter:
    """DuckDB-backed panel data <-> Qlib DatasetH 格式转换器"""
    
    # Exchange code mappings between different formats
    # JoinQuant: XSHE (深圳), XSHG (上海)
    # Standard: SZ (深圳), SH (上海), BJ (北京)
    EXCHANGE_JQ_TO_STD = {'XSHE': 'SZ', 'XSHG': 'SH'}
    EXCHANGE_STD_TO_JQ = {'SZ': 'XSHE', 'SH': 'XSHG', 'BJ': 'BJ'}
    VALID_EXCHANGES = {'SH', 'SZ', 'BJ', 'XSHE', 'XSHG'}
    
    @staticmethod
    def _convert_stock_code_to_instrument(stock_code: str) -> str:
        """
        股票代码格式转换为内部Qlib格式
        
        支持多种输入格式:
        - "600519.SH" / "000001.SZ" -> "SH600519" / "SZ000001"
        - "002006.XSHE" / "600519.XSHG" -> "SZ002006" / "SH600519" (聚宽格式)
        - "600519" -> "SH600519" (自动推断交易所)
        
        输出: "SH600519" 或 "SZ000001" (标准内部格式)
        """
        if pd.isna(stock_code) or not str(stock_code).strip():
            raise ValueError("Missing stock code cannot be converted to a Qlib instrument")
        stock_code = str(stock_code).strip()
        
        if '.' in stock_code:
            code, exchange = stock_code.split('.', 1)
            exchange = exchange.upper()
            
            # Convert JoinQuant format to standard
            if exchange in DataConverter.EXCHANGE_JQ_TO_STD:
                exchange = DataConverter.EXCHANGE_JQ_TO_STD[exchange]
            
            return f"{exchange}{code}"
        else:
            # Infer exchange from code prefix
            if stock_code.startswith('6'):
                return f"SH{stock_code}"
            elif stock_code.startswith(('0', '3')):
                return f"SZ{stock_code}"
            elif stock_code.startswith('8') or stock_code.startswith('4'):
                return f"BJ{stock_code}"
            else:
                return f"SH{stock_code}"  # Default to Shanghai
    
    @staticmethod
    def _convert_instrument_to_stock_code(
        instrument: str, 
        use_jq_format: bool = True
    ) -> str:
        """
        股票代码格式反转换
        
        输入: "SH600519" 或 "SZ000001" (内部格式)
        输出: 
        - use_jq_format=True: "600519.XSHG" / "000001.XSHE" (聚宽格式)
        - use_jq_format=False: "600519.SH" / "000001.SZ" (标准格式)
        """
        if pd.isna(instrument) or not str(instrument).strip():
            raise ValueError("Missing Qlib instrument cannot be converted to a stock code")
        instrument = str(instrument).strip()
        
        if len(instrument) >= 2:
            exchange = instrument[:2].upper()
            code = instrument[2:]
            
            if exchange in ('SH', 'SZ', 'BJ'):
                if use_jq_format and exchange in DataConverter.EXCHANGE_STD_TO_JQ:
                    exchange = DataConverter.EXCHANGE_STD_TO_JQ[exchange]
                return f"{code}.{exchange}"
        
        return instrument
    
    @staticmethod
    def convert_index_to_qlib_format(index: pd.MultiIndex) -> pd.MultiIndex:
        """
        将现有MultiIndex转换为Qlib格式
        
        输入: MultiIndex (date, stock_code)
        输出: MultiIndex (datetime, instrument)
        """
        if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
            raise ValueError("Expected a two-level (date, stock_code) index")
        if not index.is_unique:
            raise ValueError(f"Duplicate input keys: {index[index.duplicated()].tolist()[:3]}")
        dates = pd.to_datetime(index.get_level_values(0))
        stock_codes = index.get_level_values(1)
        unique_codes = stock_codes.unique()
        mapping = {
            code: DataConverter._convert_stock_code_to_instrument(code)
            for code in unique_codes
        }
        if len(set(mapping.values())) != len(mapping):
            aliases = pd.Series(mapping)
            aliases = aliases[aliases.duplicated(keep=False)]
            raise ValueError(f"Stock code alias collision: {aliases.to_dict()}")
        instruments = stock_codes.map(mapping)
        
        new_index = pd.MultiIndex.from_arrays(
            [dates, instruments],
            names=['datetime', 'instrument']
        )
        if not new_index.is_unique:
            raise ValueError("Duplicate keys after converting dates and stock codes")
        return new_index
    
    @staticmethod
    def convert_index_from_qlib_format(
        qlib_index: pd.MultiIndex,
        original_index: Optional[pd.MultiIndex] = None
    ) -> pd.MultiIndex:
        """
        将Qlib格式MultiIndex转换回原始格式
        
        输入: MultiIndex (datetime, instrument)
        输出: MultiIndex (date, stock_code)
        """
        if not qlib_index.is_unique:
            raise ValueError("Duplicate Qlib prediction keys")
        if original_index is not None:
            converted = DataConverter.convert_index_to_qlib_format(original_index)
            positions = converted.get_indexer(qlib_index)
            if (positions < 0).any():
                missing = qlib_index[positions < 0].tolist()[:3]
                raise ValueError(f"Prediction keys missing from original index: {missing}")
            return original_index.take(positions)

        dates = qlib_index.get_level_values('datetime')
        instruments = qlib_index.get_level_values('instrument')
        
        stock_codes = [
            DataConverter._convert_instrument_to_stock_code(inst)
            for inst in instruments
        ]
        
        new_index = pd.MultiIndex.from_arrays(
            [dates, stock_codes],
            names=['date', 'stock_code']
        )
        return new_index
    
    @staticmethod
    def create_dataset_from_factors(
        factor_df: pd.DataFrame,
        forward_returns: pd.Series,
        train_start: str,
        train_end: str,
        valid_start: str,
        valid_end: str,
        test_start: str,
        test_end: str,
        config: Optional[dict] = None,
        logger: Optional[logging.Logger] = None
    ) -> 'FactorPilotDataset':
        """
        将现有因子数据转换为Qlib兼容的DatasetH格式
        
        Args:
            factor_df: pd.DataFrame
                - Index: MultiIndex (date: datetime, stock_code: str)
                - Columns: 因子名称列表
                - Values: float, 因子值
                
            forward_returns: pd.Series
                - Index: MultiIndex (date: datetime, stock_code: str)
                - Values: float, N日前瞻收益率
                
            train_start/end: str, 格式 "YYYY-MM-DD"
            valid_start/end: str, 格式 "YYYY-MM-DD"
            test_start/end: str, 格式 "YYYY-MM-DD"
            config: dict, 可选的配置参数
            logger: 日志记录器
            
        Returns:
            FactorPilotDataset: 兼容Qlib接口的数据集对象
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        # Validate before intersection/selection can hide or expand duplicate keys.
        DataConverter.convert_index_to_qlib_format(factor_df.index)
        DataConverter.convert_index_to_qlib_format(forward_returns.index)

        # 确保索引对齐
        common_index = factor_df.index.intersection(forward_returns.index)
        factor_df_aligned = factor_df.loc[common_index]
        forward_returns_aligned = forward_returns.loc[common_index]
        
        logger.info(
            f"[DataConverter] 数据对齐完成: {len(common_index)} 样本, "
            f"{len(factor_df.columns)} 个因子"
        )
        
        # 转换索引格式
        qlib_index = DataConverter.convert_index_to_qlib_format(common_index)
        
        # 构建特征DataFrame
        feature_df = factor_df_aligned.copy()
        feature_df.index = qlib_index
        
        # 构建标签DataFrame
        label_df = forward_returns_aligned.to_frame(name='LABEL0')
        label_df.index = qlib_index
        
        # 创建自定义数据集
        dataset = FactorPilotDataset(
            feature_df=feature_df,
            label_df=label_df,
            segments={
                'train': (train_start, train_end),
                'valid': (valid_start, valid_end),
                'test': (test_start, test_end),
            },
            original_index=common_index,
            logger=logger
        )
        
        return dataset
    
    @staticmethod
    def convert_predictions_to_series(
        predictions: pd.Series,
        original_index: Optional[pd.MultiIndex] = None
    ) -> pd.Series:
        """
        将Qlib预测结果转换回原始格式
        
        Args:
            predictions: pd.Series
                - Index: MultiIndex (datetime, instrument)
                - Values: float, 模型预测分数
                
            original_index: pd.MultiIndex, 可选
                - 原始数据的索引 (date, stock_code)
                
        Returns:
            pd.Series
                - Index: MultiIndex (date, stock_code)
                - Values: float, 预测分数
        """
        if predictions.empty:
            return predictions
        
        # 转换索引
        new_index = DataConverter.convert_index_from_qlib_format(
            predictions.index, original_index
        )
        
        result = pd.Series(
            predictions.values,
            index=new_index,
            name=predictions.name
        )
        
        return result


class FactorPilotDataset:
    """
    兼容Qlib DatasetH接口的自定义数据集
    
    与Qlib原生DatasetH的主要区别:
    1. 数据直接从内存中的DataFrame获取，而非Qlib数据目录
    2. 保留原始索引格式用于结果转换
    """
    
    def __init__(
        self,
        feature_df: pd.DataFrame,
        label_df: pd.DataFrame,
        segments: Dict[str, Tuple[str, str]],
        original_index: Optional[pd.MultiIndex] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化数据集
        
        Args:
            feature_df: 特征DataFrame, Index=(datetime, instrument)
            label_df: 标签DataFrame, Index=(datetime, instrument)
            segments: 数据分段定义 {"train": (start, end), ...}
            original_index: 原始数据索引
            logger: 日志记录器
        """
        self.feature_df = feature_df
        self.label_df = label_df
        self.segments = segments
        self.original_index = original_index
        self.logger = logger or logging.getLogger(__name__)
        
        # 缓存分段数据
        self._segment_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    
    def prepare(
        self,
        segments: Union[str, List[str]],
        col_set: Union[str, List[str]] = ["feature", "label"],
        data_key: Optional[str] = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
        """
        准备指定分段的数据，兼容Qlib DatasetH接口
        
        Args:
            segments: str 或 List[str], 如 "train", ["train", "valid"]
            col_set: str 或 List[str], "feature", "label", 或 ["feature", "label"]
            data_key: 保留参数，兼容Qlib接口
            
        Returns:
            如果segments是str: pd.DataFrame
            如果segments是list: Tuple[pd.DataFrame, ...]
        """
        if isinstance(segments, str):
            return self._prepare_single_segment(segments, col_set)
        else:
            return tuple(
                self._prepare_single_segment(seg, col_set)
                for seg in segments
            )
    
    def _prepare_single_segment(
        self,
        segment: str,
        col_set: Union[str, List[str]]
    ) -> pd.DataFrame:
        """准备单个分段的数据"""
        
        if segment not in self.segments:
            raise ValueError(f"未知的分段: {segment}, 可用: {list(self.segments.keys())}")
        
        start_date, end_date = self.segments[segment]
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 获取分段内的数据
        dates = self.feature_df.index.get_level_values('datetime')
        mask = (dates >= start_dt) & (dates <= end_dt)
        
        # 根据col_set返回数据
        if isinstance(col_set, str):
            col_set = [col_set]
        
        result_dfs = {}
        
        if 'feature' in col_set:
            result_dfs['feature'] = self.feature_df.loc[mask]
        
        if 'label' in col_set:
            if not self.feature_df.index.is_unique or not self.label_df.index.is_unique:
                raise ValueError("Feature and label indices must be unique for label requests")
            selected_index = self.feature_df.index[mask]
            if not selected_index.isin(self.label_df.index).all():
                raise ValueError("Selected feature keys are missing from labels")
            result_dfs['label'] = self.label_df.loc[selected_index]
        
        # 合并结果
        if len(result_dfs) == 1:
            return list(result_dfs.values())[0]
        else:
            return pd.concat(result_dfs, axis=1)
    
    def __repr__(self) -> str:
        n_samples = len(self.feature_df)
        n_features = len(self.feature_df.columns)
        segments_str = ", ".join(
            f"{k}: {v[0]} ~ {v[1]}" for k, v in self.segments.items()
        )
        return f"FactorPilotDataset(samples={n_samples}, features={n_features}, segments=[{segments_str}])"

