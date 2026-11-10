# -*- coding: utf-8 -*-
"""
策略注册表

支持通过装饰器注册预筛选策略，并提供工厂函数动态创建策略实例。
"""

from typing import Dict, Type, List, Any, Optional
import logging

from .base import PreSelectionStrategy


# 全局策略注册表
_STRATEGY_REGISTRY: Dict[str, Type[PreSelectionStrategy]] = {}


def register_strategy(cls: Type[PreSelectionStrategy]) -> Type[PreSelectionStrategy]:
    """
    装饰器：注册预筛选策略到全局注册表
    
    Usage:
        >>> @register_strategy
        ... class ICIRPreSelectionStrategy(PreSelectionStrategy):
        ...     name = "ic_ir"
        ...     def select(self, X, y):
        ...         ...
    
    Args:
        cls: 策略类（必须继承 PreSelectionStrategy）
        
    Returns:
        原始类（用于装饰器链）
        
    Raises:
        TypeError: 如果类未继承 PreSelectionStrategy
        ValueError: 如果策略名称重复
    """
    if not issubclass(cls, PreSelectionStrategy):
        raise TypeError(f"{cls.__name__} 必须继承 PreSelectionStrategy")
    
    strategy_name = getattr(cls, 'name', None)
    if not strategy_name or strategy_name == 'base':
        raise ValueError(f"{cls.__name__} 必须定义有效的 'name' 类属性")
    
    if strategy_name in _STRATEGY_REGISTRY:
        existing_cls = _STRATEGY_REGISTRY[strategy_name]
        if existing_cls is not cls:
            raise ValueError(
                f"策略名称 '{strategy_name}' 已被 {existing_cls.__name__} 注册，"
                f"无法再注册 {cls.__name__}"
            )
    
    _STRATEGY_REGISTRY[strategy_name] = cls
    return cls


def get_preselection_strategy(
    name: str, 
    config: Dict[str, Any], 
    logger: Optional[logging.Logger] = None
) -> PreSelectionStrategy:
    """
    工厂函数：根据策略名称创建策略实例
    
    Args:
        name: 策略名称（如 "ic_ir", "mutual_info" 等）
        config: 策略配置字典
        logger: 日志记录器
        
    Returns:
        PreSelectionStrategy: 策略实例
        
    Raises:
        ValueError: 如果策略名称未注册
    """
    if name not in _STRATEGY_REGISTRY:
        available = list(_STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"未知的预筛选策略: '{name}'。可用策略: {available}"
        )
    
    strategy_cls = _STRATEGY_REGISTRY[name]
    return strategy_cls(config, logger)


def list_strategies() -> List[str]:
    """
    列出所有已注册的策略名称
    
    Returns:
        List[str]: 策略名称列表
    """
    return list(_STRATEGY_REGISTRY.keys())


def get_strategy_class(name: str) -> Type[PreSelectionStrategy]:
    """
    获取策略类（不创建实例）
    
    Args:
        name: 策略名称
        
    Returns:
        策略类
        
    Raises:
        ValueError: 如果策略名称未注册
    """
    if name not in _STRATEGY_REGISTRY:
        raise ValueError(f"未知的预筛选策略: '{name}'")
    return _STRATEGY_REGISTRY[name]
