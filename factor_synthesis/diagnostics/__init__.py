"""
多重共线性诊断模块

用于LASSO因子合成的稳定性和共线性分析。
在滚动窗口训练完成后、回测前执行诊断，检测因子选择的一致性问题。

主要功能：
1. 窗口间特征选择稳定性分析（Jaccard相似度）
2. 多重共线性检测（VIF、条件数）
3. 权重稳定性分析（变异系数）
4. 诊断报告生成

Author: QuantationAI
Date: 2026-02-07
"""

import logging
from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .stability_analyzer import (
    compute_jaccard_stability,
    compute_weight_variability,
    identify_unstable_factors,
    STABILITY_SWITCH_VERSION,
)
from .multicollinearity_checker import (
    compute_vif_metrics,
    compute_condition_number,
    compute_factor_correlation_matrix,
)


def run_multicollinearity_diagnostics(
    lasso_weight_rows: List[pd.Series],
    training_factor_names: List[str],
    factor_df: pd.DataFrame,
    config: dict,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    执行LASSO因子选择的多重共线性和稳定性诊断
    
    Args:
        lasso_weight_rows: 每个窗口的LASSO权重Series列表
        training_factor_names: 训练因子名称列表
        factor_df: 原始因子数据DataFrame (date, stock_code索引)
        config: 实验配置字典
        logger: 日志记录器
        
    Returns:
        诊断结果字典，包含：
        - jaccard_mean: 平均Jaccard相似度
        - vif_metrics: VIF指标字典 {factor: vif_value}
        - condition_number: 条件数
        - weight_cv: 权重变异系数字典 {factor: cv}
        - diagnostics_passed: bool，是否通过诊断
        - recommendations: 改进建议列表
    """
    logger.info("=" * 80)
    logger.info("开始LASSO多重共线性和稳定性诊断")
    logger.info("=" * 80)
    
    # 获取诊断配置
    diag_config = config.get('multicollinearity_diagnostics', {})
    if not diag_config.get('enabled', False):
        logger.info("多重共线性诊断未启用，跳过分析")
        return {}
    
    # 提取阈值
    jaccard_threshold = diag_config.get('jaccard_threshold', 0.5)
    vif_threshold = diag_config.get('vif_threshold', 10.0)
    cond_threshold = diag_config.get('condition_number_threshold', 30.0)
    cv_threshold = diag_config.get('weight_cv_threshold', 2.0)
    max_samples_vif = diag_config.get('max_samples_for_vif', 50000)
    save_report = diag_config.get('save_detailed_report', True)
    
    # 数据验证
    if len(lasso_weight_rows) < 2:
        logger.warning(f"滚动窗口数量不足（{len(lasso_weight_rows)}），至少需要2个窗口进行诊断")
        return {}
    
    if factor_df.empty:
        logger.warning("因子数据为空，无法执行诊断")
        return {}
    
    logger.info(f"诊断参数: {len(lasso_weight_rows)}个窗口, {len(training_factor_names)}个训练因子")
    
    # 初始化结果
    result = {
        'jaccard_mean': np.nan,
        'jaccard_series': [],
        'vif_metrics': {},
        'condition_number': np.nan,
        'weight_cv': {},
        'zero_nonzero_switches': {},
        'diagnostics_passed': True,
        'stability_switch_version': STABILITY_SWITCH_VERSION,
        'issues_found': [],
        'recommendations': []
    }
    
    try:
        # ===================================================================
        # 1. 窗口间特征选择稳定性分析（Jaccard相似度）
        # ===================================================================
        logger.info("\n[1/4] 计算窗口间Jaccard相似度...")
        jaccard_mean, jaccard_series = compute_jaccard_stability(
            lasso_weight_rows, logger
        )
        result['jaccard_mean'] = jaccard_mean
        result['jaccard_series'] = jaccard_series
        
        if jaccard_mean < jaccard_threshold:
            issue = f"特征选择不稳定: 平均Jaccard={jaccard_mean:.3f} < {jaccard_threshold}"
            result['issues_found'].append(issue)
            result['diagnostics_passed'] = False
            logger.warning(f"⚠️  {issue}")
        else:
            logger.info(f"✅ 特征选择稳定: 平均Jaccard={jaccard_mean:.3f}")
        
        # ===================================================================
        # 2. 权重稳定性分析（变异系数）
        # ===================================================================
        logger.info("\n[2/4] 计算权重变异系数...")
        weight_cv, switches = compute_weight_variability(
            lasso_weight_rows, training_factor_names, logger
        )
        result['weight_cv'] = weight_cv
        result['zero_nonzero_switches'] = switches
        
        # 识别不稳定因子
        unstable_factors = identify_unstable_factors(
            weight_cv, switches, cv_threshold, logger, n_windows=len(lasso_weight_rows)
        )
        if unstable_factors:
            issue = f"检测到{len(unstable_factors)}个权重不稳定因子"
            result['issues_found'].append(issue)
            result['diagnostics_passed'] = False
            logger.warning(f"⚠️  {issue}: {unstable_factors[:5]}")  # 只显示前5个
        else:
            logger.info(f"✅ 所有因子权重稳定（CV < {cv_threshold}）")
        
        # ===================================================================
        # 3. 多重共线性检测 - VIF
        # ===================================================================
        logger.info("\n[3/4] 计算方差膨胀因子（VIF）...")
        
        # 提取最后一个窗口的非零权重因子
        last_weights = lasso_weight_rows[-1]
        selected_factors = last_weights[last_weights != 0].index.tolist()
        
        if len(selected_factors) < 2:
            logger.info("选中因子少于2个，跳过VIF计算")
        else:
            vif_metrics = compute_vif_metrics(
                factor_df, selected_factors, max_samples_vif, logger
            )
            result['vif_metrics'] = vif_metrics
            
            high_vif = {f: v for f, v in vif_metrics.items() if v > vif_threshold}
            if high_vif:
                issue = f"检测到{len(high_vif)}个高VIF因子（>{vif_threshold}）"
                result['issues_found'].append(issue)
                result['diagnostics_passed'] = False
                logger.warning(f"⚠️  {issue}")
                for f, v in list(high_vif.items())[:5]:
                    logger.warning(f"     {f}: VIF={v:.2f}")
            else:
                logger.info(f"✅ 所有因子VIF < {vif_threshold}")
        
        # ===================================================================
        # 4. 多重共线性检测 - 条件数
        # ===================================================================
        logger.info("\n[4/4] 计算协方差矩阵条件数...")
        
        if len(selected_factors) >= 2:
            cond_number = compute_condition_number(
                factor_df, selected_factors, max_samples_vif, logger
            )
            result['condition_number'] = cond_number
            
            if cond_number > cond_threshold:
                issue = f"协方差矩阵病态: 条件数={cond_number:.1f} > {cond_threshold}"
                result['issues_found'].append(issue)
                result['diagnostics_passed'] = False
                logger.warning(f"⚠️  {issue}")
            else:
                logger.info(f"✅ 条件数正常: {cond_number:.1f} < {cond_threshold}")
        
        # ===================================================================
        # 5. 生成改进建议
        # ===================================================================
        if not result['diagnostics_passed']:
            result['recommendations'] = _generate_recommendations(result, diag_config)
            logger.info("\n" + "=" * 80)
            logger.info("📋 改进建议:")
            for i, rec in enumerate(result['recommendations'], 1):
                logger.info(f"  {i}. {rec}")
        else:
            logger.info("\n" + "=" * 80)
            logger.info("✅ 所有诊断指标通过，因子选择稳定且无严重共线性")
        
        # ===================================================================
        # 6. 保存详细报告
        # ===================================================================
        if save_report:
            _save_diagnostic_report(result, config, logger)
        
        logger.info("=" * 80)
        
    except LabelIsolationError:
        raise
    except Exception as e:
        logger.error(f"诊断过程发生异常: {e}", exc_info=True)
        result['diagnostics_passed'] = False
        result['issues_found'].append(f"诊断异常: {str(e)}")
    
    return result


def _generate_recommendations(result: Dict[str, Any], config: dict) -> List[str]:
    """根据诊断结果生成改进建议"""
    recommendations = []
    
    # Jaccard低 -> 特征选择不稳定
    if result['jaccard_mean'] < config.get('jaccard_threshold', 0.5):
        recommendations.append(
            "使用Stability Selection或Randomized Lasso提升特征选择稳定性"
        )
        recommendations.append(
            "考虑增大alpha正则化强度，减少选中因子数量"
        )
    
    # 高VIF -> 强共线性
    if result['vif_metrics'] and max(result['vif_metrics'].values(), default=0) > config.get('vif_threshold', 10):
        recommendations.append(
            "检测到强共线性，建议使用Elastic Net（l1_ratio=0.5-0.7）替代纯LASSO"
        )
        recommendations.append(
            "考虑使用层次聚类预筛选（hierarchical_clustering）移除冗余因子"
        )
    
    # 高条件数 -> 矩阵病态
    if result['condition_number'] > config.get('condition_number_threshold', 30):
        recommendations.append(
            "协方差矩阵病态，建议使用Ridge Regression或增大ElasticNet的l2惩罚"
        )
    
    # 权重CV高 -> 权重不稳定
    if result['weight_cv'] and max(result['weight_cv'].values(), default=0) > config.get('weight_cv_threshold', 2.0):
        recommendations.append(
            "权重波动过大，建议启用sample_weight_decay降低旧样本权重"
        )
        recommendations.append(
            "考虑使用更长的训练窗口（train_window_days）提升模型稳定性"
        )
    
    return recommendations


def _save_diagnostic_report(result: Dict[str, Any], config: dict, logger: logging.Logger):
    """保存详细诊断报告到CSV"""
    try:
        import os
        from datetime import datetime
        
        output_dir = config.get('output_settings', {}).get('weights_output_dir', './results')
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取实验名称
        synthesis_methods = config.get('synthesis_methods', {})
        exp_name = list(synthesis_methods.keys())[0] if synthesis_methods else 'unknown'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(output_dir, f'lasso_diagnostics_{exp_name}_{timestamp}.csv')
        
        # 构造报告DataFrame
        rows = []
        
        # 1. Jaccard相似度
        rows.append({
            'metric_type': 'jaccard_stability',
            'metric_name': 'mean_jaccard',
            'value': result['jaccard_mean'],
            'threshold': config.get('multicollinearity_diagnostics', {}).get('jaccard_threshold', 0.5),
            'passed': result['jaccard_mean'] >= config.get('multicollinearity_diagnostics', {}).get('jaccard_threshold', 0.5)
        })
        
        # 2. VIF
        for factor, vif in result.get('vif_metrics', {}).items():
            rows.append({
                'metric_type': 'vif',
                'metric_name': factor,
                'value': vif,
                'threshold': config.get('multicollinearity_diagnostics', {}).get('vif_threshold', 10.0),
                'passed': vif <= config.get('multicollinearity_diagnostics', {}).get('vif_threshold', 10.0)
            })
        
        # 3. 条件数
        rows.append({
            'metric_type': 'condition_number',
            'metric_name': 'covariance_condition',
            'value': result['condition_number'],
            'threshold': config.get('multicollinearity_diagnostics', {}).get('condition_number_threshold', 30.0),
            'passed': result['condition_number'] <= config.get('multicollinearity_diagnostics', {}).get('condition_number_threshold', 30.0)
        })
        
        # 4. 权重CV
        for factor, cv in result.get('weight_cv', {}).items():
            rows.append({
                'metric_type': 'weight_cv',
                'metric_name': factor,
                'value': cv,
                'threshold': config.get('multicollinearity_diagnostics', {}).get('weight_cv_threshold', 2.0),
                'passed': cv <= config.get('multicollinearity_diagnostics', {}).get('weight_cv_threshold', 2.0)
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(report_path, index=False, encoding='utf-8')
        
        logger.info(f"📄 详细诊断报告已保存: {report_path}")
        
    except LabelIsolationError:
        raise
    except Exception as e:
        logger.warning(f"保存诊断报告失败: {e}")


__all__ = [
    'run_multicollinearity_diagnostics',
    'compute_jaccard_stability',
    'compute_weight_variability',
    'identify_unstable_factors',
    'compute_vif_metrics',
    'compute_condition_number',
    'compute_factor_correlation_matrix',
]
