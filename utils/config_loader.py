import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .dynamic_stock_pool import StockFilter

logger = logging.getLogger(__name__)


# =============================================================================
# PIT Configuration Loading
# =============================================================================

def load_pit_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and validate PIT adjustment configuration from config dict.
    
    Uses factor_end method for PIT adjustment, which is mathematically 
    equivalent to JoinQuant fq='pre' for price ratios.
    
    Formula: adjusted_price = raw_price × (factor / factor_end)
    Where factor_end = last day's factor for each stock in the data window.
    
    Args:
        config: Full configuration dictionary that may contain 'pit_adjustment' section.
        
    Returns:
        Dictionary with PIT configuration:
        - enabled: bool (default True)
        
    Example config section:
        {
            "pit_adjustment": {
                "enabled": true
            }
        }
    """
    pit_section = config.get('pit_adjustment', {})
    
    # Simplified configuration - only enabled flag needed
    # PIT adjustment uses factor_end method internally
    pit_config = {
        'enabled': pit_section.get('enabled', True),  # Default enabled
    }
    
    # Log configuration
    if pit_config['enabled']:
        logger.info("PIT adjustment enabled (factor_end method)")
    else:
        logger.debug("PIT adjustment disabled")
    
    return pit_config

def load_dynamic_pool_config(config_path: str) -> Dict:
    """Load dynamic pool configuration from JSON file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def json_to_stock_filters(filter_configs: List[Dict]) -> List[StockFilter]:
    """Convert JSON filter configurations to StockFilter objects
    
    Supported operators:
        - Standard: '>', '<', '>=', '<=', '==', '!=', 'between', 'in', 'not_in'
        - Percentile: 'percentile_bottom', 'percentile_top', 'percentile_between'
          - percentile_bottom: value is single number (e.g., 10 for bottom 10%)
          - percentile_top: value is single number (e.g., 10 for top 10%)
          - percentile_between: value is [low_pct, high_pct] (e.g., [10, 30])
    """
    filters = []
    
    for filter_config in filter_configs:
        try:
            field = filter_config['field']
            operator = filter_config['operator']
            value = filter_config['value']
            description = filter_config.get('description', '')
            
            # Handle tuple values for 'between' and 'percentile_between' operators
            if operator in ('between', 'percentile_between') and isinstance(value, list):
                value = tuple(value)
            
            filter_obj = StockFilter(field, operator, value, description)
            filters.append(filter_obj)
            
            # Log percentile filters for debugging
            if operator.startswith('percentile_'):
                logger.debug(f"Loaded percentile filter: {field} {operator} {value}")
            
        except KeyError as e:
            logger.error(f"Missing required field in filter config: {e}")
            continue
        except Exception as e:
            logger.error(f"Error creating filter from config {filter_config}: {e}")
            continue
    
    return filters

def create_pool_configs_from_json(json_config: Dict) -> Dict[str, Dict]:
    """Create pool configurations with StockFilter objects from JSON config"""
    pool_configs = {}
    
    dynamic_pools = json_config.get('dynamic_pools', {})
    
    for pool_name, pool_config in dynamic_pools.items():
        if not pool_config.get('enabled', True):
            continue
            
        try:
            # Convert filter configs to StockFilter objects
            filter_configs = pool_config.get('filters', [])
            stock_filters = json_to_stock_filters(filter_configs)
            
            # Create pool configuration
            pool_configs[pool_name] = {
                'description': pool_config.get('description', ''),
                'filters': stock_filters,
                'max_stocks': pool_config.get('max_stocks'),
                'sort_by': pool_config.get('sort_by'),
                'ascending': pool_config.get('ascending', False),
                'rebalance_period_months': pool_config.get('rebalance_period_months', 1),
                'rebalance_schedule': pool_config.get('rebalance_schedule', 'quarterly')
            }
            
            logger.info(f"Loaded pool config '{pool_name}' with {len(stock_filters)} filters")
            
        except Exception as e:
            logger.error(f"Error processing pool config '{pool_name}': {e}")
            continue
    
    return pool_configs

def load_and_parse_config(config_path: str) -> tuple:
    """Load and parse complete configuration
    
    Returns:
        tuple: (database_config, analysis_config, pool_configs)
    """
    json_config = load_dynamic_pool_config(config_path)
    
    from utils.data_store import resolve_database_config
    json_config = resolve_database_config(json_config)
    database_config = json_config.get('database', {})
    analysis_config = json_config.get('analysis_settings', {})
    if database_config.get('data_config_path'):
        database_config['data_config_path'] = str(
            (Path(config_path).resolve().parent / Path(database_config['data_config_path']).expanduser()).resolve())
    for key in ('data_config_path', 'source_policy'):
        analysis_config[key] = database_config.get(key)
    pool_configs = create_pool_configs_from_json(json_config)
    
    # Add analysis period to analysis config
    analysis_period = json_config.get('analysis_period', {})
    analysis_config.update(analysis_period)
    
    # Add results database configuration to analysis config
    results_database = json_config.get('results_database', {})
    if results_database:
        analysis_config['results_output_table'] = results_database.get('output_table', 'factor_analysis_results')
        analysis_config['results_monthly_output_table'] = results_database.get('monthly_output_table', 'monthly_pool_results')
        
        # DuckDB configuration for results database
        backend = str(results_database.get('db_backend', 'auto') or 'auto').lower()
        if backend == 'sqlite':
            raise ValueError("results_database SQLite backend is no longer supported; configure DuckDB")
        if backend not in {'auto', 'duckdb'}:
            raise ValueError("results_database.db_backend must be 'duckdb' or 'auto'")
        duckdb_path = results_database.get('duckdb_path')
        if not duckdb_path:
            raise ValueError("results_database.duckdb_path is required for DuckDB results")
        analysis_config['results_db_backend'] = 'duckdb'
        analysis_config['results_duckdb_path'] = duckdb_path
        analysis_config['results_parquet_dir'] = results_database.get('parquet_dir')
        analysis_config['results_export_parquet'] = results_database.get('export_parquet', True)
    
    # Add database safety settings
    database_safety = json_config.get('database_safety', {})
    if database_safety:
        analysis_config.update(database_safety)
    
    return database_config, analysis_config, pool_configs

# Example usage function
def example_usage():
    """Example of how to use the config loader"""
    try:
        # Load configuration
        db_config, analysis_config, pool_configs = load_and_parse_config('config/dynamic_pool_config.json')
        
        print("Database config:", db_config)
        print("Analysis config:", analysis_config)
        print("Pool configs:")
        for name, config in pool_configs.items():
            print(f"  {name}: {len(config['filters'])} filters, max_stocks={config['max_stocks']}")
            
        return db_config, analysis_config, pool_configs
        
    except Exception as e:
        logger.error(f"Example usage failed: {e}")
        return None, None, None

if __name__ == "__main__":
    example_usage()
