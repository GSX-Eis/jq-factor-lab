import hashlib
import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple
from utils.factor_base import FactorBase
from utils.numba_compat import SERIAL_NUMBA, normalize_numba_imports

class FactorLoader:
    """动态加载因子文件的类"""

    @staticmethod
    def _module_name_for_path(file_path: Path) -> str:
        """Build a unique module name per file path to avoid collisions."""
        resolved_path = file_path.resolve()
        path_digest = hashlib.sha1(str(resolved_path).encode("utf-8")).hexdigest()[:12]
        sanitized_stem = "".join(
            ch if ch.isalnum() else "_" for ch in resolved_path.stem
        )
        return f"_factor_loader_{sanitized_stem}_{path_digest}"

    @staticmethod
    def _load_module_from_path(file_path: Path, *, serial: bool = False) -> Tuple[str, Any]:
        """Load a factor module and register it before execution.

        Registering the module in ``sys.modules`` before ``exec_module`` keeps
        recursive Numba JIT helpers resolvable under dynamic loading.
        """
        module_name = FactorLoader._module_name_for_path(file_path)
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to create import spec for factor file: {file_path}")

        sys.modules.pop(module_name, None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            source = file_path.read_text(encoding="utf-8")
            if serial:
                module.__dict__.update(numba=SERIAL_NUMBA, IS_JOINQUANT=True)
            tree = normalize_numba_imports(source, inline=serial)
            exec(compile(tree, str(file_path), "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        return module_name, module

    @staticmethod
    def _get_factor_classes_from_module(module_name: str, module: Any) -> List[type]:
        """Extract distinct FactorBase classes defined here, ignoring class aliases."""
        classes: List[type] = []
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj.__module__ == module_name
                and issubclass(obj, FactorBase)
                and obj is not FactorBase
                and all(obj is not existing for existing in classes)
            ):
                classes.append(obj)
                logging.getLogger(__name__).info(f"Loaded factor class: {obj.__name__}")
        return classes
    
    @staticmethod
    def load_all_factors(factor_dir: str, db: Any, start_date: str, end_date: str, 
                        code_list: List[str], cache_dir: str, **factor_kwargs) -> List:
        """
        扫描指定目录，动态加载并实例化所有因子
        
        Args:
            factor_dir: 因子文件目录
            db: 数据库连接
            start_date: 开始日期
            end_date: 结束日期
            code_list: 股票代码列表
            cache_dir: 缓存目录
            factor_kwargs: 原样传入因子构造函数的数据来源等额外参数
            
        Returns:
            List of instantiated factor objects
        """
        factors = []
        
        # 确保目录存在
        if not os.path.exists(factor_dir):
            raise FileNotFoundError(f"Factor directory {factor_dir} does not exist")
        
        # 遍历目录中的所有 .py 文件
        for filename in sorted(os.listdir(factor_dir)):
            if filename.endswith('.py') and not filename.startswith('__'):
                file_path = os.path.join(factor_dir, filename)
                if not os.path.isfile(file_path):
                    continue
                
                # 动态加载模块
                module_name, module = FactorLoader._load_module_from_path(Path(file_path))
                
                # 查找模块中的所有类
                for obj in FactorLoader._get_factor_classes_from_module(module_name, module):
                    try:
                        # 尝试实例化因子类
                        factor_instance = obj(db, obj.__name__, start_date, end_date, code_list, cache_dir=cache_dir, **factor_kwargs)
                        factors.append(factor_instance)
                        logging.getLogger(__name__).info(f"Loaded factor: {obj.__name__}")
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to instantiate factor {obj.__name__!r} from {file_path}"
                        ) from e
        
        return factors

    @staticmethod
    def load_factor_classes(factor_dir: str) -> List[type]:
        """
        扫描指定目录，动态加载所有因子类（不实例化）
        
        Args:
            factor_dir: 因子文件目录
            
        Returns:
            List of factor classes

        Raises:
            ImportError: A selected module could not be loaded; no partial list is returned.
        """
        factor_classes = []
        
        # 确保目录存在
        if not os.path.exists(factor_dir):
            raise FileNotFoundError(f"Factor directory {factor_dir} does not exist")
        
        # 遍历目录中的所有 .py 文件
        for filename in sorted(os.listdir(factor_dir)):
            if filename.endswith('.py') and not filename.startswith('__'):
                file_path = os.path.join(factor_dir, filename)
                if not os.path.isfile(file_path):
                    continue
                logging.getLogger(__name__).info(f"Loading factor class from {file_path}")
                try:
                    # 动态加载模块
                    module_name, module = FactorLoader._load_module_from_path(Path(file_path))
                    
                    # 查找模块中的所有类
                    factor_classes.extend(
                        FactorLoader._get_factor_classes_from_module(module_name, module)
                    )
                            
                except Exception as e:
                    raise ImportError(f"Failed to load factor module {file_path}") from e
        
        return factor_classes

    @staticmethod
    def _load_classes_from_file(file_path: str, class_name: str = None) -> List[type]:
        """Load all FactorBase subclasses from a single .py file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Factor file not found: {file_path}")
        if not p.suffix == '.py':
            raise ValueError(f"Expected a .py file, got: {file_path}")

        module_name, module = FactorLoader._load_module_from_path(p)
        classes = FactorLoader._get_factor_classes_from_module(module_name, module)
        if class_name is not None:
            classes = [cls for cls in classes if cls.__name__ == class_name]
            if len(classes) != 1:
                raise ValueError(f"Expected one factor class {class_name!r} in {file_path}")
        return classes

    @staticmethod
    def load_factor_class_from_file(file_path: str, class_name: str = None) -> List[type]:
        """Load factor classes (without instantiation) from a single .py file.

        Returns a list for API consistency with load_factor_classes().
        """
        return FactorLoader._load_classes_from_file(file_path, class_name)

    @staticmethod
    def load_factor_from_file(
        file_path: str,
        db: Any,
        start_date: str,
        end_date: str,
        code_list: List[str],
        cache_dir: str,
        class_name: str = None,
        **factor_kwargs,
    ) -> List:
        """Load and instantiate factors from a single .py file.

        Returns a list for API consistency with load_all_factors().
        Extra keyword arguments are forwarded to each factor constructor.
        Construction failures raise with file/class context; no partial list is returned.
        """
        classes = FactorLoader._load_classes_from_file(file_path, class_name)
        factors: List = []
        for cls in classes:
            try:
                instance = cls(db, cls.__name__, start_date, end_date, code_list, cache_dir=cache_dir, **factor_kwargs)
                factors.append(instance)
                logging.getLogger(__name__).info(f"Loaded factor: {cls.__name__}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to instantiate factor {cls.__name__!r} from {file_path}"
                ) from e
        return factors
