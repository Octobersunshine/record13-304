import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Tuple


SHAPIRO_WILK_MAX_SAMPLE_SIZE = 5000


class NormalityTest:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def _validate_data(self, data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)
        
        if data.ndim != 1:
            raise ValueError("数据必须是一维数组")
        
        if len(data) < 3:
            raise ValueError("样本量至少需要 3 个数据点")
        
        if np.isnan(data).any():
            raise ValueError("数据中包含 NaN 值")
        
        if np.isinf(data).any():
            raise ValueError("数据中包含无穷大值")

    def shapiro_wilk_test(self, data: np.ndarray) -> Dict[str, Any]:
        data = np.array(data, dtype=float)
        self._validate_data(data)
        
        if len(data) > SHAPIRO_WILK_MAX_SAMPLE_SIZE:
            raise ValueError(
                f"Shapiro-Wilk 检验适用于样本量 ≤ {SHAPIRO_WILK_MAX_SAMPLE_SIZE} 的数据"
            )
        
        stat, p_value = stats.shapiro(data)
        
        return {
            "test_name": "Shapiro-Wilk",
            "statistic": float(stat),
            "p_value": float(p_value),
            "alpha": self.alpha,
            "is_normal": p_value > self.alpha,
            "conclusion": (
                "数据服从正态分布" 
                if p_value > self.alpha 
                else "数据不服从正态分布"
            ),
            "sample_size": len(data)
        }

    def kolmogorov_smirnov_test(self, data: np.ndarray) -> Dict[str, Any]:
        data = np.array(data, dtype=float)
        self._validate_data(data)
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            raise ValueError("数据的标准差为 0，无法进行 KS 检验")
        
        stat, p_value = stats.kstest(data, 'norm', args=(mean, std))
        
        return {
            "test_name": "Kolmogorov-Smirnov",
            "statistic": float(stat),
            "p_value": float(p_value),
            "alpha": self.alpha,
            "is_normal": p_value > self.alpha,
            "conclusion": (
                "数据服从正态分布" 
                if p_value > self.alpha 
                else "数据不服从正态分布"
            ),
            "sample_size": len(data),
            "estimated_mean": float(mean),
            "estimated_std": float(std)
        }

    def auto_test(self, data: np.ndarray) -> Dict[str, Any]:
        data = np.array(data, dtype=float)
        self._validate_data(data)
        
        n = len(data)
        
        if n <= SHAPIRO_WILK_MAX_SAMPLE_SIZE:
            result = self.shapiro_wilk_test(data)
            result["auto_selected"] = True
            result["selection_reason"] = (
                f"样本量 {n} ≤ {SHAPIRO_WILK_MAX_SAMPLE_SIZE}，"
                f"自动选择 Shapiro-Wilk 检验"
            )
        else:
            result = self.kolmogorov_smirnov_test(data)
            result["auto_selected"] = True
            result["selection_reason"] = (
                f"样本量 {n} > {SHAPIRO_WILK_MAX_SAMPLE_SIZE}，"
                f"Shapiro-Wilk 检验不可用，自动切换至 Kolmogorov-Smirnov 检验"
            )
        
        return result

    def comprehensive_test(self, data: np.ndarray) -> Dict[str, Any]:
        data = np.array(data, dtype=float)
        self._validate_data(data)
        
        n = len(data)
        results = {}
        
        if n <= SHAPIRO_WILK_MAX_SAMPLE_SIZE:
            results["shapiro_wilk"] = self.shapiro_wilk_test(data)
        else:
            results["shapiro_wilk"] = {
                "skipped": True,
                "reason": (
                    f"样本量 {n} > {SHAPIRO_WILK_MAX_SAMPLE_SIZE}，"
                    f"Shapiro-Wilk 检验不可用，已自动跳过"
                )
            }
        
        results["kolmogorov_smirnov"] = self.kolmogorov_smirnov_test(data)
        
        sw_result = results["shapiro_wilk"]
        ks_result = results["kolmogorov_smirnov"]
        
        sw_available = not sw_result.get("skipped", False)
        
        if sw_available:
            overall_normal = sw_result["is_normal"] and ks_result["is_normal"]
            overall_conclusion = (
                "综合两种检验，数据服从正态分布"
                if overall_normal
                else "综合两种检验，数据不服从正态分布"
            )
        else:
            overall_normal = ks_result["is_normal"]
            overall_conclusion = ks_result["conclusion"]
        
        return {
            "tests": results,
            "overall": {
                "is_normal": overall_normal,
                "conclusion": overall_conclusion,
                "sample_size": n,
                "descriptive_stats": {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data, ddof=1)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "median": float(np.median(data)),
                    "skewness": float(stats.skew(data)),
                    "kurtosis": float(stats.kurtosis(data))
                }
            }
        }
