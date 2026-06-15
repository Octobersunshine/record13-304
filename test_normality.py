import numpy as np
import unittest
from normality_test import NormalityTest, SHAPIRO_WILK_MAX_SAMPLE_SIZE


class TestNormalityTest(unittest.TestCase):
    def setUp(self):
        self.tester = NormalityTest(alpha=0.05)
        np.random.seed(42)

    def test_normal_distribution_shapiro(self):
        data = np.random.normal(0, 1, 30)
        result = self.tester.shapiro_wilk_test(data)
        self.assertEqual(result["test_name"], "Shapiro-Wilk")
        self.assertEqual(result["sample_size"], 30)
        self.assertTrue(result["is_normal"])
        self.assertIn("服从正态分布", result["conclusion"])

    def test_normal_distribution_ks(self):
        data = np.random.normal(0, 1, 100)
        result = self.tester.kolmogorov_smirnov_test(data)
        self.assertEqual(result["test_name"], "Kolmogorov-Smirnov")
        self.assertEqual(result["sample_size"], 100)
        self.assertTrue(result["is_normal"])
        self.assertIn("estimated_mean", result)
        self.assertIn("estimated_std", result)

    def test_non_normal_distribution_shapiro(self):
        data = np.random.exponential(1, 30)
        result = self.tester.shapiro_wilk_test(data)
        self.assertFalse(result["is_normal"])
        self.assertIn("不服从正态分布", result["conclusion"])

    def test_non_normal_distribution_ks(self):
        data = np.random.exponential(1, 100)
        result = self.tester.kolmogorov_smirnov_test(data)
        self.assertFalse(result["is_normal"])

    def test_auto_test_within_shapiro_limit(self):
        data = np.random.normal(0, 1, 100)
        result = self.tester.auto_test(data)
        self.assertEqual(result["test_name"], "Shapiro-Wilk")
        self.assertTrue(result["auto_selected"])
        self.assertIn("自动选择 Shapiro-Wilk 检验", result["selection_reason"])

    def test_auto_test_exceeds_shapiro_limit(self):
        data = np.random.normal(0, 1, 6000)
        result = self.tester.auto_test(data)
        self.assertEqual(result["test_name"], "Kolmogorov-Smirnov")
        self.assertTrue(result["auto_selected"])
        self.assertIn("Shapiro-Wilk 检验不可用，自动切换至", result["selection_reason"])

    def test_auto_test_at_shapiro_boundary(self):
        data = np.random.normal(0, 1, 5000)
        result = self.tester.auto_test(data)
        self.assertEqual(result["test_name"], "Shapiro-Wilk")
        self.assertTrue(result["auto_selected"])

    def test_auto_test_just_above_shapiro_boundary(self):
        data = np.random.normal(0, 1, 5001)
        result = self.tester.auto_test(data)
        self.assertEqual(result["test_name"], "Kolmogorov-Smirnov")
        self.assertIn("Shapiro-Wilk 检验不可用", result["selection_reason"])

    def test_comprehensive_test(self):
        data = np.random.normal(0, 1, 30)
        result = self.tester.comprehensive_test(data)
        self.assertIn("tests", result)
        self.assertIn("shapiro_wilk", result["tests"])
        self.assertIn("kolmogorov_smirnov", result["tests"])
        self.assertIn("overall", result)
        self.assertIn("descriptive_stats", result["overall"])
        stats_keys = ["mean", "std", "min", "max", "median", "skewness", "kurtosis"]
        for key in stats_keys:
            self.assertIn(key, result["overall"]["descriptive_stats"])

    def test_comprehensive_test_large_sample_within_limit(self):
        data = np.random.normal(0, 1, 100)
        result = self.tester.comprehensive_test(data)
        self.assertIn("shapiro_wilk", result["tests"])
        self.assertIn("kolmogorov_smirnov", result["tests"])
        self.assertNotIn("skipped", result["tests"]["shapiro_wilk"])

    def test_comprehensive_test_exceeds_shapiro_limit(self):
        data = np.random.normal(0, 1, 6000)
        result = self.tester.comprehensive_test(data)
        sw = result["tests"]["shapiro_wilk"]
        self.assertTrue(sw.get("skipped", False))
        self.assertIn("Shapiro-Wilk 检验不可用", sw["reason"])
        self.assertIn("kolmogorov_smirnov", result["tests"])
        self.assertEqual(result["overall"]["is_normal"], result["tests"]["kolmogorov_smirnov"]["is_normal"])

    def test_insufficient_data(self):
        with self.assertRaises(ValueError) as context:
            self.tester.shapiro_wilk_test([1, 2])
        self.assertIn("至少需要 3 个数据点", str(context.exception))

    def test_nan_data(self):
        with self.assertRaises(ValueError) as context:
            self.tester.shapiro_wilk_test([1, 2, np.nan, 4, 5])
        self.assertIn("包含 NaN 值", str(context.exception))

    def test_inf_data(self):
        with self.assertRaises(ValueError) as context:
            self.tester.shapiro_wilk_test([1, 2, np.inf, 4, 5])
        self.assertIn("包含无穷大值", str(context.exception))

    def test_zero_std_data(self):
        with self.assertRaises(ValueError) as context:
            self.tester.kolmogorov_smirnov_test([5, 5, 5, 5, 5])
        self.assertIn("标准差为 0", str(context.exception))

    def test_shapiro_sample_too_large(self):
        data = np.random.normal(0, 1, 6000)
        with self.assertRaises(ValueError) as context:
            self.tester.shapiro_wilk_test(data)
        self.assertIn(f"≤ {SHAPIRO_WILK_MAX_SAMPLE_SIZE}", str(context.exception))

    def test_custom_alpha(self):
        tester = NormalityTest(alpha=0.01)
        data = np.random.normal(0, 1, 30)
        result = tester.shapiro_wilk_test(data)
        self.assertEqual(result["alpha"], 0.01)

    def test_list_input(self):
        data = [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1]
        result = self.tester.shapiro_wilk_test(data)
        self.assertEqual(result["sample_size"], 10)

    def test_qq_plot_basic_structure(self):
        data = np.random.normal(0, 1, 30)
        result = self.tester.qq_plot_data(data)
        self.assertEqual(result["sample_size"], 30)
        self.assertEqual(len(result["theoretical_quantiles"]), 30)
        self.assertEqual(len(result["sample_quantiles"]), 30)
        self.assertIn("reference_line", result)
        ref = result["reference_line"]
        self.assertIn("slope", ref)
        self.assertIn("intercept", ref)
        self.assertIn("r_squared", ref)
        self.assertIn("x", ref)
        self.assertIn("y", ref)
        self.assertEqual(len(ref["x"]), 2)
        self.assertEqual(len(ref["y"]), 2)
        self.assertIn("sample_mean", result)
        self.assertIn("sample_std", result)

    def test_qq_plot_normal_data_high_r_squared(self):
        data = np.random.normal(0, 1, 100)
        result = self.tester.qq_plot_data(data)
        self.assertGreater(result["reference_line"]["r_squared"], 0.95)

    def test_qq_plot_non_normal_data_lower_r_squared(self):
        data = np.random.exponential(1, 100)
        result = self.tester.qq_plot_data(data)
        sw_result = self.tester.shapiro_wilk_test(data)
        if not sw_result["is_normal"]:
            self.assertLess(result["reference_line"]["r_squared"], 0.98)

    def test_qq_plot_sorted_sample_quantiles(self):
        data = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        result = self.tester.qq_plot_data(data)
        sample_q = result["sample_quantiles"]
        self.assertEqual(sample_q, sorted(sample_q))

    def test_qq_plot_reference_line_matches_slope_intercept(self):
        data = np.random.normal(0, 1, 50)
        result = self.tester.qq_plot_data(data)
        ref = result["reference_line"]
        for x, y in zip(ref["x"], ref["y"]):
            self.assertAlmostEqual(y, ref["slope"] * x + ref["intercept"], places=10)

    def test_qq_plot_insufficient_data(self):
        with self.assertRaises(ValueError):
            self.tester.qq_plot_data([1, 2])

    def test_qq_plot_nan_data(self):
        with self.assertRaises(ValueError):
            self.tester.qq_plot_data([1, 2, np.nan])

    def test_comprehensive_includes_qq_plot(self):
        data = np.random.normal(0, 1, 30)
        result = self.tester.comprehensive_test(data)
        self.assertIn("qq_plot", result)
        self.assertIn("theoretical_quantiles", result["qq_plot"])
        self.assertIn("sample_quantiles", result["qq_plot"])
        self.assertIn("reference_line", result["qq_plot"])

    def test_2d_array_error(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError) as context:
            self.tester.shapiro_wilk_test(data)
        self.assertIn("一维数组", str(context.exception))


if __name__ == "__main__":
    unittest.main()
