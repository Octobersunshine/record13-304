import numpy as np
import unittest
from normality_test import NormalityTest


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

    def test_auto_test_small_sample(self):
        data = np.random.normal(0, 1, 30)
        result = self.tester.auto_test(data)
        self.assertEqual(result["test_name"], "Shapiro-Wilk")
        self.assertTrue(result["auto_selected"])
        self.assertIn("自动选择 Shapiro-Wilk 检验", result["selection_reason"])

    def test_auto_test_large_sample(self):
        data = np.random.normal(0, 1, 100)
        result = self.tester.auto_test(data)
        self.assertEqual(result["test_name"], "Kolmogorov-Smirnov")
        self.assertTrue(result["auto_selected"])
        self.assertIn("自动选择 Kolmogorov-Smirnov 检验", result["selection_reason"])

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

    def test_comprehensive_test_large_sample(self):
        data = np.random.normal(0, 1, 100)
        result = self.tester.comprehensive_test(data)
        self.assertIn("shapiro_wilk", result["tests"])
        self.assertIn("kolmogorov_smirnov", result["tests"])

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
        self.assertIn("≤ 5000", str(context.exception))

    def test_custom_alpha(self):
        tester = NormalityTest(alpha=0.01)
        data = np.random.normal(0, 1, 30)
        result = tester.shapiro_wilk_test(data)
        self.assertEqual(result["alpha"], 0.01)

    def test_list_input(self):
        data = [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1]
        result = self.tester.shapiro_wilk_test(data)
        self.assertEqual(result["sample_size"], 10)

    def test_2d_array_error(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError) as context:
            self.tester.shapiro_wilk_test(data)
        self.assertIn("一维数组", str(context.exception))


if __name__ == "__main__":
    unittest.main()
