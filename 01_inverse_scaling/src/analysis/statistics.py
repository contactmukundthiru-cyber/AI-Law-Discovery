"""
Statistical analysis utilities for inverse scaling research.

Provides confidence intervals, hypothesis testing, and bootstrap estimation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable
from scipy import stats
import warnings


@dataclass
class ConfidenceInterval:
    """Confidence interval result."""
    lower: float
    upper: float
    point_estimate: float
    confidence_level: float
    method: str

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def margin_of_error(self) -> float:
        return self.width / 2

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "point_estimate": self.point_estimate,
            "confidence_level": self.confidence_level,
            "method": self.method,
            "width": self.width,
        }


@dataclass
class HypothesisTest:
    """Hypothesis test result."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    significant: bool
    alpha: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "significant": self.significant,
            "alpha": self.alpha,
            "description": self.description,
        }


class BootstrapEstimator:
    """
    Bootstrap estimation for statistical inference.

    Provides non-parametric confidence intervals and hypothesis tests.
    """

    def __init__(self, n_bootstrap: int = 10000, random_state: Optional[int] = None):
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_state)

    def confidence_interval(
        self,
        data: np.ndarray,
        statistic: Callable = np.mean,
        confidence_level: float = 0.95,
        method: str = "percentile",
    ) -> ConfidenceInterval:
        """
        Compute bootstrap confidence interval.

        Args:
            data: Input data array
            statistic: Statistic function to compute
            confidence_level: Confidence level (0-1)
            method: 'percentile', 'basic', or 'bca'

        Returns:
            ConfidenceInterval object
        """
        data = np.asarray(data)
        n = len(data)

        # Point estimate
        point_estimate = statistic(data)

        # Bootstrap samples
        bootstrap_stats = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            sample = self.rng.choice(data, size=n, replace=True)
            bootstrap_stats[i] = statistic(sample)

        alpha = 1 - confidence_level

        if method == "percentile":
            lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        elif method == "basic":
            lower = 2 * point_estimate - np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            upper = 2 * point_estimate - np.percentile(bootstrap_stats, 100 * alpha / 2)
        elif method == "bca":
            # Bias-corrected and accelerated
            lower, upper = self._bca_interval(data, bootstrap_stats, statistic, alpha)
        else:
            raise ValueError(f"Unknown method: {method}")

        return ConfidenceInterval(
            lower=lower,
            upper=upper,
            point_estimate=point_estimate,
            confidence_level=confidence_level,
            method=f"bootstrap_{method}",
        )

    def _bca_interval(
        self,
        data: np.ndarray,
        bootstrap_stats: np.ndarray,
        statistic: Callable,
        alpha: float,
    ) -> Tuple[float, float]:
        """Compute BCa confidence interval."""
        n = len(data)
        point_estimate = statistic(data)

        # Bias correction
        z0 = stats.norm.ppf(np.mean(bootstrap_stats < point_estimate))

        # Acceleration (jackknife)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_stats[i] = statistic(jack_sample)

        jack_mean = np.mean(jackknife_stats)
        a = np.sum((jack_mean - jackknife_stats) ** 3) / (
            6 * np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5
        )

        # Adjusted percentiles
        z_alpha = stats.norm.ppf(alpha / 2)
        z_1_alpha = stats.norm.ppf(1 - alpha / 2)

        alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        alpha2 = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))

        lower = np.percentile(bootstrap_stats, 100 * alpha1)
        upper = np.percentile(bootstrap_stats, 100 * alpha2)

        return lower, upper

    def permutation_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        statistic: Callable = lambda x, y: np.mean(x) - np.mean(y),
        alternative: str = "two-sided",
    ) -> HypothesisTest:
        """
        Permutation test for comparing two groups.

        Args:
            group1: First group data
            group2: Second group data
            statistic: Test statistic function
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            HypothesisTest result
        """
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)

        observed = statistic(group1, group2)
        combined = np.concatenate([group1, group2])
        n1 = len(group1)

        # Permutation distribution
        perm_stats = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            self.rng.shuffle(combined)
            perm_stats[i] = statistic(combined[:n1], combined[n1:])

        # Compute p-value
        if alternative == "two-sided":
            p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))
        elif alternative == "greater":
            p_value = np.mean(perm_stats >= observed)
        elif alternative == "less":
            p_value = np.mean(perm_stats <= observed)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(group1) * (len(group1) - 1) + np.var(group2) * (len(group2) - 1))
            / (len(group1) + len(group2) - 2)
        )
        effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

        return HypothesisTest(
            test_name="permutation_test",
            statistic=observed,
            p_value=p_value,
            effect_size=effect_size,
            significant=p_value < 0.05,
            alpha=0.05,
            description=f"Permutation test ({alternative})",
        )


class StatisticalAnalyzer:
    """
    Comprehensive statistical analyzer for inverse scaling experiments.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
    ):
        self.confidence_level = confidence_level
        self.bootstrap = BootstrapEstimator(bootstrap_samples, random_state)

    def compute_confidence_interval(
        self,
        data: np.ndarray,
        method: str = "bootstrap",
    ) -> ConfidenceInterval:
        """
        Compute confidence interval for mean.

        Args:
            data: Input data
            method: 'bootstrap', 't', or 'normal'

        Returns:
            ConfidenceInterval object
        """
        data = np.asarray(data)
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        if method == "bootstrap":
            return self.bootstrap.confidence_interval(
                data, np.mean, self.confidence_level
            )
        elif method == "t":
            se = std / np.sqrt(n)
            t_crit = stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
            margin = t_crit * se
            return ConfidenceInterval(
                lower=mean - margin,
                upper=mean + margin,
                point_estimate=mean,
                confidence_level=self.confidence_level,
                method="t_interval",
            )
        elif method == "normal":
            se = std / np.sqrt(n)
            z_crit = stats.norm.ppf((1 + self.confidence_level) / 2)
            margin = z_crit * se
            return ConfidenceInterval(
                lower=mean - margin,
                upper=mean + margin,
                point_estimate=mean,
                confidence_level=self.confidence_level,
                method="normal_interval",
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def compare_groups(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        test: str = "mann_whitney",
    ) -> HypothesisTest:
        """
        Compare two groups statistically.

        Args:
            group1: First group data
            group2: Second group data
            test: 'mann_whitney', 't_test', or 'permutation'

        Returns:
            HypothesisTest result
        """
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)

        if test == "mann_whitney":
            stat, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")
            # Effect size: rank-biserial correlation
            n1, n2 = len(group1), len(group2)
            effect_size = 1 - (2 * stat) / (n1 * n2)

            return HypothesisTest(
                test_name="Mann-Whitney U",
                statistic=stat,
                p_value=p_value,
                effect_size=effect_size,
                significant=p_value < 0.05,
                alpha=0.05,
                description="Non-parametric test for difference in distributions",
            )

        elif test == "t_test":
            stat, p_value = stats.ttest_ind(group1, group2)
            # Cohen's d
            pooled_std = np.sqrt(
                (np.var(group1, ddof=1) * (len(group1) - 1) +
                 np.var(group2, ddof=1) * (len(group2) - 1))
                / (len(group1) + len(group2) - 2)
            )
            effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

            return HypothesisTest(
                test_name="Independent t-test",
                statistic=stat,
                p_value=p_value,
                effect_size=effect_size,
                significant=p_value < 0.05,
                alpha=0.05,
                description="Parametric test for difference in means",
            )

        elif test == "permutation":
            return self.bootstrap.permutation_test(group1, group2)

        else:
            raise ValueError(f"Unknown test: {test}")

    def test_monotonicity(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Test for monotonic relationship (detect inverse scaling).

        Args:
            x: Independent variable (e.g., model scale)
            y: Dependent variable (e.g., performance)

        Returns:
            Dictionary with test results
        """
        x = np.asarray(x)
        y = np.asarray(y)

        # Spearman correlation (monotonicity)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        # Kendall tau
        kendall_tau, kendall_p = stats.kendalltau(x, y)

        # Mann-Kendall trend test
        mk_stat, mk_p = self._mann_kendall_test(y)

        # Detect inverse scaling
        is_inverse = spearman_r < -0.3 and spearman_p < 0.05

        return {
            "spearman": {
                "correlation": spearman_r,
                "p_value": spearman_p,
                "significant": spearman_p < 0.05,
            },
            "kendall": {
                "tau": kendall_tau,
                "p_value": kendall_p,
                "significant": kendall_p < 0.05,
            },
            "mann_kendall": {
                "statistic": mk_stat,
                "p_value": mk_p,
                "trend": "increasing" if mk_stat > 0 else "decreasing" if mk_stat < 0 else "none",
            },
            "inverse_scaling_detected": is_inverse,
            "relationship_type": self._classify_relationship(spearman_r, spearman_p),
        }

    def _mann_kendall_test(self, y: np.ndarray) -> Tuple[float, float]:
        """Perform Mann-Kendall trend test."""
        n = len(y)
        s = 0

        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = y[j] - y[i]
                if diff > 0:
                    s += 1
                elif diff < 0:
                    s -= 1

        # Variance
        var_s = n * (n - 1) * (2 * n + 5) / 18

        # Z statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0

        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return s, p_value

    def _classify_relationship(self, r: float, p: float) -> str:
        """Classify relationship type based on correlation."""
        if p >= 0.05:
            return "no_significant_relationship"
        elif r > 0.7:
            return "strong_positive"
        elif r > 0.3:
            return "moderate_positive"
        elif r > 0:
            return "weak_positive"
        elif r > -0.3:
            return "weak_negative"
        elif r > -0.7:
            return "moderate_negative"
        else:
            return "strong_negative"

    def detect_peak(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect if there's a peak (inverted-U) in the relationship.

        Args:
            x: Independent variable (scale)
            y: Dependent variable (performance)

        Returns:
            Peak information if detected, None otherwise
        """
        x = np.asarray(x)
        y = np.asarray(y)

        # Sort by x
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        # Find local maximum
        max_idx = np.argmax(y_sorted)

        # Check if it's a true peak (not at endpoints)
        if max_idx == 0 or max_idx == len(y) - 1:
            return None

        # Check if values decrease on both sides
        left_decreasing = np.all(np.diff(y_sorted[:max_idx + 1]) >= -0.01)  # Allow small noise
        right_decreasing = np.all(np.diff(y_sorted[max_idx:]) <= 0.01)

        if not (y_sorted[max_idx] > y_sorted[0] and y_sorted[max_idx] > y_sorted[-1]):
            return None

        # Compute peak prominence
        prominence = min(
            y_sorted[max_idx] - y_sorted[0],
            y_sorted[max_idx] - y_sorted[-1],
        )

        return {
            "peak_x": x_sorted[max_idx],
            "peak_y": y_sorted[max_idx],
            "peak_index": max_idx,
            "prominence": prominence,
            "left_value": y_sorted[0],
            "right_value": y_sorted[-1],
            "is_significant": prominence > 0.05,  # At least 5% drop
        }
