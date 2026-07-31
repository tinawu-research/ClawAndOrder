"""Paired statistics for the base-vs-fine-tuned comparison.

Two choices here decide whether the reported delta means anything.

Resample questions, not components
    Components inside one question are not independent — an answer that misses
    the period misses every fact stated about that period. Bootstrapping over
    components would treat those as separate draws and report a confidence
    interval several times narrower than the truth.

Pair the arms
    Both arms answer the same questions from the same frozen evidence, so the
    per-question difference removes question difficulty from the variance
    entirely. An unpaired test on 15 questions would be swamped by the spread
    between easy and hard items and would call almost any real delta
    insignificant.

No SciPy on this box, so the Wilcoxon p-value comes from an exact sign-flip
permutation over paired differences — which at n<=20 is better than the normal
approximation anyway.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass


@dataclass
class PairedDelta:
    """Difference between two arms over the same questions."""

    n: int
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n_better: int
    n_worse: int
    n_tied: int

    @property
    def significant(self) -> bool:
        """True when the 95% interval excludes zero."""
        return self.ci_low > 0 or self.ci_high < 0

    def summary(self) -> str:
        mark = "" if self.significant else "  (CI crosses zero)"
        return (
            f"{self.mean_a:.1%} -> {self.mean_b:.1%}  "
            f"delta {self.delta:+.1%} "
            f"[{self.ci_low:+.1%}, {self.ci_high:+.1%}] "
            f"p={self.p_value:.4f} "
            f"({self.n_better}W/{self.n_worse}L/{self.n_tied}T){mark}"
        )


def paired_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI for mean(b) - mean(a), resampling questions with replacement."""
    if len(scores_a) != len(scores_b):
        raise ValueError("paired comparison requires equal-length score lists")
    n = len(scores_a)
    if n == 0:
        return (0.0, 0.0)

    rng = random.Random(seed)
    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    means = []
    for _ in range(resamples):
        # One index list per resample: the same questions are drawn for both
        # arms, which is what keeps the comparison paired.
        picks = [rng.randrange(n) for _ in range(n)]
        means.append(sum(diffs[i] for i in picks) / n)
    means.sort()
    return (means[int(0.025 * resamples)], means[int(0.975 * resamples)])


def sign_flip_p(scores_a: list[float], scores_b: list[float], *, seed: int = 0) -> float:
    """Two-sided p-value from sign-flipping paired differences.

    Under the null the sign of each difference is arbitrary, so the exact
    distribution is every assignment of signs. At n<=20 that is at most 2^20
    combinations, cheap enough to enumerate; above that it is sampled.
    """
    diffs = [b - a for a, b in zip(scores_a, scores_b) if b != a]
    n = len(diffs)
    if n == 0:
        return 1.0
    observed = abs(sum(diffs))

    if n <= 20:
        total = extreme = 0
        for signs in itertools.product((1, -1), repeat=n):
            total += 1
            if abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12:
                extreme += 1
        return extreme / total

    rng = random.Random(seed)
    trials = 20000
    extreme = 0
    for _ in range(trials):
        flipped = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(flipped) >= observed - 1e-12:
            extreme += 1
    return extreme / trials


def compare(
    scores_a: list[float],
    scores_b: list[float],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> PairedDelta:
    """Full paired comparison of arm A (baseline) against arm B."""
    n = len(scores_a)
    mean_a = sum(scores_a) / n if n else 0.0
    mean_b = sum(scores_b) / n if n else 0.0
    low, high = paired_bootstrap(scores_a, scores_b, resamples=resamples, seed=seed)
    return PairedDelta(
        n=n,
        mean_a=mean_a,
        mean_b=mean_b,
        delta=mean_b - mean_a,
        ci_low=low,
        ci_high=high,
        p_value=sign_flip_p(scores_a, scores_b, seed=seed),
        n_better=sum(1 for a, b in zip(scores_a, scores_b) if b > a),
        n_worse=sum(1 for a, b in zip(scores_a, scores_b) if b < a),
        n_tied=sum(1 for a, b in zip(scores_a, scores_b) if b == a),
    )
