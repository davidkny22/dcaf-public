"""
Probing extensions for nonlinear analysis.

When LRS < threshold, linear probes may underestimate the representational
structure. This module provides polynomial and kernel-based probing to
detect nonlinear encodings of behavioral signals in activation spaces.

Diagnostics:
- Linear probing accuracy: Baseline logistic regression
- Polynomial probing accuracy: Degree-2 polynomial features
- Polynomial gap: poly - linear (indicates nonlinear structure)
- Top interacting dimensions: Strongest degree-2 interaction terms
- Kernel LDA accuracy: RBF-kernel discriminant analysis
- Kernel gap: kernel - linear (indicates kernel-exploitable structure)
"""

from dataclasses import dataclass
from typing import Dict, List, Any
import logging

import numpy as np
import torch
from torch import Tensor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_approximation import RBFSampler

logger = logging.getLogger(__name__)


@dataclass
class ProbingResults:
    """
    Probing results for nonlinear analysis.

    Compares linear, polynomial (degree-2), and kernel-based classifiers
    to detect nonlinear structure in activation representations.

    Attributes:
        linear_accuracy: Logistic regression accuracy on raw features.
        polynomial_accuracy: Logistic regression accuracy on degree-2
            polynomial features.
        polynomial_gap: polynomial_accuracy - linear_accuracy. Positive
            values indicate nonlinear structure exploited by degree-2 terms.
        top_interacting_dimensions: Top 10 interaction terms by coefficient
            magnitude, each as {i, j, w_ij} where i, j are dimension
            indices and w_ij is the interaction coefficient.
        kernel_lda_accuracy: RBF-kernel LDA accuracy.
        kernel_gap: kernel_lda_accuracy - linear_accuracy. Positive values
            indicate kernel-exploitable nonlinear structure.
    """
    linear_accuracy: float
    polynomial_accuracy: float
    polynomial_gap: float
    top_interacting_dimensions: List[Dict]
    kernel_lda_accuracy: float
    kernel_gap: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "linear_accuracy": self.linear_accuracy,
            "polynomial_accuracy": self.polynomial_accuracy,
            "polynomial_gap": self.polynomial_gap,
            "top_interacting_dimensions": self.top_interacting_dimensions,
            "kernel_lda_accuracy": self.kernel_lda_accuracy,
            "kernel_gap": self.kernel_gap,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProbingResults":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with probing result fields.

        Returns:
            ProbingResults instance.
        """
        return cls(
            linear_accuracy=data["linear_accuracy"],
            polynomial_accuracy=data["polynomial_accuracy"],
            polynomial_gap=data["polynomial_gap"],
            top_interacting_dimensions=data["top_interacting_dimensions"],
            kernel_lda_accuracy=data["kernel_lda_accuracy"],
            kernel_gap=data["kernel_gap"],
        )


def compute_kernel_lda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """
    Compute kernel LDA accuracy using RBF kernel approximation.

    Uses RBFSampler for scalable kernel approximation followed by
    LinearDiscriminantAnalysis on the transformed features.

    Args:
        X_train: Training features [n_train, d].
        y_train: Training labels [n_train].
        X_test: Test features [n_test, d].
        y_test: Test labels [n_test].

    Returns:
        Test accuracy in [0, 1].
    """
    # RBF kernel approximation
    rbf = RBFSampler(gamma=1.0, n_components=100, random_state=42)
    X_train_rbf = rbf.fit_transform(X_train)
    X_test_rbf = rbf.transform(X_test)

    # Fit LDA on kernel features
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_rbf, y_train)
    accuracy = lda.score(X_test_rbf, y_test)

    return float(accuracy)


def compute_polynomial_probing(
    activations: Tensor,
    labels: Tensor,
) -> ProbingResults:
    """
    Compute polynomial probing diagnostics on activations.

    Fits three classifiers to compare linear vs nonlinear separability:
    1. Logistic regression on raw features (linear baseline)
    2. Logistic regression on degree-2 polynomial features
    3. RBF-kernel LDA

    The polynomial gap (poly - linear) and kernel gap (kernel - linear)
    indicate the degree of nonlinear structure in the representation.

    Args:
        activations: Activation tensor [n_samples, d] or higher-dimensional
            (will be flattened to 2D).
        labels: Binary label tensor [n_samples] with values 0 or 1.

    Returns:
        ProbingResults with accuracy scores, gaps, and top interactions.
    """
    # Convert to numpy
    if isinstance(activations, torch.Tensor):
        X = activations.detach().cpu().numpy()
    else:
        X = np.asarray(activations)

    if isinstance(labels, torch.Tensor):
        y = labels.detach().cpu().numpy()
    else:
        y = np.asarray(labels)

    # Flatten to 2D
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    logger.debug(
        "Polynomial probing: X shape %s, y shape %s, classes %s",
        X.shape, y.shape, np.unique(y),
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # 1. Linear probing (logistic regression on raw features)
    lr_linear = LogisticRegression(max_iter=1000, random_state=42)
    lr_linear.fit(X_train, y_train)
    linear_accuracy = float(lr_linear.score(X_test, y_test))

    logger.debug("Linear accuracy: %.4f", linear_accuracy)

    # 2. Polynomial probing (degree-2 features)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    lr_poly = LogisticRegression(max_iter=1000, random_state=42)
    lr_poly.fit(X_train_poly, y_train)
    polynomial_accuracy = float(lr_poly.score(X_test_poly, y_test))

    logger.debug("Polynomial accuracy: %.4f", polynomial_accuracy)

    # Polynomial gap
    polynomial_gap = polynomial_accuracy - linear_accuracy

    # 3. Extract top interaction terms from polynomial coefficients
    feature_names = poly.get_feature_names_out()
    coefficients = lr_poly.coef_.flatten()

    # Find interaction terms (those containing a space, e.g., "x0 x1")
    interactions = []
    for idx, name in enumerate(feature_names):
        parts = name.split(' ')
        if len(parts) == 2 and parts[0] != parts[1]:
            # Extract dimension indices from feature names like "x0", "x1"
            try:
                i = int(parts[0].replace('x', ''))
                j = int(parts[1].replace('x', ''))
                interactions.append({
                    "i": i,
                    "j": j,
                    "w_ij": float(coefficients[idx]),
                })
            except (ValueError, IndexError):
                continue

    # Sort by absolute coefficient magnitude
    interactions.sort(key=lambda x: abs(x["w_ij"]), reverse=True)
    top_interacting_dimensions = interactions[:10]

    logger.debug(
        "Top interaction: %s",
        top_interacting_dimensions[0] if top_interacting_dimensions else "none",
    )

    # 4. Kernel LDA
    kernel_lda_accuracy = compute_kernel_lda(X_train, y_train, X_test, y_test)
    kernel_gap = kernel_lda_accuracy - linear_accuracy

    logger.debug("Kernel LDA accuracy: %.4f", kernel_lda_accuracy)

    result = ProbingResults(
        linear_accuracy=linear_accuracy,
        polynomial_accuracy=polynomial_accuracy,
        polynomial_gap=polynomial_gap,
        top_interacting_dimensions=top_interacting_dimensions,
        kernel_lda_accuracy=kernel_lda_accuracy,
        kernel_gap=kernel_gap,
    )

    logger.info(
        "Probing complete — linear=%.4f, poly=%.4f (gap=%.4f), kernel=%.4f (gap=%.4f)",
        linear_accuracy, polynomial_accuracy, polynomial_gap,
        kernel_lda_accuracy, kernel_gap,
    )

    return result


__all__ = [
    "ProbingResults",
    "compute_kernel_lda",
    "compute_polynomial_probing",
]
