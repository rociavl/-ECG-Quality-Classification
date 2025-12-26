"""
AR Model Order Selection Module

Automatic AR model order selection using information criteria.
Based on arord.m (Abel Torres, IBEC, 2024) from BSA course materials.

Methods:
    - FPE: Final Prediction Error
    - AIC: Akaike Information Criterion
    - AICm: Modified AIC (BIC-like, recommended)
"""

import numpy as np
from typing import Tuple, List, Literal


def select_ar_order(
    signal: np.ndarray,
    max_order: int = 30,
    criterion: Literal['fpe', 'aic', 'aicm'] = 'aicm'
) -> Tuple[int, List[float]]:
    """
    Select optimal AR model order using information criteria.

    Based on arord.m (Abel Torres, IBEC, 2024) from BSA course.

    Args:
        signal: Input signal (1D array)
        max_order: Maximum order to test (default: 30)
        criterion: Selection criterion
            - 'fpe': Final Prediction Error
            - 'aic': Akaike Information Criterion
            - 'aicm': Modified AIC (recommended, penalizes more)

    Returns:
        Tuple of (optimal_order, scores_list)

    Theory:
        - Higher order = better fit but risk of overfitting
        - FPE: ρ × (1 + p/N) / (1 - p/N)
        - AIC: N × ln(ρ) + 2p
        - AICm: N × ln(ρ) + p × ln(N) (BIC-like, stronger penalty)

    Typical results:
        - Clean ECG: order 12-20
        - Noisy ECG: higher order (tries to model noise)
    """
    signal = np.asarray(signal).flatten()
    N = len(signal)

    # Limit max_order based on signal length
    max_order = min(max_order, N // 10)

    if max_order < 1:
        return 1, [0.0]

    scores = []
    orders = list(range(1, max_order + 1))

    for p in orders:
        # Compute AR coefficients using Burg method
        ar_coeffs, rho = _arburg_with_variance(signal, p)

        # Ensure rho is positive
        if rho <= 0:
            rho = 1e-10

        # Compute criterion score
        if criterion == 'fpe':
            # Final Prediction Error
            # FPE = ρ × (1 + p/N) / (1 - p/N)
            score = rho * (1 + p / N) / (1 - p / N + 1e-10)

        elif criterion == 'aic':
            # Akaike Information Criterion
            # AIC = N × ln(ρ) + 2p
            score = N * np.log(rho) + 2 * p

        elif criterion == 'aicm':
            # Modified AIC (similar to BIC)
            # AICm = N × ln(ρ) + p × ln(N)
            score = N * np.log(rho) + p * np.log(N)

        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        scores.append(score)

    # Find optimal order (minimum score)
    optimal_idx = np.argmin(scores)
    optimal_order = orders[optimal_idx]

    return optimal_order, scores


def _arburg_with_variance(x: np.ndarray, order: int) -> Tuple[np.ndarray, float]:
    """
    Burg method for AR parameter estimation with prediction error variance.

    Args:
        x: Input signal
        order: Model order

    Returns:
        Tuple of (ar_coefficients, prediction_error_variance)

    Note:
        AR coefficients are returned as [1, a1, a2, ..., ap]
    """
    N = len(x)
    a = np.zeros(order + 1)
    a[0] = 1.0

    # Initialize forward and backward prediction errors
    ef = x.copy().astype(float)
    eb = x.copy().astype(float)

    # Initial error variance (signal variance)
    rho = np.var(x)

    for m in range(order):
        # Forward and backward errors for current stage
        efm = ef[m + 1:]
        ebm = eb[m:-1]

        if len(efm) == 0 or len(ebm) == 0:
            break

        # Compute reflection coefficient
        num = -2.0 * np.dot(ebm, efm)
        den = np.dot(efm, efm) + np.dot(ebm, ebm)

        if den == 0:
            k = 0.0
        else:
            k = num / den

        # Update AR coefficients (Levinson recursion)
        a_new = np.zeros(m + 2)
        a_new[0] = 1.0
        for i in range(1, m + 2):
            if i <= m:
                a_new[i] = a[i] + k * a[m + 1 - i]
            else:
                a_new[i] = k

        a[:m + 2] = a_new

        # Update prediction error variance
        rho = rho * (1 - k * k)

        # Update prediction errors for next iteration
        ef_new = efm + k * ebm
        eb_new = ebm + k * efm
        ef[m + 1:] = ef_new
        eb[m + 1:] = eb_new

    return a, rho


def get_criterion_name(criterion: str) -> str:
    """Get full name of criterion for display."""
    names = {
        'fpe': 'Final Prediction Error (FPE)',
        'aic': 'Akaike Information Criterion (AIC)',
        'aicm': 'Modified AIC (AICm/BIC-like)'
    }
    return names.get(criterion.lower(), criterion)
