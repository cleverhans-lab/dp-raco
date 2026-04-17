import numpy as np
import pandas as pd
import logging
from typing import List, Optional


def _default_names(length: int, prefix: str) -> List[str]:
    """Helper to generate fallback names when none are provided."""
    return [f"{prefix}{i}" for i in range(length)]


def format_confusion_matrix(
    matrix: np.ndarray,
    label_names: Optional[List[str]] = None,
    sensitive_names: Optional[List[str]] = None,
    round_digits: int = 2,
) -> pd.DataFrame:
    """Return a *pandas* DataFrame view of the matrix.

    The matrix is expected to be shaped (num_labels, num_sensitive_attributes).
    Rows correspond to *labels* (predicted or true, depending on context) and
    columns correspond to sensitive-attribute classes.

    Parameters
    ----------
    matrix : np.ndarray or jax Array
        Confusion/count matrix to visualise.
    label_names : list of str, optional
        Names for each label class.  Auto-generated when *None*.
    sensitive_names : list of str, optional
        Names for each sensitive attribute class.  Auto-generated when *None*.
    round_digits : int
        Number of decimal places when rounding floating values for display.

    Returns
    -------
    pd.DataFrame
        A nicely formatted DataFrame ready for printing or plotting.
    """
    # Convert to numpy eagerly for compatibility with JAX DeviceArrays
    matrix_np = np.asarray(matrix)

    # Ensure 2-D shape.
    if matrix_np.ndim != 2:
        matrix_np = matrix_np.squeeze()
        if matrix_np.ndim != 2:
            raise ValueError("Matrix must be 2-D after squeezing.")

    num_labels, num_sensitive = matrix_np.shape

    if label_names is None:
        label_names = _default_names(num_labels, "L")
    if sensitive_names is None:
        sensitive_names = _default_names(num_sensitive, "S")

    df = pd.DataFrame(matrix_np, index=label_names, columns=sensitive_names)
    return df.round(round_digits)


def pretty_print_confusion_matrix(
    matrix: np.ndarray,
    name: str = "C",
    label_names: Optional[List[str]] = None,
    sensitive_names: Optional[List[str]] = None,
    round_digits: int = 2,
) -> None:
    """Pretty-print *matrix* with headers.

    This is a small convenience wrapper around *format_confusion_matrix* that
    directly prints the DataFrame to stdout.
    """
    df = format_confusion_matrix(
        matrix,
        label_names=label_names,
        sensitive_names=sensitive_names,
        round_digits=round_digits,
    )
    logging.debug("=== %s (Label x Sensitive) ===\n%s", name, df.to_string())