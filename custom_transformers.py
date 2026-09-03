"""Custom scikit-learn transformers for Bank Marketing feature engineering."""

from typing import List, Optional, Union
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PdaysFeatures(BaseEstimator, TransformerMixin):
    """Transformer for engineering features based on 'pdays', 'previous', and 'poutcome'.

    Creates indicator variables for clients who were never previously contacted
    ('pdays_never') and identifies logical inconsistencies in the contact history
    ('is_inconsistent'). Converts special filler values (999) in 'pdays' to NaN.
    """

    def __init__(self) -> None:
        """Initializes the PdaysFeatures transformer."""
        super().__init__()

    def fit(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "PdaysFeatures":
        """Fit method for Scikit-Learn compatibility.

        Args:
            X (pd.DataFrame): Input features DataFrame.
            y (Optional[Union[pd.Series, np.ndarray]]): Target vector. Defaults to None.

        Returns:
            PdaysFeatures: Fitted transformer instance.
        """
        self.feature_names_in_ = list(X.columns) if hasattr(X, "columns") else None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame by adding new engineered flags and handling 999 in pdays.

        Args:
            X (pd.DataFrame): Input dataset containing 'pdays', 'previous', and 'poutcome'.

        Returns:
            pd.DataFrame: Transformed DataFrame with 'pdays_never', 'is_inconsistent',
                and NaN-replaced 'pdays'.
        """
        X = X.copy()

        # Identify logically inconsistent contact records
        if 'pdays' in X.columns:
            # Create pdays_never
            X['pdays_never'] = X['pdays'].eq(999).astype(int)
            
            # Create is_inconsistent, if we have additional needed columns
            if 'previous' in X.columns and 'poutcome' in X.columns:
                mask = (
                    X['pdays'].eq(999)
                    & X['previous'].gt(0)
                    & X['poutcome'].ne('nonexistent')
                )
                X['is_inconsistent'] = mask.astype(int)
            else:
                X['is_inconsistent'] = 0

            # Replace filler code 999 with NaN for numeric model compatibility
            X.loc[X['pdays'].eq(999), 'pdays'] = np.nan
        else:
            X['pdays_never'] = 0
            X['is_inconsistent'] = 0

        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get output feature names for transformation.

        Args:
            input_features (Optional[npt.ArrayLike]): Input feature names.

        Returns:
            np.ndarray: Array of output feature names including newly engineered columns.
        """
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)

        if input_features is None:
            return np.array([])

        feature_names = list(input_features)
        
        if 'pdays_never' not in feature_names:
            feature_names.append('pdays_never')
        if 'is_inconsistent' not in feature_names:
            feature_names.append('is_inconsistent')

        return np.array(feature_names, dtype=object)


class IQRClipper(BaseEstimator, TransformerMixin):
    """Transformer for clipping numerical feature outliers using Interquartile Range (IQR).

    Calculates the lower ($Q1 - \\text{multiplier} \\times IQR$) and upper
    ($Q3 + \\text{multiplier} \\times IQR$) boundaries for specified numeric columns
    during `fit` and clips values outside this range during `transform`.
    """

    def __init__(self, columns: List[str], multiplier: float = 1.5) -> None:
        """Initializes IQRClipper with target columns and IQR multiplier.

        Args:
            columns (List[str]): List of column names to apply clipping on.
            multiplier (float): IQR multiplier factor for boundaries. Defaults to 1.5.
        """
        self.columns = columns
        self.multiplier = multiplier
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "IQRClipper":
        """Calculates IQR lower and upper bounds for each target column.

        Args:
            X (pd.DataFrame): Training DataFrame containing target columns.
            y (Optional[Union[pd.Series, np.ndarray]]): Target vector. Defaults to None.

        Returns:
            IQRClipper: Fitted transformer instance storing bounds in `bounds_`.
        """
        self.feature_names_in_ = list(X.columns) if hasattr(X, "columns") else None

        X = X.copy()
        self.bounds_ = {}

        for col in self.columns:
            if col in X.columns:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1

                lo = float(q1 - self.multiplier * iqr)
                hi = float(q3 + self.multiplier * iqr)
                self.bounds_[col] = (lo, hi)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clips numerical column values to calculated lower and upper IQR boundaries.

        Args:
            X (pd.DataFrame): Input dataset to clip.

        Returns:
            pd.DataFrame: DataFrame with values clipped within IQR bounds.
        """
        X = X.copy()

        for col in self.columns:
            if col in self.bounds_ and col in X.columns:
                lo, hi = self.bounds_[col]
                X[col] = X[col].clip(lo, hi)

        return X

    def get_feature_names_out(
        self, input_features: Optional[npt.ArrayLike] = None
    ) -> np.ndarray:
        """Get output feature names for transformation.

        Args:
            input_features (Optional[npt.ArrayLike]): Input feature names.

        Returns:
            np.ndarray: Array of output feature names (unchanged column list).
        """
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)

        if input_features is None:
            # No fit-time columns recorded and none supplied: fall back to
            # the clip-target columns only (legacy behavior).
            return np.array(self.columns, dtype=object)

        return np.asarray(input_features, dtype=object)