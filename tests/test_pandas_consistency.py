"""Tests to ensure WeightedSeries/DataFrame consistency with pandas behavior."""

import numpy as np
import pandas as pd
import pytest
from anesthetic.weighted_pandas import WeightedSeries, WeightedDataFrame


class TestPandasConsistency:
    """Test consistency with pandas for unweighted data."""

    @pytest.fixture
    def test_cases(self):
        """Test data cases for consistency checking."""
        return {
            'all_nan': [np.nan, np.nan, np.nan],
            'mixed_with_nan': [1.0, np.nan, 3.0, np.nan, 5.0],
            'no_nan': [1.0, 2.0, 3.0, 4.0, 5.0],
            'single_value': [2.0],
            'single_nan': [np.nan],
            'zeros_with_nan': [0.0, np.nan, 0.0],
            'small_values': [1e-10, 2e-10, 3e-10],
            'large_values': [1e10, 2e10, 3e10]
        }

    @pytest.mark.parametrize('method', ['mean', 'var', 'std'])
    @pytest.mark.parametrize('skipna', [True, False])
    def test_series_consistency(self, test_cases, method, skipna):
        """Test WeightedSeries consistency with pandas Series for unweighted data."""
        for case_name, data in test_cases.items():
            ps = pd.Series(data)
            ws = WeightedSeries(data)  # Unweighted
            
            # Get results
            try:
                pandas_result = getattr(ps, method)(skipna=skipna)
                if method in ['var', 'std']:
                    # Use ddof=0 to match weighted statistics framework
                    pandas_result = getattr(ps, method)(skipna=skipna, ddof=0)
            except Exception:
                pandas_result = np.nan
                
            try:
                weighted_result = getattr(ws, method)(skipna=skipna)
            except Exception:
                weighted_result = np.nan
            
            # Check consistency
            both_nan = np.isnan(pandas_result) and np.isnan(weighted_result)
            values_match = np.allclose([pandas_result], [weighted_result], 
                                     equal_nan=True, rtol=1e-10)
            
            assert both_nan or values_match, (
                f"{method}(skipna={skipna}) inconsistent for {case_name}: "
                f"pandas={pandas_result}, weighted={weighted_result}"
            )

    @pytest.mark.parametrize('method', ['mean', 'var', 'std'])
    @pytest.mark.parametrize('skipna', [True, False])
    @pytest.mark.parametrize('axis', [0, 1])
    def test_dataframe_consistency(self, method, skipna, axis):
        """Test WeightedDataFrame consistency with pandas DataFrame for unweighted data."""
        # Create test data
        data = np.array([
            [1.0, 2.0, np.nan],
            [4.0, np.nan, 6.0], 
            [7.0, 8.0, 9.0]
        ])
        
        pdf = pd.DataFrame(data, columns=['A', 'B', 'C'])
        wdf = WeightedDataFrame(data, columns=['A', 'B', 'C'])  # Unweighted
        
        # Get results
        try:
            pandas_result = getattr(pdf, method)(skipna=skipna, axis=axis)
            if method in ['var', 'std']:
                # Use ddof=0 to match weighted statistics framework
                pandas_result = getattr(pdf, method)(skipna=skipna, axis=axis, ddof=0)
        except Exception:
            pandas_result = pd.Series([np.nan] * (3 if axis == 0 else 3))
            
        try:
            weighted_result = getattr(wdf, method)(skipna=skipna, axis=axis)
        except Exception:
            weighted_result = pd.Series([np.nan] * (3 if axis == 0 else 3))
        
        # Check consistency
        pandas_vals = pandas_result.values if hasattr(pandas_result, 'values') else [pandas_result]
        weighted_vals = weighted_result.values if hasattr(weighted_result, 'values') else [weighted_result]
        
        both_nan = np.isnan(pandas_vals) & np.isnan(weighted_vals)
        values_match = np.allclose(pandas_vals, weighted_vals, equal_nan=True, rtol=1e-10)
        
        assert np.all(both_nan | values_match), (
            f"{method}(skipna={skipna}, axis={axis}) inconsistent: "
            f"pandas={pandas_vals}, weighted={weighted_vals}"
        )

    def test_weighted_data_differs_from_pandas(self):
        """Test that truly weighted data gives different results from pandas."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        weights = [0.1, 0.2, 0.3, 0.3, 0.1]  # Non-uniform weights
        
        ps = pd.Series(data)
        ws = WeightedSeries(data, weights=weights)
        
        # These should be different (weighted vs unweighted)
        assert not np.allclose(ps.mean(), ws.mean())
        assert not np.allclose(ps.var(ddof=0), ws.var())
        assert not np.allclose(ps.std(ddof=0), ws.std())

    def test_edge_cases(self):
        """Test edge cases that should be handled consistently."""
        # Empty data
        empty_ps = pd.Series([], dtype=float)
        empty_ws = WeightedSeries([], dtype=float)
        
        assert np.isnan(empty_ps.mean()) == np.isnan(empty_ws.mean())
        assert np.isnan(empty_ps.var()) == np.isnan(empty_ws.var())
        
        # All zero weights should return NaN
        data = [1.0, 2.0, 3.0]
        ws_zero_weights = WeightedSeries(data, weights=[0.0, 0.0, 0.0])
        
        assert np.isnan(ws_zero_weights.mean())
        assert np.isnan(ws_zero_weights.var())
        assert np.isnan(ws_zero_weights.std())

    def test_mathematical_properties(self):
        """Test that mathematical properties are preserved."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ws = WeightedSeries(data)  # Unweighted
        
        # std should be sqrt of var
        assert np.allclose(ws.std(), np.sqrt(ws.var()))
        
        # Mean of constants should be the constant
        constant_ws = WeightedSeries([5.0, 5.0, 5.0, 5.0])
        assert np.allclose(constant_ws.mean(), 5.0)
        assert np.allclose(constant_ws.var(), 0.0)
        
        # Variance of constants should be zero
        assert np.allclose(constant_ws.var(), 0.0)