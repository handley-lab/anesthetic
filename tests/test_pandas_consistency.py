"""Tests for WeightedSeries/DataFrame consistency with pandas behavior."""
import warnings
import numpy as np
import pandas as pd
import pytest
from anesthetic.weighted_pandas import WeightedSeries, WeightedDataFrame


class TestPandasConsistency:
    """Test consistency with pandas for unweighted data."""

    @pytest.mark.parametrize('data', [[np.nan, np.nan, np.nan, np.nan, np.nan],
                                      [1.0, np.nan, 3.0, 4.0, 5.0, 9.0],
                                      [2.0],
                                      [2.0, 3.0],
                                      [2.0, 3.0, 9.0],
                                      [2.0, 3.0, 9.0, 2.0],
                                      [np.nan],
                                      [0.0, np.nan, 0.0, 0.0, 0.0],
                                      [1e-4, 2e-4, 3e-4, 4e-4, 9e-4],
                                      [1e10, 2e10, 3e10, 4e10, 9e10]])
    @pytest.mark.parametrize('method, ddof', [
        ('mean', None),
        ('var', 0),
        ('var', 1),
        ('std', 0),
        ('std', 1),
        ('sem', 0),
        ('sem', 1),
        ('skew', None),
        ('kurt', None),
    ])
    @pytest.mark.parametrize('skipna', [True, False])
    @pytest.mark.parametrize('weight_type', ['frequency', 'reliability'])
    def test_series_consistency(self, data, method, ddof, skipna, weight_type):
        """Test WeightedSeries consistency with pandas Series."""
        ps = pd.Series(data)
        if weight_type == 'frequency':
            weights = np.ones_like(data, dtype=int)
            ps = ps.loc[ps.index.repeat(weights)].reset_index(drop=True)
        else:
            weights = np.ones_like(data, dtype=float) / len(data)
        ws = WeightedSeries(data, weights=weights)  # equally weighted

        # Get results
        if ddof is not None:
            pandas_result = getattr(ps, method)(skipna=skipna, ddof=ddof)
            weight_result = getattr(ws, method)(skipna=skipna, ddof=ddof)
        else:
            pandas_result = getattr(ps, method)(skipna=skipna)
            weight_result = getattr(ws, method)(skipna=skipna)

        # Check consistency
        assert np.isnan(weight_result) is np.isnan(pandas_result)
        assert weight_result == pytest.approx(pandas_result, rel=1e-10,
                                              nan_ok=True)
        assert type(weight_result) is type(pandas_result)

    @pytest.mark.parametrize('data', [
        np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0],
                  [7.0, 8.0, 9.0]]),
        np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [7.0, 8.0]]),
        np.array([[1.0, 2.0],
                  [7.0, 8.0]]),
        np.array([[1.0, 2.0]]),
        np.array([[1.0],
                  [4.0],
                  [7.0],
                  [7.0]]),
        np.array([[1.0]]),
        np.array([[np.nan]]),
        np.array([[1.0, 2.0, np.nan],
                  [4.0, np.nan, 6.0],
                  [7.0, 8.0, np.nan],
                  [7.0, 8.0, 9.0]]),
        np.array([[np.nan, np.nan, np.nan],
                  [np.nan, np.nan, np.nan],
                  [np.nan, np.nan, np.nan],
                  [np.nan, np.nan, np.nan]]),
    ])
    @pytest.mark.parametrize('method, ddof, skipna', [
        ('mean', None, True),
        ('mean', None, False),
        ('var', 0, True), ('var', 0, False),
        ('var', 1, True), ('var', 1, False),
        ('std', 0, True), ('std', 0, False),
        ('std', 1, True), ('std', 1, False),
        ('sem', 0, True), ('sem', 0, False),
        ('sem', 1, True), ('sem', 1, False),
        # ('cov', 0, None),  # not testing because of pandas bug: issue #45814
        ('cov', 1, None),
        ('skew', None, True),
        ('skew', None, False),
        ('kurt', None, True),
        ('kurt', None, False),
    ])
    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.parametrize('weight_type', ['frequency', 'reliability'])
    def test_dataframe_consistency(self, data, method, ddof, skipna, axis,
                                   weight_type):
        """Test WeightedDataFrame consistency with pandas DataFrame."""
        if method in ['cov', 'corr'] and axis == 1:
            pytest.skip(f"`{method}` does not support `axis` kwarg")
        elif data.shape[0] == 1:
            warnings.filterwarnings(action='ignore', category=RuntimeWarning)
        else:
            warnings.resetwarnings()
        columns = ['A', 'B', 'C'][:data.shape[1]]
        pdf = pd.DataFrame(data, columns=columns)
        if weight_type == 'frequency':
            weights = np.ones(data.shape[0], dtype=int) * 3
            if axis == 0:
                pdf = pdf.loc[pdf.index.repeat(weights)].reset_index(drop=True)
        else:
            weights = np.ones(data.shape[0], dtype=float) / data.shape[0]
        wdf = WeightedDataFrame(data, columns=columns, weights=weights)

        # Get results
        if skipna is None:
            pandas_result = getattr(pdf, method)(ddof=ddof)
            with pytest.raises(TypeError):
                getattr(wdf, method)(ddof=ddof, skipna=True, axis=1)
            weight_result = getattr(wdf, method)(ddof=ddof)
        elif ddof is not None:
            pandas_result = getattr(pdf, method)(skipna=skipna, axis=axis,
                                                 ddof=ddof)
            weight_result = getattr(wdf, method)(skipna=skipna, axis=axis,
                                                 ddof=ddof)
        else:
            pandas_result = getattr(pdf, method)(skipna=skipna, axis=axis)
            weight_result = getattr(wdf, method)(skipna=skipna, axis=axis)

        # Check consistency
        pandas_vals = pandas_result.values
        weight_vals = weight_result.values

        assert pandas_vals.shape == weight_vals.shape
        both_nan = np.isnan(pandas_vals) & np.isnan(weight_vals)
        values_match = np.allclose(pandas_vals, weight_vals,
                                   equal_nan=True, rtol=1e-10)

        assert np.all(both_nan | values_match), (
            f"{method}(skipna={skipna}, axis={axis}, ddof={ddof}) "
            f"inconsistent for {weight_type} weights: \ndata =\n{data}, \n"
            f"pandas =\n{pandas_vals}, \nweighted =\n{weight_vals}"
        )
        assert type(weight_vals) is type(pandas_vals)

    def test_weighted_data_differs_from_pandas(self):
        """Test that truly weighted data gives different results."""
        data = [1.0, 2.0, 3.0, 4.0, 9.0]
        weights = [0.1, 0.2, 0.3, 0.3, 0.1]  # Non-uniform weights

        ps = pd.Series(data)
        ws = WeightedSeries(data, weights=weights)

        # These should be different (weighted vs unweighted)
        assert not np.allclose(ps.mean(), ws.mean())
        assert not np.allclose(ps.var(ddof=0), ws.var(ddof=0))
        assert not np.allclose(ps.var(ddof=1), ws.var(ddof=1))
        assert not np.allclose(ps.std(ddof=0), ws.std(ddof=0))
        assert not np.allclose(ps.std(ddof=1), ws.std(ddof=1))
        assert not np.allclose(ps.skew(), ws.skew())
        assert not np.allclose(ps.kurt(), ws.kurt())

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
