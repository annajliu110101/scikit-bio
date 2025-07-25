# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Testing utilities."""

import inspect
import os
import sys

import numpy as np
import numpy.testing as npt
import pandas.testing as pdt
from scipy.spatial.distance import pdist


class ReallyEqualMixin:
    """Use this for testing __eq__/__ne__.

    Taken and modified from the following public domain code:
      https://ludios.org/testing-your-eq-ne-cmp/

    """

    def assertReallyEqual(self, a, b):
        # assertEqual first, because it will have a good message if the
        # assertion fails.
        self.assertEqual(a, b)
        self.assertEqual(b, a)
        self.assertTrue(a == b)
        self.assertTrue(b == a)
        self.assertFalse(a != b)
        self.assertFalse(b != a)

    def assertReallyNotEqual(self, a, b):
        # assertNotEqual first, because it will have a good message if the
        # assertion fails.
        self.assertNotEqual(a, b)
        self.assertNotEqual(b, a)
        self.assertFalse(a == b)
        self.assertFalse(b == a)
        self.assertTrue(a != b)
        self.assertTrue(b != a)


def get_data_path(fn, subfolder="data"):
    """Return path to filename ``fn`` in the data folder.

    During testing it is often necessary to load data files. This
    function returns the full path to files in the ``data`` subfolder
    by default.

    Parameters
    ----------
    fn : str
        File name.

    subfolder : str, defaults to ``data``
        Name of the subfolder that contains the data.


    Returns
    -------
    str
        Inferred absolute path to the test data for the module where
        ``get_data_path(fn)`` is called.

    Notes
    -----
    The requested path may not point to an existing file, as its
    existence is not checked.

    """
    # getouterframes returns a list of tuples: the second tuple
    # contains info about the caller, and the second element is its
    # filename
    callers_filename = inspect.getouterframes(inspect.currentframe())[1][1]
    path = os.path.dirname(os.path.abspath(callers_filename))
    data_path = os.path.join(path, subfolder, fn)
    return data_path


def assert_ordination_results_equal(
    left,
    right,
    ignore_method_names=False,
    ignore_axis_labels=False,
    ignore_directionality=False,
    decimal=7,
):
    """Assert that ordination results objects are equal.

    This is a helper function intended to be used in unit tests that need to
    compare ``OrdinationResults`` objects.

    Parameters
    ----------
    left, right : OrdinationResults
        Ordination results to be compared for equality.
    ignore_method_names : bool, optional
        Ignore differences in `short_method_name` and `long_method_name`.
    ignore_axis_labels : bool, optional
        Ignore differences in axis labels (i.e., column labels).
    ignore_directionality : bool, optional
        Ignore differences in directionality (i.e., differences in signs) for
        attributes `samples`, `features` and `biplot_scores`.
    decimal : int, optional
        Number of decimal places to compare when checking numerical values.
        Defaults to 7.

    Raises
    ------
    AssertionError
        If the two objects are not equal.

    """
    npt.assert_equal(type(left) is type(right), True)

    if not ignore_method_names:
        npt.assert_equal(left.short_method_name, right.short_method_name)
        npt.assert_equal(left.long_method_name, right.long_method_name)

    _assert_frame_dists_equal(
        left.samples,
        right.samples,
        ignore_columns=ignore_axis_labels,
        ignore_directionality=ignore_directionality,
        decimal=decimal,
    )

    _assert_frame_dists_equal(
        left.features,
        right.features,
        ignore_columns=ignore_axis_labels,
        ignore_directionality=ignore_directionality,
        decimal=decimal,
    )

    _assert_frame_dists_equal(
        left.biplot_scores,
        right.biplot_scores,
        ignore_columns=ignore_axis_labels,
        ignore_directionality=ignore_directionality,
        decimal=decimal,
    )

    _assert_frame_dists_equal(
        left.sample_constraints,
        right.sample_constraints,
        ignore_columns=ignore_axis_labels,
        ignore_directionality=ignore_directionality,
        decimal=decimal,
    )

    _assert_series_equal(
        left.eigvals, right.eigvals, ignore_axis_labels, decimal=decimal
    )

    _assert_series_equal(
        left.proportion_explained,
        right.proportion_explained,
        ignore_axis_labels,
        decimal=decimal,
    )


def assert_ordination_results_equal_np(obs, exp, ignore_method_names=False, decimal=7):
    """NumPy version of testing ordination results."""
    if not ignore_method_names:
        npt.assert_equal(obs.short_method_name, exp.short_method_name)
        npt.assert_equal(obs.long_method_name, exp.long_method_name)

    # do this for samples, features, biplot_scores, sample_constraints
    obs_dists_samp = pdist(obs.samples)
    exp_dists_samp = pdist(exp.samples)
    npt.assert_almost_equal(obs_dists_samp, exp_dists_samp, decimal=decimal)

    # test features
    if exp.features is None:
        assert obs.features is None
    else:
        obs_dists_feat = pdist(obs.features)
        exp_dists_feat = pdist(exp.features)
        npt.assert_almost_equal(obs_dists_feat, exp_dists_feat, decimal=decimal)

    # test biplot_scores
    if exp.biplot_scores is None:
        assert obs.biplot_scores is None
    else:
        obs_dists_biplot = pdist(obs.biplot_scores)
        exp_dists_biplot = pdist(exp.biplot_scores)
        npt.assert_almost_equal(obs_dists_biplot, exp_dists_biplot, decimal=decimal)

    # test sample_constraints
    if exp.sample_constraints is None:
        assert obs.sample_constraints is None
    else:
        obs_dists_cons = pdist(obs.sample_constraints)
        exp_dists_cons = pdist(exp.sample_constraints)
        npt.assert_almost_equal(obs_dists_cons, exp_dists_cons, decimal=decimal)

    # test eigvals
    npt.assert_almost_equal(obs.eigvals, exp.eigvals, decimal=decimal)

    # test proportion_explained
    if exp.proportion_explained is None:
        assert obs.proportion_explained is None
    else:
        npt.assert_almost_equal(
            obs.proportion_explained, exp.proportion_explained, decimal=decimal
        )


def _assert_series_equal(left_s, right_s, ignore_index=False, decimal=7):
    # assert_series_equal doesn't like None...
    if left_s is None or right_s is None:
        assert left_s is None and right_s is None
    else:
        npt.assert_almost_equal(left_s.values, right_s.values, decimal=decimal)
        if not ignore_index:
            pdt.assert_index_equal(left_s.index, right_s.index)


def _assert_frame_dists_equal(
    left_df,
    right_df,
    ignore_index=False,
    ignore_columns=False,
    ignore_directionality=False,
    decimal=7,
):
    if left_df is None or right_df is None:
        assert left_df is None and right_df is None
    else:
        left_values = left_df.values
        right_values = right_df.values
        left_dists = pdist(left_values)
        right_dists = pdist(right_values)
        npt.assert_almost_equal(left_dists, right_dists, decimal=decimal)

        if not ignore_index:
            pdt.assert_index_equal(left_df.index, right_df.index)
        if not ignore_columns:
            pdt.assert_index_equal(left_df.columns, right_df.columns)


def _normalize_signs(arr1, arr2):
    """Change column signs so that "column" and "-column" compare equal.

    This is needed because results of eigenproblmes can have signs
    flipped, but they're still right.

    Notes
    -----
    This function tries hard to make sure that, if you find "column"
    and "-column" almost equal, calling a function like np.allclose to
    compare them after calling `normalize_signs` succeeds.

    To do so, it distinguishes two cases for every column:

    - It can be all almost equal to 0 (this includes a column of
      zeros).
    - Otherwise, it has a value that isn't close to 0.

    In the first case, no sign needs to be flipped. I.e., for
    |epsilon| small, np.allclose(-epsilon, 0) is true if and only if
    np.allclose(epsilon, 0) is.

    In the second case, the function finds the number in the column
    whose absolute value is largest. Then, it compares its sign with
    the number found in the same index, but in the other array, and
    flips the sign of the column as needed.

    """
    # Let's convert everything to floating point numbers (it's
    # reasonable to assume that eigenvectors will already be floating
    # point numbers). This is necessary because np.array(1) /
    # np.array(0) != np.array(1.) / np.array(0.)
    arr1 = np.asarray(arr1, dtype=np.float64)
    arr2 = np.asarray(arr2, dtype=np.float64)

    if arr1.shape != arr2.shape:
        raise ValueError(
            "Arrays must have the same shape ({0} vs {1}).".format(
                arr1.shape, arr2.shape
            )
        )

    # To avoid issues around zero, we'll compare signs of the values
    # with highest absolute value
    max_idx = np.abs(arr1).argmax(axis=0)
    max_arr1 = arr1[max_idx, range(arr1.shape[1])]
    max_arr2 = arr2[max_idx, range(arr2.shape[1])]

    sign_arr1 = np.sign(max_arr1)
    sign_arr2 = np.sign(max_arr2)

    # Store current warnings, and ignore division by zero (like 1. /
    # 0.) and invalid operations (like 0. / 0.)
    wrn = np.seterr(invalid="ignore", divide="ignore")
    differences = sign_arr1 / sign_arr2
    # The values in `differences` can be:
    #    1 -> equal signs
    #   -1 -> diff signs
    #   Or nan (0/0), inf (nonzero/0), 0 (0/nonzero)
    np.seterr(**wrn)

    # Now let's deal with cases where `differences != \pm 1`
    special_cases = (~np.isfinite(differences)) | (differences == 0)
    # In any of these cases, the sign of the column doesn't matter, so
    # let's just keep it
    differences[special_cases] = 1

    return arr1 * differences, arr2


def assert_data_frame_almost_equal(left, right, rtol=1e-5):
    """Raise AssertionError if ``pd.DataFrame`` objects are not "almost equal".

    Wrapper of ``pd.util.testing.assert_frame_equal``. Floating point values
    are considered "almost equal" if they are within a threshold defined by
    ``assert_frame_equal``. This wrapper uses a number of
    checks that are turned off by default in ``assert_frame_equal`` in order to
    perform stricter comparisons (for example, ensuring the index and column
    types are the same). It also does not consider empty ``pd.DataFrame``
    objects equal if they have a different index.

    Other notes:

    * Index (row) and column ordering must be the same for objects to be equal.
    * NaNs (``np.nan``) in the same locations are considered equal.

    This is a helper function intended to be used in unit tests that need to
    compare ``pd.DataFrame`` objects.

    Parameters
    ----------
    left, right : pd.DataFrame
        ``pd.DataFrame`` objects to compare.
    rtol : float, optional
        The relative tolerance parameter used for comparison. Defaults to 1e-5.

    Raises
    ------
    AssertionError
        If `left` and `right` are not "almost equal".

    See Also
    --------
    pandas.util.testing.assert_frame_equal

    """
    # pass all kwargs to ensure this function has consistent behavior even if
    # `assert_frame_equal`'s defaults change
    pdt.assert_frame_equal(
        left,
        right,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_frame_type=True,
        check_names=True,
        by_blocks=False,
        check_exact=False,
        rtol=rtol,
    )
    # this check ensures that empty DataFrames with different indices do not
    # compare equal. exact=True specifies that the type of the indices must be
    # exactly the same
    assert_index_equal(left.index, right.index)


def _data_frame_to_default_int_type(df):
    """Convert integer columns in a data frame into the platform-default integer type.

    Pandas DataFrame defaults to int64 when reading integers, rather than respecting
    the platform default (Linux and MacOS: int64, Windows: int32). This causes issues
    in comparing observed and expected data frames in Windows. This function repairs
    the issue by converting int64 columns of a data frame into int32 in Windows.

    See: https://github.com/unionai-oss/pandera/issues/726

    """
    for col in df.select_dtypes("int").columns:
        df[col] = df[col].astype(int)


def assert_series_almost_equal(left, right):
    # pass all kwargs to ensure this function has consistent behavior even if
    # `assert_series_equal`'s defaults change
    pdt.assert_series_equal(
        left,
        right,
        check_dtype=True,
        check_index_type=True,
        check_series_type=True,
        check_names=True,
        check_exact=False,
        check_datetimelike_compat=False,
        obj="Series",
    )
    # this check ensures that empty Series with different indices do not
    # compare equal.
    assert_index_equal(left.index, right.index)


def assert_index_equal(a, b):
    pdt.assert_index_equal(a, b, exact=True, check_names=True, check_exact=True)


def pytestrunner():
    try:
        import numpy

        try:
            # NumPy 1.14 changed repr output breaking our doctests,
            # request the legacy 1.13 style
            numpy.set_printoptions(legacy="1.13")
        except TypeError:
            # Old Numpy, output should be fine as it is :)
            # TypeError: set_printoptions() got an unexpected
            # keyword argument 'legacy'
            pass
    except ImportError:
        numpy = None

    try:
        import pandas

        # Max columns is automatically set by pandas based on terminal
        # width, so set columns to unlimited to prevent the test suite
        # from passing/failing based on terminal size.
        pandas.options.display.max_columns = None
    except ImportError:
        pandas = None

    try:
        import matplotlib
    except ImportError:
        matplotlib = None
    else:
        # Set a non-interactive backend for Matplotlib, such that it can work on
        # systems without graphics
        matplotlib.use("agg")

    # import here, cause outside the eggs aren't loaded
    import pytest

    errno = pytest.main(args=["--pyargs", "skbio"] + sys.argv[1:])
    sys.exit(errno)
