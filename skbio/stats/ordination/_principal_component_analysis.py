# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from numbers import Integral
from warnings import warn

import numpy as np
from numpy import dot
from numpy.linalg import svd
from scipy.linalg import eigh

from skbio.table._tabular import _create_table, _create_table_1d, _ingest_table
from ._ordination_results import OrdinationResults
from ._utils import scale
from skbio.binaries import (
    pcoa_fsvd_available as _skbb_pcoa_fsvd_available,
    pcoa_fsvd as _skbb_pcoa_fsvd,
)
from ._principal_coordinate_analysis import _fsvd

def pca(
    table,
    method="svd",
    dimensions=0,
    inplace=False,
    seed=None,
    warn_neg_eigval=0.01,
    output_format=None,
):
    r"""Perform Principal Coordinate Analysis (PCA).
    PCA is an ordination method operating on sample x observation tables,
    calculated using Euclidean distances.

    The main computational difference with PCoA is that the input table
    is centered along the columns only, instead of double-centered in
    PCoA.

    Computing an intermediate matrix using A @ A.T ensures that all
    downstreamcomputations are virtually identical to PCoA while computing
    the covariance matrix (more likely) means the eigenvectors are in
    feature space, which will need to be converted back to sample space
    when computing the samples/coordinates.

    Parameters
    ----------
    table : Table-like object
        The input sample x feature table.
    method : str, optional
        Matrix decomposition method to use. Default is "svd".
        which computes exact eigenvectors and eigenvalues for all dimensions. The
        alternate is "fsvd" (fast singular value decomposition), a heuristic that can
        compute only a given number of dimensions.
    dimensions : int or float, optional
        Dimensions to reduce the distance matrix to. This number determines how many
        eigenvectors and eigenvalues will be returned. If an integer is provided, the
        exact number of dimensions will be retained. If a float between 0 and 1, it
        represents the fractional cumulative variance to be retained. Default is 0,
        which will retain the same number of dimensions as the feature table.
    inplace : bool, optional
        If True, the input table will be centered in-place to reduce memory
        consumption, at the cost of losing the original observations. Default is False.
    seed : int or np.random.Generator, optional
        A user-provided random seed or random generator instance for method "fsvd".
        See :func:`details <skbio.util.get_rng>`.

        .. versionadded:: 0.6.3

    warn_neg_eigval : bool or float, optional
        Raise a warning if any negative eigenvalue is obtained and its magnitude
        exceeds the specified fraction threshold compared to the largest positive
        eigenvalue, which suggests potential inaccuracy in the PCoA result. Default is
        0.01. Set True to warn regardless of the magnitude. Set False to disable
        warning completely.

        .. versionadded:: 0.6.3
    Notes
    -----

    """

    feature_table, sample_ids, feature_ids = _ingest_table(table)

    if dimensions == 0:
        if method == "fsvd" and min(feature_table.shape) > 10:
            warn(
                "FSVD: since no value for number_of_dimensions is specified, "
                "PCA for all dimensions will be computed, which may "
                "result in long computation time if the original "
                "feature table is large and/or if number of features"
                "is similar or larger than the number of samples",
                RuntimeWarning,
            )
        dimensions = min(feature_table.shape)
    elif dimensions < 0:
        raise ValueError(
            "Invalid operation: cannot reduce table "
            "to negative dimensions using PCA. Did you intend "
            'to specify the default value "0", which sets '
            "the number_of_dimensions equal to the "
            "number of features in the given table?"
        )
    elif dimensions > max(feature_table.shape):
        raise ValueError("Invalid operation: cannot extend past size of matrix.")
    elif dimensions > min(feature_table.shape):
        warn(
            "The number of non-negative singular values / eigenvectors"
            "are bounded by the rank of the feature table.  At maximum,"
            "it is min(n_samples, n_features).  The maximum rank will be"
            "calculated instead.",
            RuntimeWarning,
        )
        dimensions = min(feature_table.shape)
    elif not isinstance(dimensions, Integral) and dimensions > 1:
        raise ValueError(
            "Invalid operation: A floating-point number greater than 1 cannot be "
            "supplied as the number of dimensions."
        )

    if warn_neg_eigval and not 0 <= warn_neg_eigval <= 1:
        raise ValueError(
            "warn_neg_eigval must be Boolean or a floating-point number between 0 "
            "and 1."
        )

    centered_table = scale(feature_table, with_std=False, copy=not inplace)

    if method == "svd":
        U, S, Vt = svd(centered_table, full_matrices=False)
        long_method_name = f"Principal Component Analysis with SVD"

        # Compute eigenvalues now to unify post-processing
        # downstream.  If eigenvalues were left as singular
        # values, the proportions explained would be incorrect.
        eigvals = S**2
    else:
        # It is more numerically stable and more efficient
        # to compute covariance matrix (A @ A.T) for short
        # and wide matrices, and vice versa for the G matrix
        # (A.T @ A). The eigenvectors of a covariance matrix
        # are in sample space, and are in feature space
        # from a G matrix.  This corresponds to U (left
        # singular vectors) and V (right singular vectors),
        # respectively.
        if feature_table.shape[0] <= feature_table.shape[1]:
            centered_table = centered_table.T
        matrix_data = dot(centered_table.T, centered_table)

        if method == "eigh":
            eigvals, eigvecs = eigh(matrix_data)
            long_method_name = (
                f"Principal Component Analysis Using Full Eigendecomposition"
            )
            # Eigh returns values in sorted ascending order.
            # Eigenvectors are purposely kept in fortran-order
            # to maintain homogenity with downstream post-processing
            # with the other methods.  Additionally, ensures efficient
            # memory access for svd_flip
            eigvals = np.flip(eigvals, axis = 0)
            eigvecs = np.flip(eigvecs, axis = 1)

        elif method == "fsvd":
            num_dimensions = dimensions
            if 0 < dimensions < 1:
                warn(
                    "FSVD: since value for number_of_dimensions is "
                    "specified as float. PCA for all dimensions will be"
                    "computed, which may result in long computation time"
                    "if the original distance matrix is large. Consider"
                    "specifying an integer value to optimize performance.",
                    RuntimeWarning,
                )
                num_dimensions = min(feature_table.shape)
            eigvals, eigvecs = _fsvd(
                matrix_data, num_dimensions, seed=seed,
            )
            long_method_name = (
                "Approximate Principal Coordinate Analysis using FSVD"
            )
        else:
            raise ValueError(
                "PCA eigendecomposition method {} not supported.".format(method)
            )

        # Since an intermediate matrix was computed,
        # the eigenvectors produced are either in
        # sample space or in feature space.  Since
        # svd_flip expects Vt, not V, eigenvectors
        # are transposed.
        if feature_table.shape[0] <= feature_table.shape[1]:
            U, Vt = eigvecs, None
        else:
            U, Vt = None, eigvecs.T

    # If U exists, its columns are used to determine the sign.
    # The signs of all the values are with respect to the largest
    # eigenvector in each column.
    U, V = normalize_signs(U, Vt, U is not None)

    # If Vt exists, the embeddings in feature space are equivalent
    # to the transpose of V.
    if V is not None:
        V = V.T

    # In PCA, it is assumed that the metric used is euclidean.  Therefore
    # theoretically, there should not be any negative eigenvalues.  However,
    # eigh is known to have small rounding errors that may introduce small
    # negatives, so we set them equal to zero here.
    negative_close_to_zero = np.isclose(eigvals, 0.0) & (eigvals < 0)
    eigvals[negative_close_to_zero] = 0.0

    # large negative eigenvalues suggest result inaccuracy
    # see: https://github.com/scikit-bio/scikit-bio/issues/1410
    if warn_neg_eigval and eigvals[-1] < 0:
        if warn_neg_eigval is True or -eigvals[-1] > eigvals[0] * warn_neg_eigval:
            warn(
                "The result contains negative eigenvalues that are large in magnitude,"
                " which may suggest result inaccuracy. See Notes for details. The"
                " negative-most eigenvalue is {0} whereas the largest positive one is"
                " {1}.".format(eigvals[-1], eigvals[0]),
                RuntimeWarning,
            )

    if method == "fsvd":
        # Since the dimension parameter, hereafter referred to as 'd',
        # restricts the number of eigenvalues and eigenvectors that FSVD
        # computes, we need to use an alternative method to compute the sum
        # of all eigenvalues, used to compute the array of proportions
        # explained. Otherwise, the proportions calculated will only be
        # relative to d number of dimensions computed; whereas we want
        # it to be relative to the number of features in the input table.

        # An alternative method of calculating th sum of eigenvalues is by
        # computing the trace of the centered feature table.
        # See proof outlined here: https://goo.gl/VAYiXx
        sum_eigvals = np.trace(matrix_data)
    else:
        sum_eigvals = np.sum(eigvals)

    proportion_explained = eigvals / sum_eigvals

    if 0 < dimensions < 1:
        # gives the number of dimensions needed to reach specified variance
        # updates number of dimensions to reach the requirement of variance.
        cumulative_variance = np.cumsum(proportion_explained)
        num_dimensions = (
            np.searchsorted(cumulative_variance, dimensions, side="left") + 1
        )

        dimensions = num_dimensions

    eigvals = eigvals[:dimensions]
    proportion_explained = proportion_explained[:dimensions]
    if V is not None:
        V = V[:, :dimensions]
    if U is not None:
        U = U[:, :dimensions]

    # When this conditional statement evaluates to True, U
    # must exist.  In this pathway, centered_table is the
    # transposed view of itself.  V is computed by projecting
    # the basis vectors onto the feature space and normalized
    # by dividing by the singular-values.
    # X = U*S*Vᵗ -> Xᵗ*U = V*S -> V = Xᵗ*U*S⁻¹ = features
    if V is None:
        V = dot(centered_table, U)
        V *= np.maximum(np.power(eigvals, -0.5, where = eigvals > 0), 0)[np.newaxis, :]
    features = V

    # The embeddings in sample space are equivalent to the projection
    # of V onto the centered_table.  Samples, by convention, is the
    # left-singular basis vectors scaled by the singular values.
    # Alternatively, the samples can be computed by projecting the
    # loadings onto the original data.
    if U is None:
        # X = U*S*Vᵗ -> XV = U*S = coordinates
        U = dot(centered_table, V)
    else:
        # U*S = coordinates
        U *= np.maximum(np.power(eigvals, 0.5, where = eigvals > 0), 0)
    samples = U
    eigvals /= (feature_table.shape[0] - 1)

    return _encapsulate_pca_result(
        long_method_name,
        eigvals,
        samples,
        features,
        proportion_explained,
        sample_ids,
        feature_ids,
        output_format,
    )

def normalize_signs(u, v, u_based_decision=True):
    # This is pretty much identical to svd_flip in sklearn
    # temporary for now, until something else is figured out
    # to keep the signs consistent (for plotting purposes)
    if u_based_decision:
        # columns of u, rows of v, or equivalently rows of u.T and v
        max_abs_u_cols = np.argmax(np.abs(u.T), axis=1)
        shift = np.arange(u.T.shape[0])
        indices = max_abs_u_cols + shift * u.T.shape[1]
        signs = np.sign(np.take(np.reshape(u.T, (-1,)), indices, axis=0))
        u *= signs[np.newaxis, :]
        if v is not None:
            v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_v_rows = np.argmax(np.abs(v), axis=1)
        shift = np.arange(v.shape[0])
        indices = max_abs_v_rows + shift * v.shape[1]
        signs = np.sign(np.take(np.reshape(v, (-1,)), indices, axis=0))
        if u is not None:
            u *= signs[np.newaxis, :]
        v *= signs[:, np.newaxis]
    return u, v

def _encapsulate_pca_result(
    long_method_name,
    eigvals,
    coordinates,
    loadings,
    proportion_explained,
    sample_ids,
    feature_ids,
    output_format,
):
    dimensions = eigvals.shape[0]
    axis_labels = ["PC%d" % i for i in range(1, dimensions + 1)]
    return OrdinationResults(
        short_method_name="PCA",
        long_method_name=long_method_name,
        eigvals=_create_table_1d(eigvals, index=axis_labels, backend=output_format),
        samples=_create_table(
            coordinates,
            index=sample_ids,
            columns=axis_labels,
            backend=output_format,
        ),
        features=_create_table(
            loadings,
            index=feature_ids,
            columns=axis_labels,
            backend=output_format,
        ),
        proportion_explained=_create_table_1d(
            proportion_explained, index=axis_labels, backend=output_format
        ),
    )
