# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from numbers import Integral
from warnings import warn
import math

import numpy as np
from numpy.linalg import svd
from scipy.linalg import eigh
from sklearn.decomposition import TruncatedSVD

from skbio.table._tabular import _create_table, _create_table_1d, _ingest_table
from ._ordination_results import OrdinationResults
from ._utils import scale, _f_matrix_inplace
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
    feature_table, sample_ids, feature_ids = _ingest_table(table)

    # Maximum possible rank for the input samples x features table
    rank_max = min(feature_table.shape)  # maximum possible rank of the input table

    if dimensions == 0:
        if method == "fsvd" and rank_max > 10:
            warn(
                "FSVD: since no value for number_of_dimensions is specified, "
                "PCA for all dimensions will be computed, which may "
                "result in long computation time if the original "
                "feature table is large and/or if number of features"
                "is similar or larger than the number of samples",
                RuntimeWarning,
            )
        dimensions = rank_max
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
    elif dimensions > rank_max:
        warn(
            "The number of non-negative singular values / eigenvectors"
            "are bounded by the rank of the feature table.  At maximum,"
            "it is min(n_samples, n_features).  The maximum rank will be"
            "calculated instead.",
            RuntimeWarning,
        )
        dimensions = rank_max
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

    n_samples, n_features = feature_table.shape
    in_sample_space = n_samples <= n_features
    ndim = rank_max

    if method == "svd":
        feature_table = scale(feature_table, with_std=False, copy=not inplace)

        # SVD returns FORTRAN-ordered U, S, Vt
        U, S, Vt = svd(feature_table, full_matrices=False)
        V = Vt.T
        long_method_name = f"Principal Component Analysis with SVD"

        # Compute eigenvalues now to unify post-processing
        # downstream.  If eigenvalues were left as singular
        # values, the proportions explained would be incorrect.
        eigvals = S**2
    else:
        ndim = dimensions
        if 0 < dimensions < 1:
            ndim = math.ceil(rank_max * dimensions)
            if ndim > 10:
                warn(
                    f"{method.upper()}: since value for number_of_dimensions"
                    " is specified as float,"
                    " PCA for all dimensions will be computed, which may"
                    " result in long computation time if the original"
                    " distance matrix is large."
                    " Consider specifying an integer value to optimize"
                    " performance.",
                    RuntimeWarning,
                )

        # It is more numerically stable and more efficient to use the kernel
        # trick; specifically compute covariance matrix  (A @ A.T) when
        # samples << features or vice versa (A.T @ A). Then, eigensolver
        # returns eigenvectors in sample space from  a covariance matrix
        # (A @ A.T) or in feature space from a G matrix (A.T @ A).
        if in_sample_space:
            # Computed using native skbio function for efficiency and potntially
            # for later optimizations The feature table is transposed to unify
            # downstream post-processing  since the eigenvectors returned are
            # equivalently U from SVD, and we want to compute V later by
            # projecting onto the feature space (hence the transpose). This
            # means we do not need to center the output when project
            matrix_data = feature_table @ feature_table.T
            matrix_data = _f_matrix_inplace(matrix_data)
            feature_table = feature_table.T
        else:
            # The feature table and the mean are both in feature space so the
            # outer product works  to center the columns of the intermediate
            # matrix computed in feature space without touching the original.
            # Will need to save the mean later to get the coordinates which
            # still need to be centered.
            mean = feature_table.mean(axis=0)
            matrix_data = feature_table.T @ feature_table
            matrix_data -= n_samples * mean[:, None] * mean[None, :]

        if method == "eigh":
            if ndim >= 0.3 * rank_max:
                ndim = rank_max
                subidx = None
            else:
                subidx = [rank_max - ndim, rank_max - 1]

            eigvals, eigvecs = eigh(matrix_data, subset_by_index=subidx)
            long_method_name = (
                f"Principal Component Analysis Using Full Eigendecomposition"
            )

            # The intermediate matrix is positive semi-definite by definition.
            # Therefore theoretically there should not be any negative eigenvalues.
            # However, as this is the only method that does not guard against negative
            # numbers, and because the kernel trick squares the condition number, we
            # clip large negative numbers that may arise due to numerical instability.
            eigvals[eigvals < 0] = 0.0

            eigvals = np.flip(eigvals)
            eigvecs = np.flip(eigvecs, axis=1)

        elif method == "fsvd":
            # Note that because the intermediate matrix was computed
            # the condition number is effectively squared.  This method
            # should be treated similarly to a randomized svd method,
            # in that there likely exists numerical instabilities.
            # Consider using eigh with subset_by_index
            eigvals, eigvecs = _fsvd(
                matrix_data,
                ndim,
                seed=seed,
            )
            long_method_name = "Approximate Principal Component Analysis using FSVD"
        else:
            raise ValueError(
                "PCA eigendecomposition method {} not supported.".format(method)
            )
        # Prepare placeholders for U or V and compute S for unified downstream post
        # processing
        S = np.sqrt(eigvals)
        eigvecs_projected = np.empty((max(n_samples, n_features), 0))

        if in_sample_space:
            # If intrmediate matrix computed is in sample space (A @ A.T),
            # then eigvecs are in sample space (U from SVD).  Otherwise
            # they correspond to V from SVD.
            U, V = eigvecs, eigvecs_projected
        else:
            U, V = eigvecs_projected, eigvecs

    # Normalize signs for consistency
    U, V = normalize_signs(U, V)

    if ndim == rank_max or eigvals[-1] == 0:
        sum_eigvals = np.sum(eigvals)
    else:
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

    # Release extra memory to cache to speed up back-projection
    U = np.asarray(U[:, :dimensions], copy=True)
    S = np.asarray(S[:dimensions], copy=True)
    V = np.asarray(V[:, :dimensions], copy=True)

    # U and V are guaranteed to be invertible
    # X = U*S*Vᵗ -> Xᵗ*U = V*S -> V = Xᵗ*U*S⁻¹ = features
    if V.size == 0:
        V = feature_table @ U
        V /= S[np.newaxis, :]
    # X = U*S*Vᵗ -> XV = U*S = coordinates
    if U.size == 0:
        U = feature_table @ V
        U -= mean @ V
    else:
        U *= S

    eigvals /= n_samples - 1

    return _encapsulate_pca_result(
        long_method_name,
        eigvals,
        U,
        V,
        proportion_explained,
        sample_ids,
        feature_ids,
        output_format,
    )


def normalize_signs(u, v):
    # to keep the signs consistent (for plotting purposes)
    if u.size != 0 and v.size != 0:
        # columns of u, rows of v_t (columns of v)
        # take maximum absolute value in each column of u (out = row vector)
        # do not transpose to keep Fortran orderedness (efficiency) since
        # accessing columns is faster when matrix is in Fortran order
        max_abs_u_cols = np.argmax(np.abs(u), axis=0)
        shift = np.arange(u.shape[1])
        indices = max_abs_u_cols + shift * u.shape[0]
        signs = np.sign(np.take(np.reshape(u, (-1,)), indices, axis=0))
        u *= signs[np.newaxis, :]
        v *= signs[np.newaxis, :]
    elif u.size != 0:
        max_abs_u_cols = np.argmax(np.abs(u), axis=0)
        shift = np.arange(u.shape[1])
        signs = np.sign(u[max_abs_u_cols, shift])
        u *= signs
    elif v.size != 0:
        # this route won't ever be taken with SVD, only if intermediate
        # is computed in feature space (A.T @ A) using eigh or fsvd
        max_abs_v_cols = np.argmax(np.abs(v), axis=0)
        indices = np.arange(v.shape[1])
        signs = np.sign(v[max_abs_v_cols, indices])
        v *= signs
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
