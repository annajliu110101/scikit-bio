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
    output_format=None,
):
    r"""Perform Principal Component Analysis (PCA).

    PCA is a dimensionality reduction technique that transforms a dataset into
    a new coordinate system where the greatest variance by any projection of
    the data comes to lie on the first coordinate (called the first principal
    component), the second greatest variance on the second coordinate, and so on.

    Parameters
    ----------
    table : DataFrame, array-like, or sparse matrix
        The input feature table with shape (n_samples, n_features). Rows represent
        samples and columns represent features. The table will be mean-centered
        before decomposition.
    method : {'svd', 'eigh', 'fsvd'}, optional
        Matrix decomposition method to use. Default is "svd".

        - "svd": Singular value decomposition, an exact method that computes all
          principal components. Best for general use cases.
        - "eigh": Eigendecomposition of the covariance matrix, supports computing
          a subset of dimensions efficiently for large matrices. Uses the kernel
          trick to optimize computation based on the relationship between number
          of samples and features.
        - "fsvd": Fast singular value decomposition, a randomized approximation
          method that can be faster for very large datasets when computing a
          subset of dimensions.

    dimensions : int or float, optional
        Number of dimensions (principal components) to retain. Default is 0, which
        retains all dimensions up to the rank of the matrix. If an integer >= 1,
        retains exactly that number of dimensions. If a float between 0 and 1,
        represents the target fraction of cumulative variance to be retained, and
        the minimum number of dimensions to achieve this variance will be computed.
    inplace : bool, optional
        If True, the input table will be mean-centered in-place to reduce memory
        consumption. Only applies to method "svd". Default is False.
    seed : int or np.random.Generator, optional
        A user-provided random seed or random generator instance for method "fsvd".
        See :func:`details <skbio.util.get_rng>`.
    output_format : optional
        Standard table parameters. See :ref:`table_params` for details.

    Returns
    -------
    OrdinationResults
        Object that stores the PCA results, including:

        - eigvals: The variance explained by each principal component (eigenvalues
          divided by n_samples - 1).
        - samples: Principal component scores (coordinates) for each sample.
        - features: Principal component loadings for each feature.
        - proportion_explained: Proportion of total variance explained by each
          principal component.

    See Also
    --------
    pcoa
    OrdinationResults

    Notes
    -----
    Principal Component Analysis (PCA) is a fundamental technique in multivariate
    statistics and machine learning. It performs an orthogonal transformation to
    convert potentially correlated features into linearly uncorrelated principal
    components.

    This implementation provides three decomposition methods:

    **SVD Method**
        Uses numpy's full SVD to decompose the mean-centered data matrix X as:
        X = U * S * V^T, where columns of U are the left singular vectors
        (coordinates in sample space), S contains singular values, and columns
        of V are the right singular vectors (loadings in feature space).
        Eigenvalues are computed as S^2.

    **Eigendecomposition Methods (eigh, fsvd)**
        Use the kernel trick for efficiency: when n_samples << n_features,
        compute eigenvectors from the covariance matrix (X @ X^T); when
        n_features < n_samples, compute from the Gram matrix (X^T @ X). This
        approach is more efficient but squares the condition number of the matrix,
        which may lead to numerical instabilities in ill-conditioned data.

        The eigh method uses SciPy's eigendecomposition and can efficiently
        compute a subset of eigenvalues when the subset is small relative to
        the matrix rank. The fsvd method uses randomized algorithms for
        approximate decomposition.

    **Sign Normalization**
        To ensure reproducibility and consistency across runs, the signs of
        eigenvectors are normalized so that the element with the largest
        absolute value in each column is positive.

    **Variance Explained**
        The proportion of variance explained by each principal component is
        computed as eigenvalue / sum(all eigenvalues). The sum of all
        eigenvalues equals the trace of the covariance matrix.

    References
    ----------
    .. [1] Pearson, K. (1901). On lines and planes of closest fit to systems of
       points in space. The London, Edinburgh, and Dublin Philosophical Magazine
       and Journal of Science, 2(11), 559-572.

    .. [2] Hotelling, H. (1933). Analysis of a complex of statistical variables
       into principal components. Journal of educational psychology, 24(6), 417.

    .. [3] Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure
       with randomness: Probabilistic algorithms for constructing approximate
       matrix decompositions. SIAM review, 53(2), 217-288.

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.ordination import pca

    Create a simple feature table with 4 samples and 3 features:

    >>> data = np.array([[2.5, 2.4],
    ...                  [0.5, 0.7],
    ...                  [2.2, 2.9],
    ...                  [1.9, 2.2],
    ...                  [3.1, 3.0]])

    Perform PCA using the default SVD method:

    >>> result = pca(data)
    >>> print(result.proportion_explained)  # doctest: +SKIP
    [0.96... 0.03...]

    Retain only components explaining 95% of variance:

    >>> result = pca(data, dimensions=0.95)
    >>> result.samples.shape  # doctest: +SKIP
    (5, 1)

    """
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

    if method == "svd":
        feature_table = scale(feature_table, with_std=False, copy=not inplace)

        # SVD returns C-CONTIGUOUS U, S, Vt
        U, S, Vt = svd(feature_table, full_matrices=False)
        U, V = normalize_signs(U, Vt.T)

        long_method_name = f"Principal Component Analysis with SVD"

        # Compute eigenvalues now to unify post-processing
        # downstream.  If eigenvalues were left as singular
        # values, the proportions explained would be incorrect.
        eigvals = S**2
    else:
        ndim = dimensions
        if 0 < dimensions < 1:
            # See mathematical proof in github README
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
            # Under the hood, lapack's 'evr' or 'evx' calls DSTEBZ and
            # DSTEIN to compute the subset of eigenvalues and eigenvectors.
            # This is more efficient than computing the full decomposition
            # when only a subset is needed, if the subset is small compared
            # to the rank of matrix.  However, as dimensions approaches rank_max,
            # the efficiency gain is reduced, and actually becomes slower than
            # computing the full decomposition.  The current cutoff is currently
            # a rough estimate, will need to be tuned based on benchmarking.
            subidx = [rank_max - ndim, rank_max - 1]
            if ndim >= 0.3 * rank_max:
                subidx = None

            eigvals, eigvecs = eigh(matrix_data, subset_by_index=subidx)
            long_method_name = (
                f"Principal Component Analysis Using Full Eigendecomposition"
            )

            signs = deterministic_signs(eigvecs)
            eigvecs *= signs

            eigvals = np.flip(eigvals)
            eigvecs = np.flip(eigvecs, axis=1)

            # The intermediate matrix is positive semi-definite by definition.
            # Therefore theoretically there should not be any negative eigenvalues.
            # However, as this is the only method that does not guard against negative
            # numbers, and because the kernel trick squares the condition number, we
            # clip large negative numbers that may arise due to numerical instability.
            eigvals[eigvals < 0] = 0.0

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

            signs = deterministic_signs(eigvecs)
            eigvecs *= signs
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

    if rank_max == eigvals.shape[0] or eigvals[-1] == 0:
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
    if V.size == 0:
        # X = U*S*Vᵗ -> Xᵗ*U = V*S -> V = Xᵗ*U*S⁻¹ = features
        V = feature_table @ U
        V /= S[np.newaxis, :]

    if U.size == 0:
        # X = U*S*Vᵗ -> XV = U*S = coordinates
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


def normalize_signs(u, v, in_sample_space=True):
    # to keep the signs consistent (for plotting purposes)

    # SVD (for some reason) returns U, V such that U is C-ordered
    # and V is Fortran-ordered.  So we're going to use V for sign
    # normalization since accessing its columns is faster.
    # columns of u, rows of v_t (columns of v)
    # take maximum absolute value in each column of u (out = row vector)
    # do not transpose to keep Fortran orderedness (efficiency) since
    # accessing columns is faster when matrix is in Fortran order
    if in_sample_space:
        max_abs_v_cols = np.argmax(np.abs(u.T), axis=1)
        shift = np.arange(u.T.shape[0])
        indices = max_abs_v_cols + shift * u.T.shape[1]
        signs = np.sign(np.take(np.reshape(u.T, (-1,)), indices, axis=0))
    else:
        max_abs_v_cols = np.argmax(np.abs(v), axis=0)
        shift = np.arange(v.shape[1])
        indices = max_abs_v_cols + shift * v.shape[0]
        signs = np.sign(np.take(np.reshape(v, (-1,)), indices, axis=0))
    u *= signs[np.newaxis, :]
    v *= signs[np.newaxis, :]
    return u, v


def deterministic_signs(u):
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    return signs


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
