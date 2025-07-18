# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
from scipy.linalg import svd

from ._ordination_results import OrdinationResults
from ._utils import svd_rank
from skbio.table._tabular import _create_table, _create_table_1d, _ingest_table


def ca(X, scaling=1, sample_ids=None, feature_ids=None, output_format=None):
    r"""Compute correspondence analysis.

    Correspondence analysis is a multivariate statistical technique for ordination.
    In general, rows in the data table will correspond to samples and
    columns to features, but the method is symmetric. In order to
    measure the correspondence between rows and columns, the
    :math:`\chi^2` distance is used, and those distances are preserved
    in the transformed space. The :math:`\chi^2` distance doesn't take
    double zeros into account, and so it is expected to produce better
    ordination that PCA when the data has lots of zero values.

    It is related to Principal Component Analysis (PCA) but it should
    be preferred in the case of steep or long gradients, that is, when
    there are many zeros in the input data matrix.

    Parameters
    ----------
    X : table_like of shape (n_samples, n_features)
        Input data table. See :doc:`../reference/table_like` for supported formats.
        Data must be non-negative and dimensionally homogeneous (numeric or binary).
    scaling : {1, 2}
        Scaling type 1 maintains :math:`\chi^2` distances between rows.
        Scaling type 2 preserves :math:`\chi^2` distances between columns.
        For a more detailed explanation of the interpretation,
        check notes below and Legendre & Legendre 1998, section 9.4.3.
    sample_ids, feature_ids, output_format : optional
        Standard ``TableLike`` parameters. See the `TableLike <https://scikit.bio/
        docs/dev/generated/skbio.util.config.html#the-datatable-type>`_ type
        documentation for details.

    Returns
    -------
    OrdinationResults
        Object that stores the computed eigenvalues, the transformed sample
        coordinates, the transformed features coordinates and the proportion
        explained.

    Raises
    ------
    NotImplementedError
        If the scaling value is not either `1` or `2`.
    ValueError
        If any of the input matrix elements are negative.

    See Also
    --------
    cca
    rda
    OrdinationResults

    Notes
    -----
    The algorithm is based on [1]_, \S 9.4.1., and is expected to give the same
    results as ``cca(X)`` in R's package vegan.

    In Scaling type 1, the euclidean distances between rows in the transformed
    space equal their :math:`\chi^2` distances in the original space.
    Rows (samples) near a column (features) indicate high contributions from
    that feature.

    In Scaling type 2, the euclidean distances between columns in the
    transformed space equal their :math:`\chi^2` distances in the original
    space. Columns (features) near a row (sample) indicate higher abundance
    in that sample.
    Other types of scalings are currently not implemented, as they are less used
    by ecologists (Legendre & Legendre 1998, p. 456).

    Features far from the center of the biplot and far from its edges often
    exhibit better relationships than features either in the center (may represent
    multimodal features, not related to the shown ordination axes) or the
    edges (sparse features).

    References
    ----------
    .. [1] Legendre P. and Legendre L. 1998. Numerical Ecology. Elsevier,
       Amsterdam.

    """
    if scaling not in {1, 2}:
        raise NotImplementedError("Scaling {0} not implemented.".format(scaling))

    short_method_name = "CA"
    long_method_name = "Correspondance Analysis"

    # we deconstruct the dataframe to avoid duplicating the data and be able
    # to perform operations on the matrix
    X, row_ids, column_ids = _ingest_table(
        X, sample_ids=sample_ids, feature_ids=feature_ids
    )

    # Correspondance Analysis
    r, c = X.shape

    if X.min() < 0:
        raise ValueError("Input matrix elements must be non-negative.")

    # Step 1 (similar to Pearson chi-square statistic)
    grand_total = X.sum()
    Q = X / grand_total

    column_marginals = Q.sum(axis=0)
    row_marginals = Q.sum(axis=1)

    # Formula 9.32 in Lagrange & Lagrange (1998). Notice that it's
    # an scaled version of the contribution of each cell towards
    # Pearson chi-square statistic.
    expected = np.outer(row_marginals, column_marginals)
    Q_bar = (Q - expected) / np.sqrt(expected)  # Eq. 9.32

    # Step 2 (Singular Value Decomposition)
    U_hat, W, Ut = svd(Q_bar, full_matrices=False)
    # Due to the centering, there are at most min(r, c) - 1 non-zero
    # eigenvalues (which are all positive)
    rank = svd_rank(Q_bar.shape, W)
    assert rank <= min(r, c) - 1
    U_hat = U_hat[:, :rank]
    W = W[:rank]
    U = Ut[:rank].T

    # Both scalings are a bit intertwined, so we'll compute both and
    # then choose
    V = column_marginals[:, None] ** -0.5 * U
    V_hat = row_marginals[:, None] ** -0.5 * U_hat
    F = V_hat * W
    # According to Formula 9.43, this should hold
    # assert np.allclose(F, (row_marginals**-1)[:, None] * Q.dot(V))
    # but it doesn't (notice that W**2==Lambda):
    # (9.43a) F = V_hat W = D(p_i+)^{-1/2} U_hat W
    #           = D(p_i+)^{-1/2} Q_bar U W^{-1} W  (substituting 9.38)
    #           = D(p_i+)^{-1/2} Q_bar U
    # (9.43b) F = D(p_i+)^{-1} Q V
    #           = D(p_i+)^{-1} Q D(p_+j)^{-1/2} U  (substituting 9.41)
    #           = D(p_i+)^{-1/2} D(p_i+)^{-1/2} Q D(p_+j)^{-1/2} U
    #           = D(p_i+)^{-1/2} Q_tilde U         (using 9.40)
    # It holds if we replace Q in 9.43b with Q after centering, ie
    # assert np.allclose(
    #    F,
    #    (row_marginals**-1)[:, None] * (Q - expected).dot(V))
    # Comparing results with vegan and the examples in the book, 9.43a
    # is the right one. The same issue happens in 9.44, where also
    # 9.44a is the one that matches vegan's output.
    # (9.44a) F_hat = V W = D(p_+j)^{-1/2} U W
    #               = D(p_+j)^{-1/2} Q_bar' U_hat W^{-1} W (using 9.39)
    #               = D(p_+j)^{-1/2} Q_bar' U_hat
    # (9.44b) F_hat = D(p_+j)^{-1} Q' V_hat
    #               = D(p_+j)^{-1/2} Q_tilde' U_hat (using 9.40 and 9.42)
    F_hat = V * W

    # Eigenvalues
    eigvals = W**2

    # features scores
    features_scores = [V, F_hat][scaling - 1]
    # sample scores (weighted averages of features scores)
    sample_scores = [F, V_hat][scaling - 1]

    # build the OrdinationResults object
    sample_columns = [
        "%s%d" % (short_method_name, i + 1) for i in range(sample_scores.shape[1])
    ]
    feature_columns = [
        "%s%d" % (short_method_name, i + 1) for i in range(features_scores.shape[1])
    ]
    eigvals = _create_table_1d(
        eigvals,
        index=["%s%d" % (short_method_name, i + 1) for i in range(eigvals.shape[0])],
        backend=output_format,
    )
    samples = _create_table(
        sample_scores, index=row_ids, columns=sample_columns, backend=output_format
    )
    features = _create_table(
        features_scores,
        index=column_ids,
        columns=feature_columns,
        backend=output_format,
    )

    return OrdinationResults(
        short_method_name,
        long_method_name,
        eigvals,
        samples=samples,
        sample_ids=row_ids,
        features=features,
        feature_ids=column_ids,
        proportion_explained=eigvals / eigvals.sum(),
    )
