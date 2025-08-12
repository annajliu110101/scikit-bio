import numpy as np
import pandas as pd
from typing import Literal
from copy import deepcopy
from warnings import catch_warnings
from unittest import TestCase, main
from sklearn.decomposition import PCA

from skbio import OrdinationResults
from skbio.stats.ordination._principal_component_analysis import pca
from skbio.util import (get_data_path, assert_ordination_results_equal,
                        assert_data_frame_almost_equal)

class TestPCA(TestCase):
    def setUp(self):
        self.data = pd.read_csv(get_data_path("PCA_example1_input.csv"), index_col = 0)
        self.transposed_data = pd.read_csv(get_data_path("PCA_example1_input.csv"), index_col = 0).transpose()
        self.large_sparse_data = pd.read_csv(get_data_path("PCA_example1_inputsparse.csv"), index_col = 0)
    
    def sklearn_results(self, data, copy = True, n_components = None, whiten = False, svd_solver:Literal['auto', 'full', 'arpack', 'randomized'] = "auto", tol = 0):
        pca_ = PCA(copy = copy, n_components = n_components, whiten = whiten, svd_solver = svd_solver, tol = tol)
        coordinates = pca_.fit_transform(data.copy())
        loadings = pca_.components_.T
        eigvals = pca_.explained_variance_
        proportion_explained = pca_.explained_variance_ratio_
        dimensions = eigvals.shape[0]
        axis_id=[f"PC{i+1}" for i in range(dimensions)]
        return OrdinationResults(short_method_name = "PCA", long_method_name = "Principal Component Analysis", eigvals = pd.Series(eigvals, index = axis_id), samples = pd.DataFrame(coordinates, index = data.index, columns = axis_id),
                                             features = pd.DataFrame(loadings, index = data.columns, columns = axis_id),
                                             proportion_explained = pd.Series(proportion_explained, index = axis_id))
    
    def pca_results(self, data, method = "svd", dimensions = 50, inplace = False ):
        return pca(data, method, dimensions, inplace)
    
    def test_simple(self):
        results = self.pca_results(self.data, method = "svd", dimensions = 50)
        expected_results = self.sklearn_results(self.data, n_components = 50, svd_solver = "full")
        assert_ordination_results_equal(results, expected_results,
                                        ignore_method_names=True,
                                        )
    def test_simple_transposed(self):
        results = self.pca_results(self.transposed_data, method = "svd", dimensions = 50)
        expected_results = self.sklearn_results(self.transposed_data, n_components = 50, svd_solver = "full")
        assert_ordination_results_equal(results, expected_results,
                                        ignore_method_names=True,
                                        )
    def test_svd_self(self):
        results = self.pca_results(self.data, method = "svd", dimensions = 50)
        expected_results = pca(self.data, method = "svd", dimensions = 50)
        assert_ordination_results_equal(results, expected_results,
                                        ignore_method_names=True,
                                        )
    
    def test_fsvd_self(self):
        results = self.pca_results(self.data, method = "fsvd", dimensions = 50)
        expected_results = self.pca_results(self.data, method = "fsvd", dimensions = 50)
        assert_ordination_results_equal(results, expected_results,
                                        ignore_method_names=True,
                                        ) 
    def test_eigh_self(self):
        results = self.pca_results(self.data, method = "eigh", dimensions = 50)
        expected_results = self.pca_results(self.data, method = "eigh", dimensions = 50)
        assert_ordination_results_equal(results, expected_results,
                                        ignore_method_names=True,
                                        ) 

    def test_eigh(self):
        results = self.pca_results(self.data, method = "eigh", dimensions = 50)
        expected_results = pca(self.data, method = "svd", dimensions = 50)
        assert_ordination_results_equal(results, expected_results,
                                        ignore_method_names=True,
                                        )

    def test_eigh_transposed(self):
        results = self.pca_results(self.transposed_data, method = "eigh", dimensions = 50)
        expected_results = pca(self.transposed_data, method = "svd", dimensions = 50)
        assert_ordination_results_equal(results, expected_results,
                                        ignore_method_names=True,
                                        )

    def test_fsvd(self):
        results = self.pca_results(self.data, method = "fsvd", dimensions = 50)
        expected_results = self.pca_results(self.data, method = "svd", dimensions = 50)
        assert_ordination_results_equal(results, expected_results,
                                        ignore_method_names=True,
                                        )
    def test_fsvd_transposed(self):
        results = self.pca_results(self.transposed_data, method = "fsvd", dimensions = 50)
        expected_results = self.pca_results(self.transposed_data, method = "svd", dimensions = 50)
        assert_ordination_results_equal(results, expected_results,
                                        ignore_method_names=True,
                                        )
    def test_fsvd_inplace(self):
        results = self.pca_results(self.data.copy(), method = "fsvd", dimensions = 50)
        results_inplace = self.pca_results(self.data.copy(), method = "fsvd", inplace = True, dimensions = 50)
        expected_results = self.pca_results(self.data, method = "svd", dimensions = 50)

        assert_ordination_results_equal(results, expected_results,
                                        ignore_method_names=True)

        assert_ordination_results_equal(results, results_inplace,
                                        ignore_method_names=True)
        
        results2 = self.pca_results(self.data, method="fsvd", dimensions=0, inplace=False)
        expected_results2 = self.pca_results(self.data, method="fsvd", dimensions = min(self.data.values.shape), inplace=False)

        
        assert_ordination_results_equal(results2, expected_results2,
                                        ignore_method_names=True,
                                        ignore_directionality = True,
                                        )

        with self.assertRaises(ValueError):
            dim_too_large = max(self.data.shape) + 10
            self.pca_results(self.data, method="fsvd", dimensions=dim_too_large)

        with self.assertRaises(ValueError):
            self.pca_results(self.data, method="fsvd", dimensions=-1)

        with self.assertRaises(ValueError):
            dim_too_large =  max(self.data.shape) + 10
            self.pca_results(self.data, method="eigh", dimensions=dim_too_large)

        with self.assertRaises(ValueError):
            self.pca_results(self.data, method="eigh", dimensions=-1)

        with self.assertRaises(ValueError):
            dim_too_large =  max(self.data.shape) + 10
            self.pca_results(self.data, method="svd", dimensions=dim_too_large)

        with self.assertRaises(ValueError):
            self.pca_results(self.data, method="svd", dimensions=-1)

        # larger (more stringent) threshold
        with catch_warnings(record=True) as obs:
            results = pca(self.data, warn_neg_eigval=0.2)
        self.assertEqual(obs, [])
        msg = ("warn_neg_eigval must be Boolean or a floating-point number between 0 "
               "and 1.")
        with self.assertRaisesRegex(ValueError, msg):
            pca(self.data, warn_neg_eigval=5.0)
        with self.assertRaisesRegex(ValueError, msg):
            pca(self.data, warn_neg_eigval=-2.5)

    def test_large_float(self):
        """Test with a float dimensions > 1."""
        msg = "A floating-point number greater than 1 cannot be"
        with self.assertRaisesRegex(ValueError, msg):
            pca(self.data, dimensions=2.5)

    def test_edge_case_for_all_variance(self):
        """Test with a dimensions close to 1 to retain nearly all variance.
        """
        results = pca(self.data, dimensions=0.9999, method = "fsvd")
        cumulative_variance = np.cumsum(results.proportion_explained.values)
        self.assertGreaterEqual(cumulative_variance[-1], 0.9999)

    def test_eigh_method_with_float(self):
        """Test with a float dimensions with eigh method to retain 80%
        variance."""
        results = pca(self.data, method="eigh", dimensions=0.8,
                       inplace=False, seed=None)
        cumulative_variance = np.cumsum(results.proportion_explained.values)
        self.assertGreaterEqual(cumulative_variance[-1], 0.8)
    
    def test_fsvd_method_with_float(self):
        """Test FSVD with float dimensions for variance threshold."""
        with self.assertWarns(RuntimeWarning):
            results = pca(self.data, method="fsvd", dimensions=0.8,
                           inplace=False, seed=None)
        cumulative_variance = np.cumsum(results.proportion_explained.values)
        self.assertGreaterEqual(cumulative_variance[-1], 0.8)

    def test_invalid_method(self):
        """Test that correct error is raised when method is invalid."""
        with self.assertRaisesRegex(
            ValueError,
            "PCA eigendecomposition method asdf not supported."
            ):
            results = pca(self.data, method="asdf")

if __name__ == '__main__':
    main()
