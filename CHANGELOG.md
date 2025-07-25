# scikit-bio changelog

## Version 0.7.1-dev


## Version 0.7.0

### Features

* Added a development roadmap for scikit-bio to the website ([#2251](https://github.com/scikit-bio/scikit-bio/pull/2251)).
* Added function `dirmult_lme`, a differential abundance test for longitudinal data through fitting a Dirichlet-multinomial linear mixed effects model ([#2080](https://github.com/scikit-bio/scikit-bio/pull/2080) and [#2250](https://github.com/scikit-bio/scikit-bio/pull/2250)).
* Added function `pair_align`, a re-designed pairwise sequence alignment engine that is versatile, efficient, and generalizable ([#2226](https://github.com/scikit-bio/scikit-bio/pull/2226) and [#2196](https://github.com/scikit-bio/scikit-bio/pull/2196)). It is meant to replace the old slow Python engine and the SSW wrapper. It supports:
  - Global, local and semi-global alignments (with all four ends customizable).
  - Nucleotide, protein, and un-grammared sequences, plain strings (ASCII and Unicode), words/tokens, and numbers.
  - Match/mismatch scores or substitution matrix.
  - Linear and affine gap penalties.
  - Integer, decimal and infinite scores.
  - Returning one, multiple or all optimal alignment paths.
* Added wrapper functions `pair_align_prot` and `pair_align_nucl` which are preloaded with scoring schemes consistent with BLASTP and BLASTN, respectively ([#2234](https://github.com/scikit-bio/scikit-bio/pull/2234)).
* Added function `align_score` to calculate the score of a pairwise or multiple sequence alignment ([#2201](https://github.com/scikit-bio/scikit-bio/pull/2201), [#2192](https://github.com/scikit-bio/scikit-bio/pull/2192)).
* Implemented a dispatch system in scikit-bio to handle a variety of table formats. Currently, it handles arrays, Pandas and Polars dataframes, BIOM tables, and AnnData. Scikit-bio functions can now operate on any of these table formats and outputs an object of the same format, or a format designated by the user
([#2187](https://github.com/scikit-bio/scikit-bio/pull/2187), [#2203](https://github.com/scikit-bio/scikit-bio/pull/2203), [#2246](https://github.com/scikit-bio/scikit-bio/pull/2246), [#2258](https://github.com/scikit-bio/scikit-bio/pull/2258), [#2260](https://github.com/scikit-bio/scikit-bio/pull/2260)).
* Added four augmentation methods: `phylomix`, `compos_cutmix`, `aitchison_mixup` and `mixup` to enable generation of synthetic samples ([#2214](https://github.com/scikit-bio/scikit-bio/pull/2214), [#2190](https://github.com/scikit-bio/scikit-bio/pull/2190), [#2253](https://github.com/scikit-bio/scikit-bio/pull/2253)).
* Added support for scikit-bio-binaries, a separate package which currently increases performance of the `pcoa` and `permanova` functions within scikit-bio ([#2247](https://github.com/scikit-bio/scikit-bio/pull/2247)).
* Added pre-built wheels of scikit-bio to PyPI for easier installation across platforms ([#2233](https://github.com/scikit-bio/scikit-bio/pull/2233), [#2232](https://github.com/scikit-bio/scikit-bio/pull/2232), [#2228](https://github.com/scikit-bio/scikit-bio/pull/2228), [#2252](https://github.com/scikit-bio/scikit-bio/pull/2252)).
* Adopting the Python array API standard in scikit-bio to enable GPU support for select functions. Further expansion of GPU support within scikit-bio is expected ([#2239](https://github.com/scikit-bio/scikit-bio/pull/2239) and [#2250](https://github.com/scikit-bio/scikit-bio/pull/2250)).
* Added `AlignPath.to_aligned` and `AlignPath.from_aligned` to extract aligned regions of the original sequences, and to reconstruct a path from aligned sequences ([#2226](https://github.com/scikit-bio/scikit-bio/pull/2226)).
* Added parameter `starts` to `AlignPath.from_tabular` to specify starting positions in the original sequences ([#2226](https://github.com/scikit-bio/scikit-bio/pull/2226)).

### Performance enhancements

* Enriched the tutorials of modules `sequence`, `alignment`, and `table` ([#2263](https://github.com/scikit-bio/scikit-bio/pull/2263)).
* Improved the performance of `dirmult_ttest` ([#2250](https://github.com/scikit-bio/scikit-bio/pull/2250)).
* Improved the performance of `ancom`. This is primarily due to exploiting vectorization of the statistical testing function (such as `f_oneway`). As a consequence, a custom testing function now must accept 2-D arrays as input and return 1-D arrays. Function names available under `scipy.stats` are not impacted ([#2250](https://github.com/scikit-bio/scikit-bio/pull/2250)).
* Added attributes `ranges` and `stops` to `AlignPath`. They facilitate locating the aligned part of each sequence as `seq[start:stop]` ([#2226](https://github.com/scikit-bio/scikit-bio/pull/2226) and [#2201](https://github.com/scikit-bio/scikit-bio/pull/2201)).
* Improved the performance of `SubstitutionMatrix.identity`.
* Enhanced `TabularMSA.from_path_seqs`. It now can extract the aligned region from the middle of a sequence. Also added docstring and doctests ([#2201](https://github.com/scikit-bio/scikit-bio/pull/2201)).
* Enhanced and changed the default behavior of `AlignPath.to_bits`, which now returns a bit array representing positions instead of segments. This is desired because with the old default behavior, `to_bits` and `from_bits` are not consistent with each other ([#2201](https://github.com/scikit-bio/scikit-bio/pull/2201)).

### Bug Fixes

* Fixed a bug that `PairAlignPath.from_cigar` would ignore the first insertion (`I`) of a CIGAR string ([#2236](https://github.com/scikit-bio/scikit-bio/pull/2236)).
* Fixed an inaccurate statement that one can specify `gap` as np.inf or np.nan in `AlignPath.to_indices`. These cases are impossible because the output is integer type.
* Fixed an inaccurate statement in the documentation of `SubstitutionMatrix.is_ascii`. This attribute is True when all characters in the alphabet are ASCII codes (0 to 127), not extended ASCII codes (0 to 255) ([#2226](https://github.com/scikit-bio/scikit-bio/pull/2226)).
* Fixed a bug that a `SubstitutionMatrix` cannot be copied ([#2226](https://github.com/scikit-bio/scikit-bio/pull/2226)).
* Fixed a bug in `AlignPath.to_indices` which would throw an error if the alignment path has only one segment ([#2201](https://github.com/scikit-bio/scikit-bio/pull/2201)).
* Fixed a bug in the documentation in which the `source` button would link to decorator code, instead of the relevant function ([#2184](https://github.com/scikit-bio/scikit-bio/pull/2184)).

### Miscellaneous

* In `TreeNode.root_at` and `TreeNode.root_at_midpoint`, the default value of `branch_attrs` was changed to an empty list; that of `root_name` was changed to None; that of `reset` was changed to True ([#2259](https://github.com/scikit-bio/scikit-bio/pull/2259)).
* In `TreeNode.unrooted_copy`, the default value of `branch_attrs` was changed to `{"length", "support"}`. Specifically, "name" was removed from this set, as a node label is often an attribute of the node instead of the branch. The default value of `root_name` was changed to None ([#2259](https://github.com/scikit-bio/scikit-bio/pull/2259)).
* In `TreeNode.copy`, the default value of `deep` was set to False. Now `tree.copy()` returns a shallow copy instead of a deep copy ([#2259](https://github.com/scikit-bio/scikit-bio/pull/2259)).
* In `TreeNode.compare_cophenet`, the default value of `ignore_self` was set to True. Therefore the estimated cophenetic distance between trees better correlates with their discrepancy ([#2259](https://github.com/scikit-bio/scikit-bio/pull/2259)).
* Renamed column "Reject null hypothesis" as "Signif" in `ancom` and `dirmult_ttest`'s report tables for conciseness ([#2250](https://github.com/scikit-bio/scikit-bio/pull/2250)).
* Renamed the parameter `significance_test` as `sig_test` in `ancom` for conciseness. The old name is preserved as an alias ([#2250](https://github.com/scikit-bio/scikit-bio/pull/2250)).
* Set the default data type of `SubstitutionMatrix` as `np.float32` (previous it was `float`, which is equivalent to `np.float64`). Made `dtype` an optional parameter in `from_dict` and `identity` methods.
* Adjusted the `__repr__` of `AlignPath` and `PairAlignPath` ([#2226](https://github.com/scikit-bio/scikit-bio/pull/2226) and [#2235](https://github.com/scikit-bio/scikit-bio/pull/2235)).
* Changed `AlignPath.shape`'s type from a named tuple to a normal tuple ([#2235](https://github.com/scikit-bio/scikit-bio/pull/2235)). Let the values be native Python `int` rather than `np.int64` ([#2201](https://github.com/scikit-bio/scikit-bio/pull/2201)).
* Changed `AlignPath.lengths` and `AlignPath.starts`'s dtype from `int64` to `intp`, as these attributes are to facilitate indexing ([#2226](https://github.com/scikit-bio/scikit-bio/pull/2226)).
* Changed `Sequence.to_indices`'s output index array dtype from `uint8` to `intp`, which is the native NumPy indexing type ([#2226](https://github.com/scikit-bio/scikit-bio/pull/2226)).
* Enriched the documentation of `SubstitutionMatrix` ([#2226](https://github.com/scikit-bio/scikit-bio/pull/2226)).
* Let `AlignPath.states` be uniformly 2-D, even if there are 8 or less sequences in the alignment ([#2201](https://github.com/scikit-bio/scikit-bio/pull/2201)).
* Updated documentation to include description of how to stream data through stdin with scikit-bio's `read` function ([#2185](https://github.com/scikit-bio/scikit-bio/pull/2185)).
* Improved documentation for the `DistanceMatrix` object ([#2204](https://github.com/scikit-bio/scikit-bio/pull/2204)).
* Remove autoplotting functionality to enable inplace operations on large in-memory objects, and improve documentation of existing plotting methods ([#2216](https://github.com/scikit-bio/scikit-bio/pull/2216), [#2223](https://github.com/scikit-bio/scikit-bio/pull/2223)).
* Initiated efforts to add type annotations to scikit-bio's codebase, starting with the `stats.distance` module ([#2219](https://github.com/scikit-bio/scikit-bio/pull/2219))
* Restored functionality to scikit-bio's benchmarking system and introduced a new repository for storing, running, and hosting benchmarks to prevent performance regression ([#2245](https://github.com/scikit-bio/scikit-bio/pull/2245)).
* Renamed the parameter `number_of_dimensions` to `dimensions` for the `pcoa` and `permdisp` functions. `number_of_dimensions` will remain a valid alias of the parameter, such that either option may be used. ([#2257](https://github.com/scikit-bio/scikit-bio/pull/2257)).
* Enriched the tutorial for `skbio.Sequence` ([#2243](https://github.com/scikit-bio/scikit-bio/pull/2243)).
* The `tree.nj` function can now operate on `DistanceMatrix` objects containing float32 or float64 values ([#2217](https://github.com/scikit-bio/scikit-bio/pull/2217)).
* Improved documentation for conversion between scikit-bio sequence alignments and Biopython and Biotite alignments ([#2229](https://github.com/scikit-bio/scikit-bio/pull/2229), [#2230](https://github.com/scikit-bio/scikit-bio/pull/2230)).
* Rewrote the `install` page for the website to reflect availability of wheels and to explicitly state scikit-bio's version support windows ([#2254](https://github.com/scikit-bio/scikit-bio/pull/2254)).
* Renamed the parameter `distance_matrix` to `distmat` in `pcoa`, `bioenv`, `anosim`, `permanova`, and `permdisp`. `distance_matrix` will remain a valid alias of the parameter, such that either option may be used. ([#2261](https://github.com/scikit-bio/scikit-bio/pull/2261))

### Backward-incompatible changes

* Removed `TreeNode.unrooted_deepcopy`. Use `TreeNode.unrooted_copy(deep=True)` instead ([#2259](https://github.com/scikit-bio/scikit-bio/pull/2259)).
* Removed `TreeNode.deepcopy`. Use `TreeNode.copy(deep=True)` instead ([#2259](https://github.com/scikit-bio/scikit-bio/pull/2259)).
* Removed `TreeNode.subtree`. It was a placehold but never implemented ([#2259](https://github.com/scikit-bio/scikit-bio/pull/2259)).
* Removed the wrapper for the Striped Smith Waterman (SSW) library ([#2240](https://github.com/scikit-bio/scikit-bio/pull/2240), [#2241](https://github.com/scikit-bio/scikit-bio/pull/2241)). Specifically, this removes `local_pairwise_align_ssw`, `StripedSmithWaterman`, and `AlignmentStructure` under `skbio.alignment`. We recommend using the new `skbio.alignment.pair_align` function for pairwise sequence alignment, or other packages that provide production-ready alignment algorithms. See [#1814](https://github.com/biocore/scikit-bio/issues/1814) for discussions.
* Removed `skbio.alignment.make_identity_substitution_matrix`. This has been replaced with `skbio.sequence.SubstitutionMatrix.identity`.


## Version 0.6.3

### Features

* Python 3.13+ is now supported ([#2146](https://github.com/scikit-bio/scikit-bio/pull/2146)).
* Added Balanced Minimum Evolution (BME) function for phylogenetic reconstruction and `balanced` option for NNI ([#2105](https://github.com/scikit-bio/scikit-bio/pull/2105) and [#2169](https://github.com/scikit-bio/scikit-bio/pull/2169)).
* Added functions `rf_dists`, `wrf_dists` and `path_dists` under `skbio.tree` to calculate multiple pariwise distance metrics among an arbitrary number of trees. They correspond to `TreeNode` methods `compare_rfd`, `compare_wrfd` and `compare_cophenet` for two trees ([#2166](https://github.com/scikit-bio/scikit-bio/pull/2166)).
* Added `height` and `depth` methods under `TreeNode` to calculate the height and depth of a given node.
* Added `TreeNode.compare_wrfd` to calculate the weighted Robinson-Foulds distance or its variants between two trees ([#2144](https://github.com/scikit-bio/scikit-bio/pull/2144)).
* Wrapped UPGMA and WPGMA from SciPy's linkage method ([#2094](https://github.com/scikit-bio/scikit-bio/pull/2094)).
* Added `TreeNode` methods: `bipart`, `biparts` and `compare_biparts` to encode and compare bipartitions in a tree ([#2144](https://github.com/scikit-bio/scikit-bio/pull/2144)).
* Added `TreeNode.has_caches` to check if a tree has caches ([#2103](https://github.com/scikit-bio/scikit-bio/pull/2103)).
* Added `TreeNode.is_bifurcating` to check if a tree is bifurcating (i.e., binary) ([#2117](https://github.com/scikit-bio/scikit-bio/pull/2117)).
* Added support for Python's `pathlib` module in the IO system ([#2119](https://github.com/scikit-bio/scikit-bio/pull/2119)).
* Added `TreeNode.path` to return a list of nodes representing the path from one node to another ([#2131](https://github.com/scikit-bio/scikit-bio/pull/2131)).
* Exposed `vectorize_counts_and_tree` function from the `diversity` module to allow use for improving ML accuracy in downstream pipelines ([#2173](https://github.com/scikit-bio/scikit-bio/pull/2173))

### Performance enhancements

* Significantly improved the performance of the neighbor joining (NJ) algorithm (`nj`) ([#2147](https://github.com/scikit-bio/scikit-bio/pull/2147)) and the greedy minimum evolution (GME) algorithm (`gme`) for phylogenetic reconstruction, and the NNI algorithm for tree rearrangement ([#2169](https://github.com/scikit-bio/scikit-bio/pull/2169)).
* Significantly improved the performance of `TreeNode.cophenet` (renamed from `tip_tip_distances`) for computing a patristic distance matrix among all or selected tips of a tree ([#2152](https://github.com/scikit-bio/scikit-bio/pull/2152)).
* Supported Robinson-Foulds distance calculation (`TreeNode.compare_rfd`) based on bipartitions (equivalent to `compare_biparts`). This is automatically enabled when the input tree is unrooted. Otherwise the calculation is still based on subsets (equivalent to `compare_subsets`). The user can override this behavior using the `rooted` parameter ([#2144](https://github.com/scikit-bio/scikit-bio/pull/2144)).
* Re-wrote the underlying algorithm of `TreeNode.compare_subsets` because it is equivalent to the Robinson-Foulds distance on rooted trees. Added parameter `proportion`. Renamed parameter `exclude_absent_taxa` as `shared_only` ([#2144](https://github.com/scikit-bio/scikit-bio/pull/2144)).
* Added parameter `include_self` to `TreeNode.subset`. Added parameters `within`, `include_full` and `include_tips` to `TreeNode.subsets` ([#2144](https://github.com/scikit-bio/scikit-bio/pull/2144)).
* Improved the performance and customizability of `TreeNode.total_length` (renamed from `descending_branch_length`). Added parameters `include_stem` and `include_self`.
* Improved the performance of `TreeNode.lca` ([#2132](https://github.com/scikit-bio/scikit-bio/pull/2132)).
* Improved the performance of `TreeNode` methods: `ancestors`, `siblings`, and `neighbors` ([#2133](https://github.com/scikit-bio/scikit-bio/pull/2133), [#2135](https://github.com/scikit-bio/scikit-bio/pull/2135)).
* Improved the performance of tree traversal algorithms ([#2093](https://github.com/scikit-bio/scikit-bio/pull/2093)).
* Improved the performance of tree copying ([#2103](https://github.com/scikit-bio/scikit-bio/pull/2103)).
* Further improved the caching mechanism of `TreeNode`. Specifically: 1. Node attribute caches are only registered at the root node, which improves memory efficiency. 2. Method `clear_caches` can be customized to clear node attribute and/or lookup caches, or specified attribute caches ([#2099](https://github.com/scikit-bio/scikit-bio/pull/2099)). 3. Added parameter `uncache` to multiple methods that involves tree manipulation. Default is True. When one knows that caches are not present or relevant, one may set this parameter as False to skip cache clearing to significantly improve performance ([#2103](https://github.com/scikit-bio/scikit-bio/pull/2103)).
* Expanded the functionality of `TreeNode.cache_attr`. It can now take a custom function to combine children and self attributes. This makes it possible to cache multiple useful clade properties such as node count and total branch length. Also enriched the method's docstring to provide multiple examples of caching clade properties ([#2099](https://github.com/scikit-bio/scikit-bio/pull/2099)).
* Added parameter `inplace` to methods `shear`, `root_at`, `root_at_midpoint` and `root_by_outgroup` of `TreeNode` to enable manipulating the tree in place (True), which is more efficient that making a manipulated copy of the tree (False, default) ([#2103](https://github.com/scikit-bio/scikit-bio/pull/2103)).
* `TreeNode.extend` can accept any iterable type of nodes as input ([#2103](https://github.com/scikit-bio/scikit-bio/pull/2103)).
* Added parameter `strict` to `TreeNode.shear` ([#2103](https://github.com/scikit-bio/scikit-bio/pull/2103)).
* Added parameter `exclude_attrs` to `TreeNode.unrooted_copy` ([#2103](https://github.com/scikit-bio/scikit-bio/pull/2103)).
* Added support for legacy random generator to `get_rng`, such that outputs of scikit-bio functions become reproducible with code that starts with `np.random.seed` or uses `RandomState` ([#2130](https://github.com/scikit-bio/scikit-bio/pull/2130)).
* Allowed `shuffle` and `compare_cophenet` (renamed from `compare_tip_distances`) of `TreeNode` to accept a random seed or random generator to generate the shuffling function, which ensures output reproducibility ([#2118](https://github.com/scikit-bio/scikit-bio/pull/2118)).
* Replaced `accumulate_to_ancestor` with `depth` under `TreeNode`. The latter has expanded functionality which covers the default behavior of the former.
* Added beta diversity metric `jensenshannon`, which calculates Jensen-Shannon distance. Thank @quliping for suggesting this in [#2125](https://github.com/scikit-bio/scikit-bio/pull/2125).
* Added parameter `include_self` to `TreeNode.ancestors` to optionally include the initial node in the path (default: False) ([#2135](https://github.com/scikit-bio/scikit-bio/pull/2135)).
* Added parameter `seed` to functions `pcoa`, `anosim`, `permanova`, `permdisp`, `randdm`, `lladser_pe`, `lladser_ci`, `isubsample`, `subsample_power`, `subsample_paired_power`, `paired_subsamples` and `hommola_cospeciation` to accept a random seed or random generator to ensure output reproducibility ([#2120](https://github.com/scikit-bio/scikit-bio/pull/2120) and [#2129](https://github.com/scikit-bio/scikit-bio/pull/2129)).
* Made the `IORegistry` sniffer only attempt file formats which are logical given a specific object, thus improving reading efficiency.
* Allowed the `number_of_dimensions` parameter in the function `pcoa` to accept float values between 0 and 1 to capture fractional cumulative variance. 

### Bug fixes

* Fixed a bug in `TreeNode.find` which returns the input node object even if it's not in the current tree ([#2153](https://github.com/scikit-bio/scikit-bio/pull/2153)).
* Fixed a bug in `TreeNode.get_max_distance` which returns tip names instead of tip instances when there are single-child nodes in the tree ([#2144](https://github.com/scikit-bio/scikit-bio/pull/2144)).
* Fixed an issue in `subsets` and `cophenet` (renamed from `tip_tip_distances`) of `TreeNode` which leaves remnant attributes at each node after execution ([#2144](https://github.com/scikit-bio/scikit-bio/pull/2144)).
* Fixed a bug in `TreeNode.compare_rfd` which raises an error if taxa of the two trees are not subsets of each other ([#2144](https://github.com/scikit-bio/scikit-bio/pull/2144)).
* Fixed a bug in `TreeNode.compare_subsets` which includes the full set (not a subset) of shared taxa between two trees if a basal clade of either tree consists of entirely unshared taxa ([#2144](https://github.com/scikit-bio/scikit-bio/pull/2144)).
* Fixed a bug in `TreeNode.lca` which returns the parent of input node X instead of X itself if X is ancestral to other input nodes ([#2132](https://github.com/scikit-bio/scikit-bio/pull/2132)).
* Fixed a bug in `TreeNode.find_all` which does not look for other nodes with the same name if a `TreeNode` instance is provided, as in contrast to what the documentation claims ([#2099](https://github.com/scikit-bio/scikit-bio/pull/2099)).
* Fixed a bug in `skbio.io.format.embed` which was not correctly updating the idptr sizing. ([#2100](https://github.com/scikit-bio/scikit-bio/pull/2100)).
* Fixed a bug in `TreeNode.unrooted_move` which does not respect specified branch attributes ([#2103](https://github.com/scikit-bio/scikit-bio/pull/2103)).
* Fixed a bug in `skbio.diversity.get_beta_diversity_metrics` which does not display metrics other than UniFrac ([#2126](https://github.com/scikit-bio/scikit-bio/pull/2126)).
* Raises an error when beta diversity metric `mahalanobis` is called but sample number is smaller than or equal to feature number in the data. Thank @quliping for noting this in [#2125](https://github.com/scikit-bio/scikit-bio/pull/2125).
* Fixed a bug in `io.format.fasta` that improperly handled sequences containing spaces. ([#2156](https://github.com/scikit-bio/scikit-bio/pull/2156))

### Miscellaneous

* Added a parameter `warn_neg_eigval` to `pcoa` and `permdisp` to control when to raise a warning when negative eigenvalues are encountered. The default setting is more relaxed than the previous behavior, therefore warnings will not be raised when the negative eigenvalues are small in magnitude, which is the case in many real-world scenarios [#2154](https://github.com/scikit-bio/scikit-bio/pull/2154).
* Refactored `dirmult_ttest` to use a separate function for fitting data to Dirichlet-multinomial distribution ([#2113](https://github.com/scikit-bio/scikit-bio/pull/2113))
* Remodeled documentation. Special methods (previously referred to as built-in methods) and inherited methods of a class no longer have separate stub pages. This significantly reduced the total number of webpages in the documentation ([#2110](https://github.com/scikit-bio/scikit-bio/pull/2110)).
* Renamed `invalidate_caches` as `clear_caches` under `TreeNode`, because the caches are indeed deleted rather than marked as obsolete. The old name is preserved as an alias ([#2099](https://github.com/scikit-bio/scikit-bio/pull/2099)).
* Renamed `remove_deleted` as `remove_by_func` under `TreeNode`. The old name is preserved as an alias ([#2103](https://github.com/scikit-bio/scikit-bio/pull/2103)).
* Renamed `descending_branch_length` as `total_length` under `TreeNode`. The old name is preserved as an alias.
* Under `TreeNode`, renamed `get_max_distance` as `maxdist`. Renamed `tip_tip_distances` as `cophenet`. Renamed `compare_tip_distances` as `compare_cophenet`. The new names are consistent with SciPy's relevant functions and the main body of the literature. The old names are preserved as aliases.

### Deprecated functionality

* Method `TreeNode.subtree` is deprecated. It will become a private member in version 0.7.0 ([#2103](https://github.com/scikit-bio/scikit-bio/pull/2103)).

### Backward-incompatible changes

* Dropped support for Python 3.8 as it has reached end-of-life (EOL). scikit-bio may still be installed under Python 3.8 and will likely work, but the development team no longer guarantee that all functionality will work as intended.
* Removed `skbio.util.SkbioWarning`. Now there are no specific warnings to scikit-bio.
* Removed `skbio.util.EfficiencyWarning`. Previously it was only used in the Python implementations of pairwise sequence alignment algorithms. The new code replaced it with `PendingDeprecationWarning`.
* Removed `skbio.util.RepresentationWarning`. Previously it was only used in `TreeNode.tip_tip_distances` when a node has no branch length. The new code removed this behavior ([#2152](https://github.com/scikit-bio/scikit-bio/pull/2152)).


## Version 0.6.2

### Features

* Added Greedy Minimum Evolution (GME) function for phylogenetic reconstruction ([#2087](https://github.com/scikit-bio/scikit-bio/pull/2087)).
* Added support for Microsoft Windows operating system. ([#2071](https://github.com/scikit-bio/scikit-bio/pull/2071), [#2068](https://github.com/scikit-bio/scikit-bio/pull/2068),
[#2067](https://github.com/scikit-bio/scikit-bio/pull/2067), [#2061](https://github.com/scikit-bio/scikit-bio/pull/2061), [#2046](https://github.com/scikit-bio/scikit-bio/pull/2046),
[#2040](https://github.com/scikit-bio/scikit-bio/pull/2040), [#2036](https://github.com/scikit-bio/scikit-bio/pull/2036), [#2034](https://github.com/scikit-bio/scikit-bio/pull/2034),
[#2032](https://github.com/scikit-bio/scikit-bio/pull/2032), [#2005](https://github.com/scikit-bio/scikit-bio/pull/2005))
* Added alpha diversity metrics: Hill number (`hill`), Renyi entropy (`renyi`) and Tsallis entropy (`tsallis`) ([#2074](https://github.com/scikit-bio/scikit-bio/pull/2074)).
* Added `rename` method for `OrdinationResults` and `DissimilarityMatrix` classes ([#2027](https://github.com/scikit-bio/scikit-bio/pull/2027), [#2085](https://github.com/scikit-bio/scikit-bio/pull/2085)).
* Added `nni` function for phylogenetic tree rearrangement using nearest neighbor interchange (NNI) ([#2050](https://github.com/scikit-bio/scikit-bio/pull/2050)).
* Added method `TreeNode.unrooted_move`, which resembles `TreeNode.unrooted_copy` but rearranges the tree in place, thus avoid making copies of the nodes ([#2073](https://github.com/scikit-bio/scikit-bio/pull/2073)).
* Added method `TreeNode.root_by_outgroup`, which reroots a tree according to a given outgroup ([#2073](https://github.com/scikit-bio/scikit-bio/pull/2073)).
* Added method `TreeNode.unroot`, which converts a rooted tree into unrooted by trifucating its root ([#2073](https://github.com/scikit-bio/scikit-bio/pull/2073)).
* Added method `TreeNode.insert`, which inserts a node into the branch connecting self and its parent ([#2073](https://github.com/scikit-bio/scikit-bio/pull/2073)).

### Performance enhancements

* The time and memory efficiency of `TreeNode` has been significantly improved by making its caching mechanism lazy ([#2082](https://github.com/scikit-bio/scikit-bio/pull/2082)).
* `Treenode.copy` and `TreeNode.unrooted_copy` can now perform shallow copy of a tree in addition to deep copy.
* `TreeNode.unrooted_copy` can now copy all attributes of the nodes, in addition to name and length ([#2073](https://github.com/scikit-bio/scikit-bio/pull/2073)).
* Paremter `above` was added to `TreeNode.root_at`, such that the user can root the tree within the branch connecting the given node and its parent, thereby creating a rooted tree ([#2073](https://github.com/scikit-bio/scikit-bio/pull/2073)).
* Parameter `branch_attrs` was added to the `unrooted_copy`, `root_at`, and `root_at_midpoint` methods of `TreeNode`, such that the user can customize which node attributes should be considered as branch attributes and treated accordingly during the rerooting operation. The default behavior is preserved but is subject ot change in version 0.7.0 ([#2073](https://github.com/scikit-bio/scikit-bio/pull/2073)).
* Parameter `root_name` was added to the `unrooted_copy`, `root_at`, and `root_at_midpoint` methods of `TreeNode`, such that the user can customize (or omit) the name to be given to the root node. The default behavior is preserved but is subject ot change in version 0.7.0 ([#2073](https://github.com/scikit-bio/scikit-bio/pull/2073)).

### Bug fixes

* Cleared the internal node references after performing midpoint rooting (`TreeNode.root_at_midpoint`), such that a deep copy of the resulting tree will not result in infinite recursion ([#2073](https://github.com/scikit-bio/scikit-bio/pull/2073)).
* Fixed the Zenodo link in the README to always point to the most recent version ([#2078](https://github.com/scikit-bio/scikit-bio/pull/2078)).

### Miscellaneous

* Added statsmodels as a dependency of scikit-bio. It replaces some of the from-scratch statistical analyses in scikit-bio, including Welch's t-test (with confidence intervals), Benjamini-Hochberg FDR correction, and Holm-Bonferroni FDR correction ([#2049](https://github.com/scikit-bio/scikit-bio/pull/2049), ([#2063](https://github.com/scikit-bio/scikit-bio/pull/2063))).

### Deprecated functionality

* Methods `deepcopy` and `unrooted_deepcopy` of `Treenode` are deprecated. Use `copy` and `unrooted_copy` instead.


## Version 0.6.1

### Features

* NumPy 2.0 is now supported ([#2051](https://github.com/scikit-bio/scikit-bio/pull/2051])). We thank @rgommers 's advice on this ([#1964](https://github.com/scikit-bio/scikit-bio/issues/1964)).
* Added module `skbio.embedding` to provide support for storing and manipulating embeddings for biological objects, such as protein embeddings outputted from protein language models ([#2008](https://github.com/scikit-bio/scikit-bio/pull/2008])).
* Added an efficient sequence alignment path data structure `AlignPath` and its derivative `PairAlignPath` to provide a uniform interface for various multiple and pariwise alignment formats ([#2011](https://github.com/scikit-bio/scikit-bio/pull/2011)).
* Added `simpson_d` as an alias for `dominance` (Simpson's dominance index, a.k.a. Simpson's D) ([#2024](https://github.com/scikit-bio/scikit-bio/pull/2024)).
* Added `inv_simpson` (inverse Simpson index), which is equivalent to `enspie` ([#2024](https://github.com/scikit-bio/scikit-bio/pull/2024)).
* Added parameter `exp` to `shannon` to calculate the exponential of Shannon index (i.e., perplexity, or effective number of species) ([#2024](https://github.com/scikit-bio/scikit-bio/pull/2024)).
* Added parameter `finite` to Simpson's _D_ (`dominance`) and derived metrics (`simpson`, `simpson_e` and `inv_simpson`) to correct for finite samples ([#2024](https://github.com/scikit-bio/scikit-bio/pull/2024)).
* Added support for dictionary and pandas DataFrame as input for `TreeNode.from_taxonomy` ([#2042](https://github.com/scikit-bio/scikit-bio/pull/2042)).

### Performance enhancements

* `subsample_counts` now uses an optimized method from `biom-format` ([#2016](https://github.com/scikit-bio/scikit-bio/pull/2016)).
* Improved efficiency of counts matrix and vector validation prior to calculating community diversity metrics ([#2024](https://github.com/scikit-bio/scikit-bio/pull/2024)).

### Miscellaneous

* Default logarithm base of Shannon index (`shannon`) was changed from 2 to e. This is to ensure consistency with other Shannon-based metrics (`pielou_e`), and with literature and implementations in the field. Meanwhile, parameter `base` was added to `pielou_e` such that the user can control this behavior ([#2024](https://github.com/scikit-bio/scikit-bio/pull/2024)). See discussions in [1884](https://github.com/scikit-bio/scikit-bio/issues/1884) and [2014](https://github.com/scikit-bio/scikit-bio/issues/2014).
* Improved treatment of empty communities (i.e., all taxa have zero counts, or there is no taxon) when calculating alpha diversity metrics. Most metrics will return `np.nan` and do not raise a warning due to zero division. Exceptions are metrics that describe observed counts, includng `sobs`, `singles`, `doubles` and `osd`, which return zero ([#2024](https://github.com/scikit-bio/scikit-bio/pull/2024)). See discussions in [#2014](https://github.com/scikit-bio/scikit-bio/issues/2014).
* Return values of `pielou_e` and `heip_e` were set to 1.0 for one-taxon communities, such that NaN is avoided, while honoring the definition (evenness of taxon abundance(s)) and the rationale (ratio between observed and maximum) ([#2024](https://github.com/scikit-bio/scikit-bio/pull/2024)).
* Removed hdmedians as a dependency by porting its `geomedian` function (geometric median) into scikit-bio ([#2003](https://github.com/scikit-bio/scikit-bio/pull/2003)).
* Removed 98% warnings issued during the test process ([#2045](https://github.com/scikit-bio/scikit-bio/pull/2045) and [#2037](https://github.com/scikit-bio/scikit-bio/pull/2037)).


## Version 0.6.0

### Performance enhancements

* Launched the new scikit-bio website: https://scikit.bio. The previous domain names _scikit-bio.org_ and _skbio.org_ continue to work and redirect to the new website.
* Migrated the scikit-bio website repo from the `gh-pages` branch of the `scikit-bio` repo to a standalone repo: [`scikit-bio.github.io`](https://github.com/scikit-bio/scikit-bio.github.io).
* Replaced the [Bootstrap theme](https://sphinx-bootstrap-theme.readthedocs.io/en/latest/) with the [PyData theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/) for building documentation using Sphinx. Extended this theme to the website. Customized design elements ([#1934](https://github.com/scikit-bio/scikit-bio/pull/1934)).
* Improved the calculation of Fisher's alpha diversity index (`fisher_alpha`). It is now compatible with optimizers in SciPy 1.11+. Edge cases such as all singletons can be handled correctly. Handling of errors and warnings was improved. Documentation was enriched ([#1890](https://github.com/scikit-bio/scikit-bio/pull/1890)).
* Allowed `delimiter=None` which represents whitespace of arbitrary length in reading lsmat format matrices ([#1912](https://github.com/scikit-bio/scikit-bio/pull/1912)).

### Features

* Added biom-format Table import and updated corresponding requirement files ([#1907](https://github.com/scikit-bio/scikit-bio/pull/1907)).
* Added biom-format 2.1.0 IO support ([#1984](https://github.com/scikit-bio/scikit-bio/pull/1984)).
* Added `Table` support to `alpha_diversity` and `beta_diversity` drivers ([#1984](https://github.com/scikit-bio/scikit-bio/pull/1984)).
* Implemented a mechanism to automatically build documentation and/or homepage and deploy them to the website ([#1934](https://github.com/scikit-bio/scikit-bio/pull/1934)).
* Added the Benjamini-Hochberg method as an option for FDR correction (in addition to the existing Holm-Bonferroni method) for `ancom` and `dirmult_ttest` ([#1988](https://github.com/scikit-bio/scikit-bio/pull/1988)).
* Added function `dirmult_ttest`, which performs differential abundance test using a Dirichilet multinomial distribution. This function mirrors the method provided by ALDEx2 ([#1956](https://github.com/scikit-bio/scikit-bio/pull/1956)).
* Added method `Sequence.to_indices` to convert a sequence into a vector of indices of characters in an alphabet (can be from a substitution matrix) or unique characters observed in the sequence. Supports gap masking and wildcard substitution ([#1917](https://github.com/scikit-bio/scikit-bio/pull/1917)).
* Added class `SubstitutionMatrix` to support subsitution matrices for nucleotides, amino acids are more general cases ([#1913](https://github.com/scikit-bio/scikit-bio/pull/1913)).
* Added alpha diversity metric `sobs`, which is the observed species richness (S_{obs}) of a sample. `sobs` will replace `observed_otus`, which uses the historical term "OTU". Also added metric `observed_features` to be compatible with the QIIME 2 terminology. All three metrics are equivalent ([#1902](https://github.com/scikit-bio/scikit-bio/pull/1902)).
* `beta_diversity` now supports use of Pandas a `DataFrame` index, issue [#1808](https://github.com/scikit-bio/scikit-bio/issues/1808).
* Added alpha diversity metric `phydiv`, which is a generalized phylogenetic diversity (PD) framework permitting unrooted or rooted tree, unweighted or weighted by abundance, and an exponent parameter of the weight term ([#1893](https://github.com/scikit-bio/scikit-bio/pull/1893)).
* Adopted NumPy's new random generator `np.random.Generator` (see [NEP 19](https://numpy.org/neps/nep-0019-rng-policy.html)) ([#1889](https://github.com/scikit-bio/scikit-bio/pull/1889)).
* SciPy 1.11+ is now supported ([#1887](https://github.com/scikit-bio/scikit-bio/pull/1887)).
* Removed IPython as a dependency. Scikit-bio continues to support displaying plots in IPython, but it no longer requires importing IPython functionality ([#1901](https://github.com/scikit-bio/scikit-bio/pull/1901)).
* Made Matplotlib an optional dependency. Scikit-bio no longer requires Matplotlib except for plotting, during which it attempts to import Matplotlib if it is present in the system, and raises an error if not ([#1901](https://github.com/scikit-bio/scikit-bio/pull/1901)).
* Ported the QIIME 2 metadata object into skbio. ([#1929](https://github.com/scikit-bio/scikit-bio/pull/1929))
* Python 3.12+ is now supported, thank you @actapia ([#1930](https://github.com/scikit-bio/scikit-bio/pull/1930))
* Introduced native character conversion ([#1971])(https://github.com/scikit-bio/scikit-bio/pull/1971)

### Backward-incompatible changes [experimental]

* Beta diversity metric `kulsinski` was removed. This was motivated by that SciPy replaced this distance metric with `kulczynski1` in version 1.11 (see SciPy issue [#2009](https://github.com/scipy/scipy/issues/2009)), and that both metrics do not return 0 on two identical vectors ([#1887](https://github.com/scikit-bio/scikit-bio/pull/1887)).

### Bug fixes

* Fixed documentation interface of `vlr` and relevant functions ([#1934](https://github.com/scikit-bio/scikit-bio/pull/1934)).
* Fixed broken link in documentation of Simpson's evenness index. See issue [#1923](https://github.com/scikit-bio/scikit-bio/issues/1923).
* Safely handle `Sequence.iter_kmers` where `k` is greater than the sequence length ([#1723](https://github.com/scikit-bio/scikit-bio/issues/1723))
* Re-enabled OpenMP support, which has been mistakenly disabled in 0.5.8 ([#1874](https://github.com/scikit-bio/scikit-bio/pull/1874))
* `permanova` and `permdist` operate on a `DistanceMatrix` and a grouping object. Element IDs must be synchronized to compare correct sets of pairwise distances. This failed in case the grouping was provided as a `pandas.Series`, because it was interpreted as an ordered `list` and indices were ignored (see issue [#1877](https://github.com/scikit-bio/scikit-bio/issues/1877) for an example). Note: `pandas.DataFrame` was handled correctly. This behavior has been fixed with PR [#1879](https://github.com/scikit-bio/scikit-bio/pull/1879)
* Fixed slicing for `TabularMSALoc` on Python 3.12. See issue [#1926](https://github.com/scikit-bio/scikit-bio/issues/1926).

### Miscellaneous

* Replaced the historical term "OTU" with the more generic term "taxon" (plural: "taxa"). As a consequence, the parameter "otu_ids" in phylogenetic alpha and beta diversity metrics was replaced by "taxa". Meanwhile, the old parameter "otu_ids" is still kept as an alias of "taxa" for backward compatibility. However it will be removed in a future release.
* Revised contributor's guidelines.
* Renamed function `multiplicative_replacement` as `multi_replace` for conciseness ([#1988](https://github.com/scikit-bio/scikit-bio/pull/1988)).
* Renamed parameter `multiple_comparisons_correction` as `p_adjust` of function `ancom` for conciseness ([#1988](https://github.com/scikit-bio/scikit-bio/pull/1988)).
* Enabled code coverage reporting via Codecov. See [#1954](https://github.com/scikit-bio/scikit-bio/pull/1954).
* Renamed the default branch from "master" to "main". See [#1953](https://github.com/scikit-bio/scikit-bio/pull/1953).
* Enabled subclassing of DNA, RNA and Protein classes to allow secondary development.
* Dropped support for NumPy < 1.17.0 in order to utilize the new random generator.
* Use CYTHON by default during build ([#1874](https://github.com/scikit-bio/scikit-bio/pull/1874))
* Implemented augmented assignments proposed in issue [#1789](https://github.com/scikit-bio/scikit-bio/issues/1789)
* Incorporated Ruff's formatting and linting via pre-commit hooks and GitHub Actions. See PR [#1924](https://github.com/scikit-bio/scikit-bio/pull/1924).
* Improved docstrings for functions accross the entire codebase. See [#1933](https://github.com/scikit-bio/scikit-bio/pull/1933) and [#1940](https://github.com/scikit-bio/scikit-bio/pull/1940)
* Removed API lifecycle decorators in favor of deprecation warnings. See [#1916](https://github.com/scikit-bio/scikit-bio/issues/1916)


## Version 0.5.9

### Features

* Adding Variance log ratio estimators in `skbio.stats.composition.vlr` and `skbio.stats.composition.pairwise_vlr` ([#1803](https://github.com/scikit-bio/scikit-bio/pull/1803))
* Added `skbio.stats.composition.tree_basis` to construct ILR bases from `TreeNode` objects. ([#1862](https://github.com/scikit-bio/scikit-bio/pull/1862))
* `IntervalMetadata.query` now defaults to obtaining all results, see [#1817](https://github.com/scikit-bio/scikit-bio/issues/1817).

### Backward-incompatible changes [experimental]
* With the introduction of the `tree_basis` object, the ILR bases are now represented in log-odds coordinates rather than in probabilities to minimize issues with numerical stability. Furthermore, the `ilr` and `ilr_inv` functions now takes the `basis` input parameter in terms of log-odds coordinates. This affects the `skbio.stats.composition.sbp_basis` as well. ([#1862](https://github.com/scikit-bio/scikit-bio/pull/1862))

### Important

* Complex multiple axis indexing operations with `TabularMSA` have been removed from testing due to incompatibilities with modern versions of Pandas. ([#1851](https://github.com/scikit-bio/scikit-bio/pull/1851))
* Pinning `scipy <= 1.10.1` ([#1851](https://github.com/scikit-bio/scikit-bio/pull/1867))

### Bug fixes

* Fixed a bug that caused build failure on the ARM64 microarchitecture due to floating-point number handling. ([#1859](https://github.com/scikit-bio/scikit-bio/pull/1859))
* Never let the Gini index go below 0.0, see [#1844](https://github.com/scikit-bio/scikit-bio/issue/1844).
* Fixed bug [#1847](https://github.com/scikit-bio/scikit-bio/issues/1847) in which the edge from the root was inadvertantly included in the calculation for `descending_branch_length`

### Miscellaneous

* Replaced dependencies `CacheControl` and `lockfile` with `requests` to avoid a dependency inconsistency issue of the former. (See [#1863](https://github.com/scikit-bio/scikit-bio/pull/1863), merged in [#1859](https://github.com/scikit-bio/scikit-bio/pull/1859))
* Updated installation instructions for developers in `CONTRIBUTING.md` ([#1860](https://github.com/scikit-bio/scikit-bio/pull/1860))

## Version 0.5.8

### Features

* Added NCBI taxonomy database dump format (`taxdump`) ([#1810](https://github.com/scikit-bio/scikit-bio/pull/1810)).
* Added `TreeNode.from_taxdump` for converting taxdump into a tree ([#1810](https://github.com/scikit-bio/scikit-bio/pull/1810)).
* scikit-learn has been removed as a dependency. This was a fairly heavy-weight dependency that was providing minor functionality to scikit-bio. The critical components have been implemented in scikit-bio directly, and the non-criticial components are listed under "Backward-incompatible changes [experimental]".
* Python 3.11 is now supported.

### Backward-incompatible changes [experimental]
* With the removal of the scikit-learn dependency, three beta diversity metric names can no longer be specified. These are `wminkowski`, `nan_euclidean`, and `haversine`. On testing, `wminkowski` and `haversine` did not work through `skbio.diversity.beta_diversity` (or `sklearn.metrics.pairwise_distances`). The former was deprecated in favor of calling `minkowski` with a vector of weights provided as kwarg `w` (example below), and the latter does not work with data of this shape. `nan_euclidean` can still be accessed fron scikit-learn directly if needed, if a user installs scikit-learn in their environment (example below).

    ```
    counts = [[23, 64, 14, 0, 0, 3, 1],
            [0, 3, 35, 42, 0, 12, 1],
            [0, 5, 5, 0, 40, 40, 0],
            [44, 35, 9, 0, 1, 0, 0],
            [0, 2, 8, 0, 35, 45, 1],
            [0, 0, 25, 35, 0, 19, 0],
            [88, 31, 0, 5, 5, 5, 5],
            [44, 39, 0, 0, 0, 0, 0]]

    # new mechanism of accessing wminkowski
    from skbio.diversity import beta_diversity
    beta_diversity("minkowski", counts, w=[1,1,1,1,1,1,2])

    # accessing nan_euclidean through scikit-learn directly
    import skbio
    from sklearn.metrics import pairwise_distances
    sklearn_dm = pairwise_distances(counts, metric="nan_euclidean")
    skbio_dm = skbio.DistanceMatrix(sklearn_dm)
    ```

### Deprecated functionality [experimental]
* `skbio.alignment.local_pairwise_align_ssw` has been deprecated ([#1814](https://github.com/scikit-bio/scikit-bio/issues/1814)) and will be removed or replaced in scikit-bio 0.6.0.

### Bug fixes
* Use `oldest-supported-numpy` as build dependency. This fixes problems with environments that use an older version of numpy than the one used to build scikit-bio ([#1813](https://github.com/scikit-bio/scikit-bio/pull/1813)).


## Version 0.5.7

### Features

* Introduce support for Python 3.10 ([#1801](https://github.com/scikit-bio/scikit-bio/pull/1801)).
* Tentative support for Apple M1 ([#1709](https://github.com/scikit-bio/scikit-bio/pull/1709)).
* Added support for reading and writing a binary distance matrix object format. ([#1716](https://github.com/scikit-bio/scikit-bio/pull/1716))
* Added support for `np.float32` with `DissimilarityMatrix` objects.
* Added support for method and number_of_dimensions to permdisp reducing the runtime by 100x at 4000 samples, [issue #1769](https://github.com/scikit-bio/scikit-bio/pull/1769).
* OrdinationResults object is now accepted as input for permdisp.

### Performance enhancements

* Avoid an implicit data copy on construction of `DissimilarityMatrix` objects.
* Avoid validation on copy of `DissimilarityMatrix` and `DistanceMatrix` objects, see [PR #1747](https://github.com/scikit-bio/scikit-bio/pull/1747)
* Use an optimized version of symmetry check in DistanceMatrix, see [PR #1747](https://github.com/scikit-bio/scikit-bio/pull/1747)
* Avoid performing filtering when ids are identical, see [PR #1752](https://github.com/scikit-bio/scikit-bio/pull/1752)
* center_distance_matrix has been re-implemented in cython for both speed and memory use. Indirectly speeds up pcoa [PR #1749](https://github.com/scikit-bio/scikit-bio/pull/1749)
* Use a memory-optimized version of permute in DistanceMatrix, see [PR #1756](https://github.com/scikit-bio/scikit-bio/pull/1756).
* Refactor pearson and spearman skbio.stats.distance.mantel implementations to drastically improve memory locality. Also cache intermediate results that are invariant across permutations, see [PR #1756](https://github.com/scikit-bio/scikit-bio/pull/1756).
* Refactor permanova to remove intermediate buffers and cythonize the internals, see [PR #1768](https://github.com/scikit-bio/scikit-bio/pull/1768).

### Bug fixes

* Fix windows and 32bit incompatibility in `unweighted_unifrac`.

### Miscellaneous

* Python 3.6 has been removed from our testing matrix.
* Specify build dependencies in pyproject.toml. This allows the package to be installed without having to first manually install numpy.
* Update hdmedians package to a version which doesn't require an initial manual numpy install.
* Now buildable on non-x86 platforms due to use of the [SIMD Everywhere](https://github.com/simd-everywhere/simde) library.
* Regenerate Cython wrapper by default to avoid incompatibilities with installed CPython.
* Update documentation for the `skbio.stats.composition.ancom` function. ([#1741](https://github.com/scikit-bio/scikit-bio/pull/1741))

## Version 0.5.6

### Features

* Added option to return a capture group compiled regex pattern to any class inheriting ``GrammaredSequence`` through the ``to_regex`` method. ([#1431](https://github.com/scikit-bio/scikit-bio/issues/1431))

* Added `Dissimilarity.within` and `.between` to obtain the respective distances and express them as a `DataFrame`. ([#1662](https://github.com/scikit-bio/scikit-bio/pull/1662))

* Added Kendall Tau as possible correlation method in the `skbio.stats.distance.mantel` function ([#1675](https://github.com/scikit-bio/scikit-bio/issues/1675)).

* Added support for IUPAC amino acid codes U (selenocysteine), O (pyrrolysine), and J (leucine or isoleucine). ([#1576](https://github.com/scikit-bio/scikit-bio/issues/1576)

### Backward-incompatible changes [stable]

### Backward-incompatible changes [experimental]

* Changed `skbio.tree.TreeNode.support` from a method to a property.
* Added `assign_supports` method to `skbio.tree.TreeNode` to extract branch support values from node labels.
* Modified the way a node's label is printed: `support:name` if both exist, or `support` or `name` if either exists.

### Performance enhancements

### Bug fixes

* Require `Sphinx <= 3.0`. Newer Sphinx versions caused build errors. [#1719](https://github.com/scikit-bio/scikit-bio/pull/1719)

* * `skbio.stats.ordination` tests have been relaxed. ([#1713](https://github.com/scikit-bio/scikit-bio/issues/1713))

* Fixes build errors for newer versions of NumPy, Pandas, and SciPy.

* Corrected a criticial bug in `skbio.alignment.StripedSmithWaterman`/`skbio.alignment.local_pairwise_align_ssw` which would cause the formatting of the aligned sequences to misplace gap characters by the number of gap characters present in the opposing aligned sequence up to that point. This was caused by a faulty implementation of CIGAR string parsing, see [#1679](https://github.com/scikit-bio/scikit-bio/pull/1679) for full details.

* Fixes build errors for newer versions of NumPy, Pandas, and SciPy.

* Corrected a criticial bug in `skbio.alignment.StripedSmithWaterman`/`skbio.alignment.local_pairwise_align_ssw` which would cause the formatting of the aligned sequences to misplace gap characters by the number of gap characters present in the opposing aligned sequence up to that point. This was caused by a faulty implementation of CIGAR string parsing, see [#1679](https://github.com/scikit-bio/scikit-bio/pull/1679) for full details.

### Deprecated functionality [stable]

### Deprecated functionality [experimental]

### Miscellaneous

* `skbio.diversity.beta_diversity` now accepts a pandas DataFrame as input.

* Avoid pandas 1.0.0 import warning ([#1688](https://github.com/scikit-bio/scikit-bio/issues/1688))

* Added support for Python 3.8 and dropped support for Python 3.5.

* This version now depends on `scipy >= 1.3` and `pandas >= 1.0`.

## Version 0.5.5 (2018-12-10)

### Features

* `skbio.stats.composition` now has methods to compute additive log-ratio transformation and inverse additive log-ratio transformation (`alr`, `alr_inv`) as well as a method to build a basis from a sequential binary partition (`sbp_basis`).

### Backward-incompatible changes [stable]

### Backward-incompatible changes [experimental]

### Performance enhancements

### Bug fixes

### Deprecated functionality [stable]

### Deprecated functionality [experimental]

### Miscellaneous
* Python 3.6 and 3.7 compatibility is now supported

* A pytest runner is shipped with every installation ([#1633](https://github.com/scikit-bio/scikit-bio/pull/1633))

* The nosetest framework has been replaced in favor of pytest ([#1624](https://github.com/scikit-bio/scikit-bio/pull/1624))

* The numpy docs are deprecated in favor of [Napoleon](http://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) ([#1629](https://github.com/scikit-bio/scikit-bio/pull/1629))

* This version is now compatible with numpy >= 1.17.0 and Pandas >= 0.23. ([#1627](https://github.com/scikit-bio/scikit-bio/pull/1627))

## Version 0.5.4 (2018-08-23)

### Features

* Added `FSVD`, an alternative fast heuristic method to perform Principal Coordinates Analysis, to `skbio.stats.ordination.pcoa`.

### Backward-incompatible changes [stable]

### Backward-incompatible changes [experimental]

### Performance enhancements

* Added optimized utility methods `f_matrix_inplace` and `e_matrix_inplace` which perform `f_matrix` and `e_matrix` computations in-place and are used by the new `center_distance_matrix` method in `skbio.stats.ordination`.

### Bug fixes

### Deprecated functionality [stable]

### Deprecated functionality [experimental]

### Miscellaneous

## Version 0.5.3 (2018-08-07)

### Features
* Added `unpack` and `unpack_by_func` methods to `skbio.tree.TreeNode` to unpack one or multiple internal nodes. The `unpack` operation removes an internal node and regrafts its children to its parent while retaining the overall length. ([#1572](https://github.com/scikit-bio/scikit-bio/pull/1572))
* Added `support` to `skbio.tree.TreeNode` to return the support value of a node.
* Added `permdisp` to `skbio.stats.distance` to test for the homogeniety of groups. ([#1228](https://github.com/scikit-bio/scikit-bio/issues/1228)).

* Added `pcoa_biplot` to `skbio.stats.ordination` to project descriptors into a PCoA plot.

* Fixed pandas to 0.22.0 due to this: https://github.com/pandas-dev/pandas/issues/20527

### Backward-incompatible changes [stable]

### Backward-incompatible changes [experimental]

### Performance enhancements

### Bug fixes

* Relaxing type checking in diversity calculations.  ([#1583](https://github.com/scikit-bio/scikit-bio/issues/1583)).

### Deprecated functionality [stable]

### Deprecated functionality [experimental]

### Miscellaneous


## Version 0.5.2 (2018-04-18)

### Features
* Added ``skbio.io.format.embl`` for reading and writing EMBL files for ``DNA``, ``RNA`` and ``Sequence`` classes.

* Removing ValueError check in `skbio.stats._subsample.subsample_counts` when `replace=True` and `n` is greater than the number of items in counts.  [#1527](https://github.com/scikit-bio/scikit-bio/pull/1527)

* Added ``skbio.io.format.gff3`` for reading and writing GFF3 files for ``DNA``, ``Sequence``, and ``IntervalMetadata`` classes. ([#1450](https://github.com/scikit-bio/scikit-bio/pull/1450))

* `skbio.metadata.IntervalMetadata` constructor has a new keyword argument, `copy_from`, for creating an `IntervalMetadata` object from an existing `IntervalMetadata` object with specified `upper_bound`.

* `skbio.metadata.IntervalMetadata` constructor allows `None` as a valid value for `upper_bound`. An `upper_bound` of `None` means that the `IntervalMetadata` object has no upper bound.

* `skbio.metadata.IntervalMetadata.drop` has a new boolean parameter `negate` to indicate whether to drop or keep the specified `Interval` objects.

### Backward-incompatible changes [stable]

### Backward-incompatible changes [experimental]

### Performance enhancements
* `skbio.tree.nj` wall-clock runtime was decreased by 99% for a 500x500 distance matrix and 93% for a 100x100 distance matrix. ([#1512](https://github.com/scikit-bio/scikit-bio/pull/1512), [#1513](https://github.com/scikit-bio/scikit-bio/pull/1513))

### Bug fixes
* The `include_self` parameter was not being honored in `skbio.TreeNode.tips`. The scope of this bug was that if `TreeNode.tips` was called on a tip, it would always result in an empty `list` when unrolled.

* In `skbio.stats.ordination.ca`, `proportion_explained` was missing in the returned `OrdinationResults` object. ([#1345](https://github.com/scikit-bio/scikit-bio/issues/1345))

* `skbio.diversity.beta_diversity` now handles qualitative metrics as expected such that `beta_diversity('jaccard', mat) == beta_diversity('jaccard', mat > 0)`. Please see [#1549](https://github.com/scikit-bio/scikit-bio/issues/1549) for further detail.

* `skbio.stats.ordination.rda` The occasional column mismatch in output `biplot_scores` is fixed ([#1519](https://github.com/scikit-bio/scikit-bio/issues/1519)).

### Deprecated functionality [stable]

### Deprecated functionality [experimental]

### Miscellaneous
* scikit-bio now depends on pandas >= 0.19.2, and is compatible with newer pandas versions (e.g. 0.20.3) that were previously incompatible.
* scikit-bio now depends on `numpy >= 1.17.0, < 1.14.0` for compatibility with Python 3.4, 3.5, and 3.6 and the available numpy conda packages in `defaults` and `conda-forge` channels.
* added support for running tests from `setup.py`. Both `python setup.py nosetests` and `python setup.py test` are now supported, however `python setup.py test` will only run a subset of the full test suite. ([#1341](https://github.com/scikit-bio/scikit-bio/issues/1341))

## Version 0.5.1 (2016-11-12)

### Features
* Added `IntervalMetadata` and `Interval` classes in `skbio.metadata` to store, query, and manipulate information of a sub-region of a sequence. ([#1414](https://github.com/scikit-bio/scikit-bio/issues/1414))
* `Sequence` and its child classes (including `GrammaredSequence`, `RNA`, `DNA`, `Protein`) now accept `IntervalMetadata` in their constructor API. Some of their relevant methods are also updated accordingly. ([#1430](https://github.com/scikit-bio/scikit-bio/pull/1430))
* GenBank parser now reads and writes `Sequence` or its subclass objects with `IntervalMetadata`. ([#1440](https://github.com/scikit-bio/scikit-bio/pull/1440))
* `DissimilarityMatrix` now has a new constructor method called `from_iterable`. ([#1343](https://github.com/scikit-bio/scikit-bio/issues/1343)).
* `DissimilarityMatrix` now allows non-hollow matrices. ([#1343](https://github.com/scikit-bio/scikit-bio/issues/1343)).
* `DistanceMatrix.from_iterable` now accepts a `validate=True` parameter. ([#1343](https://github.com/scikit-bio/scikit-bio/issues/1343)).
* ``DistanceMatrix`` now has a new method called ``to_series`` to create a ``pandas.Series`` from a ``DistanceMatrix`` ([#1397](https://github.com/scikit-bio/scikit-bio/issues/1397)).
* Added parallel beta diversity calculation support via `skbio.diversity.block_beta_diversity`. The issue and idea is discussed in ([#1181](https://github.com/scikit-bio/scikit-bio/issues/1181), while the actual code changes are in [#1352](https://github.com/scikit-bio/scikit-bio/pull/1352)).


### Backward-incompatible changes [stable]
* The constructor API for `Sequence` and its child classes (including `GrammaredSequence`, `RNA`, `DNA`, `Protein`) are changed from `(sequence, metadata=None, positional_metadata=None, lowercase=False)` to `(sequence, metadata=None, positional_metadata=None, interval_metadata=None, lowercase=False)`

  The changes are made to allow these classes to adopt `IntervalMetadata` object for interval features on the sequence. The `interval_metadata` parameter is added imediately after `positional_metadata` instead of appended to the end, because it is more natural and logical and, more importantly, because it is unlikely in practice to break user code. A user's code would break only if they had supplied `metadata`, `postional_metadata`, and `lowercase` parameters positionally. In the unlikely event that this happens, users will get an error telling them a bool isn't a valid `IntervalMetadata` type, so it won't silently produce buggy behavior.

### Backward-incompatible changes [experimental]
* Modifying basis handling in `skbio.stats.composition.ilr_inv` prior to checking for orthogonality.  Now the basis is strictly assumed to be in the Aitchison simplex.
* `DistanceMatrix.from_iterable` default behavior is now to validate matrix by computing all pairwise distances. Pass `validate=False` to get the previous behavior (no validation, but faster execution).([#1343](https://github.com/scikit-bio/scikit-bio/issues/1343)).
* GenBank I/O now parses sequence features into the attribute of `interval_metadata` instead of `positiona_metadata`. And the key of `FEATURES` is removed from `metadata` attribute.

### Performance enhancements
* `TreeNode.shear` was rewritten for approximately a 25% performance increase. ([#1399](https://github.com/scikit-bio/scikit-bio/pull/1399))
* The `IntervalMetadata` allows dramatic decrease in memory usage in reading GenBank files of feature rich sequences. ([#1159](https://github.com/scikit-bio/scikit-bio/issues/1159))

### Bug fixes

* `skbio.tree.TreeNode.prune` and implicitly `skbio.tree.TreeNode.shear` were not handling a situation in which a parent was validly removed during pruning operations as may happen if the resulting subtree does not include the root. Previously, an `AttributeError` would raise as `parent` would be `None` in this situation.
* numpy linking was fixed for installation under El Capitan.
* A bug was introduced in #1398 into `TreeNode.prune` and fixed in #1416 in which, under the special case of a single descendent existing from the root, the resulting children parent references were not updated. The cause of the bug was a call made to `self.children.extend` as opposed to `self.extend` where the former is a `list.extend` without knowledge of the tree, while the latter is `TreeNode.extend` which is able to adjust references to `self.parent`.

### Miscellaneous

* Removed deprecated functions from `skbio.util`: `is_casava_v180_or_later`, `remove_files`, and `create_dir`.
* Removed deprecated `skbio.Sequence.copy` method.

## Version 0.5.0 (2016-06-14)

**IMPORTANT**: scikit-bio is no longer compatible with Python 2. scikit-bio is compatible with Python 3.4 and later.

### Features
* Added more descriptive error message to `skbio.io.registry` when attempting to read without specifying `into` and when there is no generator reader. ([#1326](https://github.com/scikit-bio/scikit-bio/issues/1326))
* Added support for reference tags to `skbio.io.format.stockholm` reader and writer. ([#1348](https://github.com/scikit-bio/scikit-bio/issues/1348))
* Expanded error message in `skbio.io.format.stockholm` reader when `constructor` is not passed, in order to provide better explanation to user. ([#1327](https://github.com/scikit-bio/scikit-bio/issues/1327))
* Added `skbio.sequence.distance.kmer_distance` for computing the kmer distance between two sequences. ([#913](https://github.com/scikit-bio/scikit-bio/issues/913))
* Added `skbio.sequence.Sequence.replace` for assigning a character to positions in a `Sequence`. ([#1222](https://github.com/scikit-bio/scikit-bio/issues/1222))
* Added support for `pandas.RangeIndex`, lowering the memory footprint of default integer index objects. `Sequence.positional_metadata` and `TabularMSA.positional_metadata` now use `pd.RangeIndex` as the positional metadata index. `TabularMSA` now uses `pd.RangeIndex` as the default index. Usage of `pd.RangeIndex` over the previous `pd.Int64Index` [should be transparent](http://pandas.pydata.org/pandas-docs/version/0.18.0/whatsnew.html#range-index), so these changes should be non-breaking to users. scikit-bio now depends on pandas >= 0.18.0 ([#1308](https://github.com/scikit-bio/scikit-bio/issues/1308))
* Added `reset_index=False` parameter to `TabularMSA.append` and `TabularMSA.extend` for resetting the MSA's index to the default index after appending/extending.
* Added support for partial pairwise calculations via `skbio.diversity.partial_beta_diversity`. ([#1221](https://github.com/scikit-bio/scikit-bio/issues/1221), [#1337](https://github.com/scikit-bio/scikit-bio/pull/1337)). This function is immediately deprecated as its return type will change in the future and should be used with caution in its present form (see the function's documentation for details).
* `TemporaryFile` and `NamedTemporaryFile` are now supported IO sources for `skbio.io` and related functionality.  ([#1291](https://github.com/scikit-bio/scikit-bio/issues/1291))
* Added `tree_node_class=TreeNode` parameter to `skbio.tree.majority_rule` to support returning consensus trees of type `TreeNode` (the default) or a type that has the same interface as `TreeNode` (e.g. `TreeNode` subclasses) ([#1193](https://github.com/scikit-bio/scikit-bio/pull/1193))
* `TreeNode.from_linkage_matrix` and `TreeNode.from_taxonomy` now support constructing `TreeNode` subclasses. `TreeNode.bifurcate` now supports `TreeNode` subclasses ([#1193](https://github.com/scikit-bio/scikit-bio/pull/1193))
* The `ignore_metadata` keyword has been added to `TabularMSA.iter_positions` to improve performance when metadata is not necessary.
* Pairwise aligners in `skbio.alignment` now propagate per-sequence `metadata` objects (this does not include `positional_metadata`).

### Backward-incompatible changes [stable]

### Backward-incompatible changes [experimental]
* `TabularMSA.append` and `TabularMSA.extend` now require one of `minter`, `index`, or `reset_index` to be provided when incorporating new sequences into an MSA. Previous behavior was to auto-increment the index labels if `minter` and `index` weren't provided and the MSA had a default integer index, otherwise error. Use `reset_index=True` to obtain the previous behavior in a more explicit way.
* `skbio.stats.composition.ancom` now returns two `pd.DataFrame` objects, where it previously returned one. The first contains the ANCOM test results, as before, and the second contains percentile abundances of each feature in each group. The specific percentiles that are computed and returned is controlled by the new `percentiles` parameter to `skbio.stats.composition.ancom`. In the future, this second `pd.DataFrame` will not be returned by this function, but will be available through the [contingency table API](https://github.com/scikit-bio/scikit-bio/issues/848). ([#1293](https://github.com/scikit-bio/scikit-bio/issues/1293))
* `skbio.stats.composition.ancom` now performs multiple comparisons correction by default. The previous behavior of not performing multiple comparisons correction can be achieved by passing ``multiple_comparisons_correction=None``.
* The ``reject`` column in the first ``pd.DataFrame`` returned from `skbio.stats.composition.ancom` has been renamed ``Reject null hypothesis`` for clarity. ([#1375](https://github.com/scikit-bio/scikit-bio/issues/1375))

### Bug fixes
* Fixed row and column names to `biplot_scores` in the `OrdinationResults` object from `skbio.stats.ordination`. This fix affect the `cca` and `rda` methods. ([#1322](https://github.com/scikit-bio/scikit-bio/issues/1322))
* Fixed bug when using `skbio.io.format.stockholm` reader on file with multi-line tree with no id. Previously this raised an `AttributeError`, now it correctly handles this type of tree. ([#1334](https://github.com/scikit-bio/scikit-bio/issues/1334))
* Fixed bug when reading Stockholm files with GF or GS features split over multiple lines. Previously, the feature text was simply concatenated because it was assumed to have trailing whitespace. There are examples of Stockholm files with and without trailing whitespace for multi-line features, so the `skbio.io.format.stockholm` reader now adds a single space when concatenating feature text without trailing whitespace to avoid joining words together. Multi-line trees stored as GF metadata are concatenated as they appear in the file; a space is not added when concatenating. ([#1328](https://github.com/scikit-bio/scikit-bio/issues/1328))
* Fixed bug when using `Sequence.iter_kmers` on empty `Sequence` object. Previously this raised a `ValueError`, now it returns
an empty generator.
* Fixed minor bug where adding sequences to an empty `TabularMSA` with MSA-wide `positional_metadata` would result in a `TabularMSA` object in an inconsistent state. This could happen using `TabularMSA.append` or `TabularMSA.extend`. This bug only affects a `TabularMSA` object *without* sequences that has MSA-wide `positional_metadata` (for example, `TabularMSA([], positional_metadata={'column': []})`).
* `TreeNode.distance` now handles the situation in which `self` or `other` are ancestors. Previosly, a node further up the tree was used resulting in inflated distances. ([#807](https://github.com/scikit-bio/scikit-bio/issues/807))
* `TreeNode.prune` can now handle a root with a single descendent. Previously, the root was ignored from possibly having a single descendent. ([#1247](https://github.com/scikit-bio/scikit-bio/issues/1247))
* Providing the `format` keyword to `skbio.io.read` when creating a generator with an empty file will now return an empty generator instead of raising `StopIteration`. ([#1313](https://github.com/scikit-bio/scikit-bio/issues/1313))
* `OrdinationResults` is now importable from `skbio` and `skbio.stats.ordination` and correctly linked from the documentation ([#1205](https://github.com/scikit-bio/scikit-bio/issues/1205))
* Fixed performance bug in pairwise aligners resulting in 100x worse performance than in 0.2.4.

### Deprecated functionality [stable]
* Deprecated use of the term "non-degenerate", in favor of "definite". `GrammaredSequence.nondegenerate_chars`, `GrammaredSequence.nondegenerates`, and `GrammaredSequence.has_nondegenerates` have been renamed to `GrammaredSequence.definite_chars`, `GrammaredSequence.definites`, and `GrammaredSequence.has_definites`, respectively. The old names will be removed in scikit-bio 0.5.2. Relevant affected public classes include `GrammaredSequence`, `DNA`, `RNA`, and `Protein`.

### Deprecated functionality [experimental]
* Deprecated function `skbio.util.create_dir`. This function will be removed in scikit-bio 0.5.1. Please use the Python standard library
functionality described [here](https://docs.python.org/2/library/os.html#os.makedirs). ([#833](https://github.com/scikit-bio/scikit-bio/issues/833))
* Deprecated function `skbio.util.remove_files`. This function will be removed in scikit-bio 0.5.1. Please use the Python standard library
functionality described [here](https://docs.python.org/2/library/os.html#os.remove). ([#833](https://github.com/scikit-bio/scikit-bio/issues/833))
* Deprecated function `skbio.util.is_casava_v180_or_later`. This function will be removed in 0.5.1. Functionality moved to FASTQ sniffer.
([#833](https://github.com/scikit-bio/scikit-bio/issues/833))

### Miscellaneous
* When installing scikit-bio via `pip`, numpy must now be installed first ([#1296](https://github.com/scikit-bio/scikit-bio/issues/1296))

## Version 0.4.2 (2016-02-17)

Minor maintenance release. **This is the last Python 2.7 compatible release. Future scikit-bio releases will only support Python 3.**

### Features
* Added `skbio.tree.TreeNode.bifurcate` for converting multifurcating trees into bifurcating trees. ([#896](https://github.com/scikit-bio/scikit-bio/issues/896))
* Added `skbio.io.format.stockholm` for reading Stockholm files into a `TabularMSA` and writing from a `TabularMSA`. ([#967](https://github.com/scikit-bio/scikit-bio/issues/967))
* scikit-bio `Sequence` objects have better compatibility with numpy. For example, calling `np.asarray(sequence)` now converts the sequence to a numpy array of characters (the same as calling `sequence.values`).
* Added `skbio.sequence.distance` subpackage for computing distances between scikit-bio `Sequence` objects ([#913](https://github.com/scikit-bio/scikit-bio/issues/913))
* Added ``skbio.sequence.GrammaredSequence``, which can be inherited from to create grammared sequences with custom alphabets (e.g., for use with TabularMSA) ([#1175](https://github.com/scikit-bio/scikit-bio/issues/1175))
* Added ``skbio.util.classproperty`` decorator

### Backward-incompatible changes [stable]
* When sniffing or reading a file (`skbio.io.sniff`, `skbio.io.read`, or the object-oriented `.read()` interface), passing `newline` as a keyword argument to `skbio.io.open` now raises a `TypeError`. This backward-incompatible change to a stable API is necessary because it fixes a bug (more details in bug fix section below).
* When reading a FASTQ or QSEQ file and passing `variant='solexa'`, `ValueError` is now raised instead of `NotImplementedError`. This backward-incompatible change to a stable API is necessary to avoid creating a spin-locked process due to [a bug in Python](https://bugs.python.org/issue25786). See [#1256](https://github.com/scikit-bio/scikit-bio/issues/1256) for details. This change is temporary and will be reverted to `NotImplementedError` when the bug is fixed in Python.

### Backward-incompatible changes [experimental]
* `skbio.io.format.genbank`: When reading GenBank files, the date field of the LOCUS line is no longer parsed into a `datetime.datetime` object and is left as a string. When writing GenBank files, the locus date metadata is expected to be a string instead of a `datetime.datetime` object ([#1153](https://github.com/scikit-bio/scikit-bio/issues/1153))
* `Sequence.distance` now converts the input sequence (`other`) to its type before passing both sequences to `metric`. Previous behavior was to always convert to `Sequence`.

### Bug fixes
* Fixed bug when using `Sequence.distance` or `DistanceMatrix.from_iterable` to compute distances between `Sequence` objects with differing `metadata`/`positional_metadata` and passing `metric=scipy.spatial.distance.hamming` ([#1254](https://github.com/scikit-bio/scikit-bio/issues/1254))
* Fixed performance bug when computing Hamming distances between `Sequence` objects in `DistanceMatrix.from_iterable` ([#1250](https://github.com/scikit-bio/scikit-bio/issues/1250))
* Changed `skbio.stats.composition.multiplicative_replacement` to raise an error whenever a large value of `delta` is chosen ([#1241](https://github.com/scikit-bio/scikit-bio/issues/1241))
* When sniffing or reading a file (`skbio.io.sniff`, `skbio.io.read`, or the object-oriented `.read()` interface), passing `newline` as a keyword argument to `skbio.io.open` now raises a `TypeError`. The file format's `newline` character will be used when opening the file. Previous behavior allowed overriding the format's `newline` character but this could cause issues with readers that assume newline characters are those defined by the file format (which is an entirely reasonable assumption). This bug is very unlikely to have surfaced in practice as the default `newline` behavior is *universal newlines mode*.
* DNA, RNA, and Protein are no longer inheritable because they assume an IUPAC alphabet.
* `DistanceMatrix` constructor provides more informative error message when data contains NaNs ([#1276](https://github.com/scikit-bio/scikit-bio/issues/1276))

### Miscellaneous
* Warnings raised by scikit-bio now share a common subclass ``skbio.util.SkbioWarning``.

## Version 0.4.1 (2015-12-09)

### Features
* The ``TabularMSA`` object was added to represent and operate on tabular multiple sequence alignments. This satisfies [RFC 1](https://github.com/scikit-bio/scikit-bio-rfcs/blob/master/active/001-tabular-msa.md). See the ``TabularMSA`` docs for full details.
* Added phylogenetic diversity metrics, including weighted UniFrac, unweighted UniFrac, and Faith's Phylogenetic Diversity. These are accessible as ``skbio.diversity.beta.unweighted_unifrac``, ``skbio.diversity.beta.weighted_unifrac``, and ``skbio.diversity.alpha.faith_pd``, respectively.
* Addition of the function ``skbio.diversity.alpha_diversity`` to support applying an alpha diversity metric to multiple samples in one call.
* Addition of the functions ``skbio.diversity.get_alpha_diversity_metrics`` and ``skbio.diversity.get_beta_diversity_metrics`` to support discovery of the alpha and beta diversity metrics implemented in scikit-bio.
* Added `skbio.stats.composition.ancom` function, a test for OTU differential abundance across sample categories. ([#1054](https://github.com/scikit-bio/scikit-bio/issues/1054))
* Added `skbio.io.format.blast7` for reading BLAST+ output format 7 or BLAST output format 9 files into a `pd.DataFrame`. ([#1110](https://github.com/scikit-bio/scikit-bio/issues/1110))
* Added `skbio.DissimilarityMatrix.to_data_frame` method for creating a ``pandas.DataFrame`` from a `DissimilarityMatrix` or `DistanceMatrix`. ([#757](https://github.com/scikit-bio/scikit-bio/issues/757))
* Added support for one-dimensional vector of dissimilarities in `skbio.stats.distance.DissimilarityMatrix`
constructor. ([#6240](https://github.com/scikit-bio/scikit-bio/issues/624))
* Added `skbio.io.format.blast6` for reading BLAST+ output format 6 or BLAST output format 8 files into a `pd.DataFrame`. ([#1110](https://github.com/scikit-bio/scikit-bio/issues/1110))
* Added `inner`, `ilr`, `ilr_inv` and `clr_inv`, ``skbio.stats.composition``, which enables linear transformations on compositions ([#892](https://github.com/scikit-bio/scikit-bio/issues/892)
* Added ``skbio.diversity.alpha.pielou_e`` function as an evenness metric of alpha diversity. ([#1068](https://github.com/scikit-bio/scikit-bio/issues/1068))
* Added `to_regex` method to `skbio.sequence._iupac_sequence` ABC - it returns a regex object that matches all non-degenerate versions of the sequence.
* Added ``skbio.util.assert_ordination_results_equal`` function for comparing ``OrdinationResults`` objects in unit tests.
* Added ``skbio.io.format.genbank`` for reading and writing GenBank/GenPept for ``DNA``, ``RNA``, ``Protein`` and ``Sequence`` classes.
* Added ``skbio.util.RepresentationWarning`` for warning about substitutions, assumptions, or particular alterations that were made for the successful completion of a process.
* ``TreeNode.tip_tip_distances`` now supports nodes without an associated length. In this case, a length of 0.0 is assumed and an ``skbio.util.RepresentationWarning`` is raised. Previous behavior was to raise a ``NoLengthError``. ([#791](https://github.com/scikit-bio/scikit-bio/issues/791))
* ``DistanceMatrix`` now has a new constructor method called `from_iterable`.
* ``Sequence`` now accepts ``lowercase`` keyword like ``DNA`` and others. Updated ``fasta``, ``fastq``, and ``qseq`` readers/writers for ``Sequence`` to reflect this.
* The ``lowercase`` method has been moved up to ``Sequence`` meaning all sequence objects now have a ``lowercase`` method.
* Added ``reverse_transcribe`` class method to ``RNA``.
* Added `Sequence.observed_chars` property for obtaining the set of observed characters in a sequence. ([#1075](https://github.com/scikit-bio/scikit-bio/issues/1075))
* Added `Sequence.frequencies` method for computing character frequencies in a sequence. ([#1074](https://github.com/scikit-bio/scikit-bio/issues/1074))
* Added experimental class-method ``Sequence.concat`` which will produce a new sequence from an iterable of existing sequences. Parameters control how positional metadata is propagated during a concatenation.
* ``TreeNode.to_array`` now supports replacing ``nan`` branch lengths in the resulting branch length vector with the value provided as ``nan_length_value``.
* ``skbio.io.format.phylip`` now supports sniffing and reading strict, sequential PHYLIP-formatted files into ``skbio.Alignment`` objects. ([#1006](https://github.com/scikit-bio/scikit-bio/issues/1006))
* Added `default_gap_char` class property to ``DNA``, ``RNA``, and ``Protein`` for representing gap characters in a new sequence.

### Backward-incompatible changes [stable]
* `Sequence.kmer_frequencies` now returns a `dict`. Previous behavior was to return a `collections.Counter` if `relative=False` was passed, and a `collections.defaultdict` if `relative=True` was passed. In the case of a missing key, the `Counter` would return 0 and the `defaultdict` would return 0.0. Because the return type is now always a `dict`, attempting to access a missing key will raise a `KeyError`. This change *may* break backwards-compatibility depending on how the `Counter`/`defaultdict` is being used. We hope that in most cases this change will not break backwards-compatibility because both `Counter` and `defaultdict` are `dict` subclasses.

   If the previous behavior is desired, convert the `dict` into a `Counter`/`defaultdict`:

    ```python
    import collections
    from skbio import Sequence
    seq = Sequence('ACCGAGTTTAACCGAATA')

    # Counter
    freqs_dict = seq.kmer_frequencies(k=8)
    freqs_counter = collections.Counter(freqs_dict)

    # defaultdict
    freqs_dict = seq.kmer_frequencies(k=8, relative=True)
    freqs_default_dict = collections.defaultdict(float, freqs_dict)
    ```

   **Rationale:** We believe it is safer to return `dict` instead of `Counter`/`defaultdict` as this may prevent error-prone usage of the return value. Previous behavior allowed accessing missing kmers, returning 0 or 0.0 depending on the `relative` parameter. This is convenient in many cases but also potentially misleading. For example, consider the following code:

    ```python
    from skbio import Sequence
    seq = Sequence('ACCGAGTTTAACCGAATA')
    freqs = seq.kmer_frequencies(k=8)
    freqs['ACCGA']
    ```

    Previous behavior would return 0 because the kmer `'ACCGA'` is not present in the `Counter`. In one respect this is the correct answer because we asked for kmers of length 8; `'ACCGA'` is a different length so it is not included in the results. However, we believe it is safer to avoid this implicit behavior in case the user assumes there are no `'ACCGA'` kmers in the sequence (which there are!). A `KeyError` in this case is more explicit and forces the user to consider their query. Returning a `dict` will also be consistent with `Sequence.frequencies`.

### Backward-incompatible changes [experimental]
* Replaced ``PCoA``, ``CCA``, ``CA`` and ``RDA`` in ``skbio.stats.ordination`` with equivalent functions ``pcoa``, ``cca``, ``ca`` and ``rda``. These functions now take ``pd.DataFrame`` objects.
* Change ``OrdinationResults`` to have its attributes based on ``pd.DataFrame`` and ``pd.Series`` objects, instead of pairs of identifiers and values. The changes are as follows:
    - ``species`` and ``species_ids`` have been replaced by a ``pd.DataFrame`` named ``features``.
    - ``site`` and ``site_ids`` have been replaced by a ``pd.DataFrame`` named ``samples``.
    - ``eigvals`` is now a ``pd.Series`` object.
    - ``proportion_explained`` is now a ``pd.Series`` object.
    - ``biplot`` is now a ``pd.DataFrame`` object named ``biplot_scores``.
    - ``site_constraints`` is now a ``pd.DataFrame`` object named ``sample_constraints``.
* ``short_method_name`` and ``long_method_name`` are now required arguments of the ``OrdinationResults`` object.
* Removed `skbio.diversity.alpha.equitability`. Please use `skbio.diversity.alpha.pielou_e`, which is more accurately named and better documented. Note that `equitability` by default used logarithm base 2 while `pielou_e` uses logarithm base `e` as described in Heip 1974.
* ``skbio.diversity.beta.pw_distances`` is now called ``skbio.diversity.beta_diversity``. This function no longer defines a default metric, and ``metric`` is now the first argument to this function. This function can also now take a pairwise distances function as ``pairwise_func``.
* Deprecated function ``skbio.diversity.beta.pw_distances_from_table`` has been removed from scikit-bio as scheduled. Code that used this should be adapted to use ``skbio.diversity.beta_diversity``.
* ``TreeNode.index_tree`` now returns a 2-D numpy array as its second return value (the child node index) instead of a 1-D numpy array.
* Deprecated functions `skbio.draw.boxplots` and `skbio.draw.grouped_distributions` have been removed from scikit-bio as scheduled. These functions generated plots that were not specific to bioinformatics. These types of plots can be generated with seaborn or another general-purpose plotting package.
* Deprecated function `skbio.stats.power.bootstrap_power_curve` has been removed from scikit-bio as scheduled. Use `skbio.stats.power.subsample_power` or `skbio.stats.power.subsample_paired_power` followed by `skbio.stats.power.confidence_bound`.
* Deprecated function `skbio.stats.spatial.procrustes` has been removed from scikit-bio as scheduled in favor of `scipy.spatial.procrustes`.
* Deprecated class `skbio.tree.CompressedTrie` and function `skbio.tree.fasta_to_pairlist` have been removed from scikit-bio as scheduled in favor of existing general-purpose Python trie packages.
* Deprecated function `skbio.util.flatten` has been removed from scikit-bio as scheduled in favor of solutions available in the Python standard library (see [here](http://stackoverflow.com/a/952952/3639023) and [here](http://stackoverflow.com/a/406199/3639023) for examples).
* Pairwise alignment functions in `skbio.alignment` now return a tuple containing the `TabularMSA` alignment, alignment score, and start/end positions. The returned `TabularMSA`'s `index` is always the default integer index; sequence IDs are no longer propagated to the MSA. Additionally, the pairwise alignment functions now accept the following input types to align:
    - `local_pairwise_align_nucleotide`: `DNA` or `RNA`
    - `local_pairwise_align_protein`: `Protein`
    - `local_pairwise_align`: `IUPACSequence`
    - `global_pairwise_align_nucleotide`: `DNA`, `RNA`, or `TabularMSA[DNA|RNA]`
    - `global_pairwise_align_protein`: `Protein` or `TabularMSA[Protein]`
    - `global_pairwise_align`: `IUPACSequence` or `TabularMSA`
    - `local_pairwise_align_ssw`: `DNA`, `RNA`, or `Protein`. Additionally, this function now overrides the `protein` kwarg based on input type. `constructor` parameter was removed because the function now determines the return type based on input type.
* Removed `skbio.alignment.SequenceCollection` in favor of using a list or other standard library containers to store scikit-bio sequence objects (most `SequenceCollection` operations were simple list comprehensions). Use `DistanceMatrix.from_iterable` instead of `SequenceCollection.distances` (pass `key="id"` to exactly match original behavior).
* Removed `skbio.alignment.Alignment` in favor of `skbio.alignment.TabularMSA`.
* Removed `skbio.alignment.SequenceCollectionError` and `skbio.alignment.AlignmentError` exceptions as their corresponding classes no longer exist.

### Bug Fixes

* ``Sequence`` objects now handle slicing of empty positional metadata correctly. Any metadata that is empty will no longer be propagated by the internal ``_to`` constructor. ([#1133](https://github.com/scikit-bio/scikit-bio/issues/1133))
* ``DissimilarityMatrix.plot()`` no longer leaves a white border around the
  heatmap it plots (PR #1070).
* TreeNode.root_at_midpoint`` no longer fails when a node with two equal length child branches exists in the tree. ([#1077](https://github.com/scikit-bio/scikit-bio/issues/1077))
* ``TreeNode._set_max_distance``, as called through ``TreeNode.get_max_distance`` or ``TreeNode.root_at_midpoint`` would store distance information as ``list``s in the attribute ``MaxDistTips`` on each node in the tree, however, these distances were only valid for the node in which the call to ``_set_max_distance`` was made. The values contained in ``MaxDistTips`` are now correct across the tree following a call to ``get_max_distance``. The scope of impact of this bug is limited to users that were interacting directly with ``MaxDistTips`` on descendant nodes; this bug does not impact any known method within scikit-bio. ([#1223](https://github.com/scikit-bio/scikit-bio/issues/1223))
* Added missing `nose` dependency to setup.py's `install_requires`. ([#1214](https://github.com/scikit-bio/scikit-bio/issues/1214))
* Fixed issue that resulted in legends of ``OrdinationResult`` plots sometimes being truncated. ([#1210](https://github.com/scikit-bio/scikit-bio/issues/1210))

### Deprecated functionality [stable]
* `skbio.Sequence.copy` has been deprecated in favor of `copy.copy(seq)` and `copy.deepcopy(seq)`.

### Miscellaneous
* Doctests are now written in Python 3.
* ``make test`` now validates MANIFEST.in using [check-manifest](https://github.com/mgedmin/check-manifest). ([#461](https://github.com/scikit-bio/scikit-bio/issues/461))
* Many new alpha diversity equations added to ``skbio.diversity.alpha`` documentation. ([#321](https://github.com/scikit-bio/scikit-bio/issues/321))
* Order of ``lowercase`` and ``validate`` keywords swapped in ``DNA``, ``RNA``, and ``Protein``.

## Version 0.4.0 (2015-07-08)

Initial beta release. In addition to the changes detailed below, the following
subpackages have been mostly or entirely rewritten and most of their APIs are
substantially different (and improved!):

* `skbio.sequence`
* `skbio.io`

The APIs of these subpackages are now stable, and all others are experimental. See the [API stability docs](https://github.com/scikit-bio/scikit-bio/tree/0.4.0/doc/source/user/api_stability.rst) for more details, including what we mean by *stable* and *experimental* in this context. We recognize that this is a lot of backward-incompatible changes. To avoid these types of changes being a surprise to our users, our public APIs are now decorated to make it clear to developers when an API can be relied upon (stable) and when it may be subject to change (experimental).

### Features
* Added `skbio.stats.composition` for analyzing data made up of proportions
* Added new ``skbio.stats.evolve`` subpackage for evolutionary statistics. Currently contains a single function, ``hommola_cospeciation``, which implements a permutation-based test of correlation between two distance matrices.
* Added support for ``skbio.io.util.open_file`` and ``skbio.io.util.open_files`` to pull files from HTTP and HTTPS URLs. This behavior propagates to the I/O registry.
* FASTA/QUAL (``skbio.io.format.fasta``) and FASTQ (``skbio.io.format.fastq``) readers now allow blank or whitespace-only lines at the beginning of the file, between records, or at the end of the file. A blank or whitespace-only line in any other location will continue to raise an error [#781](https://github.com/scikit-bio/scikit-bio/issues/781).
* scikit-bio now ignores leading and trailing whitespace characters on each line while reading FASTA/QUAL and FASTQ files.
* Added `ratio` parameter to `skbio.stats.power.subsample_power`. This allows the user to calculate power on groups for uneven size (For example, draw twice as many samples from Group B than Group A). If `ratio` is not set, group sizes will remain equal across all groups.
* Power calculations (`skbio.stats.power.subsample_power` and `skbio.stats.power.subsample_paired_power`) can use test functions that return multiple p values, like some multivariate linear regression models. Previously, the power calculations required the test to return a single p value.
* Added ``skbio.util.assert_data_frame_almost_equal`` function for comparing ``pd.DataFrame`` objects in unit tests.

### Performance enhancements
* The speed of quality score decoding has been significantly improved (~2x) when reading `fastq` files.
* The speed of `NucleotideSequence.reverse_complement` has been improved (~6x).

### Bug fixes
* Changed `Sequence.distance` to raise an error any time two sequences are passed of different lengths regardless of the `distance_fn` being passed. [(#514)](https://github.com/scikit-bio/scikit-bio/issues/514)
* Fixed issue with ``TreeNode.extend`` where if given the children of another ``TreeNode`` object (``tree.children``), both trees would be left in an incorrect and unpredictable state. ([#889](https://github.com/scikit-bio/scikit-bio/issues/889))
* Changed the way power was calculated in `subsample_paired_power` to move the subsample selection before the test is performed. This increases the number of Monte Carlo simulations performed during power estimation, and improves the accuracy of the returned estimate. Previous power estimates from `subsample_paired_power` should be disregarded and re-calculated. ([#910](https://github.com/scikit-bio/scikit-bio/issues/910))
* Fixed issue where `randdm` was attempting to create asymmetric distance matrices.This was causing an error to be raised by the `DistanceMatrix` constructor inside of the `randdm` function, so that `randdm` would fail when attempting to create large distance matrices. ([#943](https://github.com/scikit-bio/scikit-bio/issues/943))

### Deprecated functionality
* Deprecated `skbio.util.flatten`. This function will be removed in scikit-bio 0.3.1. Please use standard python library functionality
described here [Making a flat list out of lists of lists](http://stackoverflow.com/a/952952/3639023), [Flattening a shallow list](http://stackoverflow.com/a/406199/3639023) ([#833](https://github.com/scikit-bio/scikit-bio/issues/833))
* Deprecated `skbio.stats.power.bootstrap_power_curve` will be removed in scikit-bio 0.4.1. It is deprecated in favor of using ``subsample_power`` or ``sample_paired_power`` to calculate a power matrix, and then the use of ``confidence_bounds`` to calculate the average and confidence intervals.

### Backward-incompatible changes
* Removed the following deprecated functionality:
    - `skbio.parse` subpackage, including `SequenceIterator`, `FastaIterator`, `FastqIterator`, `load`, `parse_fasta`, `parse_fastq`, `parse_qual`, `write_clustal`, `parse_clustal`, and `FastqParseError`; please use `skbio.io` instead.
    - `skbio.format` subpackage, including `fasta_from_sequence`, `fasta_from_alignment`, and `format_fastq_record`; please use `skbio.io` instead.
    - `skbio.alignment.SequenceCollection.int_map`; please use `SequenceCollection.update_ids` instead.
    - `skbio.alignment.SequenceCollection` methods `to_fasta` and `toFasta`; please use `SequenceCollection.write` instead.
    - `constructor` parameter in `skbio.alignment.Alignment.majority_consensus`; please convert returned biological sequence object manually as desired (e.g., `str(seq)`).
    - `skbio.alignment.Alignment.to_phylip`; please use `Alignment.write` instead.
    - `skbio.sequence.BiologicalSequence.to_fasta`; please use `BiologicalSequence.write` instead.
    - `skbio.tree.TreeNode` methods `from_newick`, `from_file`, and `to_newick`; please use `TreeNode.read` and `TreeNode.write` instead.
    - `skbio.stats.distance.DissimilarityMatrix` methods `from_file` and `to_file`; please use `DissimilarityMatrix.read` and `DissimilarityMatrix.write` instead.
    - `skbio.stats.ordination.OrdinationResults` methods `from_file` and `to_file`; please use `OrdinationResults.read` and `OrdinationResults.write` instead.
    - `skbio.stats.p_value_to_str`; there is no replacement.
    - `skbio.stats.subsample`; please use `skbio.stats.subsample_counts` instead.
    - `skbio.stats.distance.ANOSIM`; please use `skbio.stats.distance.anosim` instead.
    - `skbio.stats.distance.PERMANOVA`; please use `skbio.stats.distance.permanova` instead.
    - `skbio.stats.distance.CategoricalStatsResults`; there is no replacement, please use `skbio.stats.distance.anosim` or `skbio.stats.distance.permanova`, which will return a `pandas.Series` object.
* `skbio.alignment.Alignment.majority_consensus` now returns `BiologicalSequence('')` if the alignment is empty. Previously, `''` was returned.
* `min_observations` was removed from `skbio.stats.power.subsample_power` and `skbio.stats.power.subsample_paired_power`. The minimum number of samples for subsampling depends on the data set and statistical tests. Having a default parameter to set unnecessary limitations on the technique.

### Miscellaneous
* Changed testing procedures
    - Developers should now use `make test`
    - Users can use `python -m skbio.test`
    - Added `skbio.util._testing.TestRunner` (available through `skbio.util.TestRunner`). Used to provide a `test` method for each module init file. This class represents a unified testing path which wraps all `skbio` testing functionality.
    - Autodetect Python version and disable doctests for Python 3.
* `numpy` is no longer required to be installed before installing scikit-bio!
* Upgraded checklist.py to check source files non-conforming to [new header style](http://scikit-bio.org/docs/latest/development/new_module.html). ([#855](https://github.com/scikit-bio/scikit-bio/issues/855))
* Updated to use `natsort` >= 4.0.0.
* The method of subsampling was changed for ``skbio.stats.power.subsample_paired_power``. Rather than drawing a paired sample for the run and then subsampling for each count, the subsample is now drawn for each sample and each run. In test data, this did not significantly alter the power results.
* checklist.py now enforces `__future__` imports in .py files.

## Version 0.2.3 (2015-02-13)

### Features
* Modified ``skbio.stats.distance.pwmantel`` to accept a list of filepaths. This is useful as it allows for a smaller amount of memory consumption as it only loads two matrices at a time as opposed to requiring that all distance matrices are loaded into memory.
* Added ``skbio.util.find_duplicates`` for finding duplicate elements in an iterable.

### Bug fixes
* Fixed floating point precision bugs in ``Alignment.position_frequencies``, ``Alignment.position_entropies``, ``Alignment.omit_gap_positions``, ``Alignment.omit_gap_sequences``, ``BiologicalSequence.k_word_frequencies``, and ``SequenceCollection.k_word_frequencies`` ([#801](https://github.com/scikit-bio/scikit-bio/issues/801)).

### Backward-incompatible changes
* Removed ``feature_types`` attribute from ``BiologicalSequence`` and all subclasses ([#797](https://github.com/scikit-bio/scikit-bio/pull/797)).
* Removed ``find_features`` method from ``BiologicalSequence`` and ``ProteinSequence`` ([#797](https://github.com/scikit-bio/scikit-bio/pull/797)).
* ``BiologicalSequence.k_word_frequencies`` now returns a ``collections.defaultdict`` of type ``float`` instead of type ``int``. This only affects the "default" case, when a key isn't present in the dictionary. Previous behavior would return ``0`` as an ``int``, while the new behavior is to return ``0.0`` as a ``float``. This change also affects the ``defaultdict``s that are returned by ``SequenceCollection.k_word_frequencies``.

### Miscellaneous
* ``DissimilarityMatrix`` and ``DistanceMatrix`` now report duplicate IDs in the ``DissimilarityMatrixError`` message that can be raised during validation.

## Version 0.2.2 (2014-12-04)

### Features
* Added ``plot`` method to ``skbio.stats.distance.DissimilarityMatrix`` for creating basic heatmaps of a dissimilarity/distance matrix (see [#684](https://github.com/scikit-bio/scikit-bio/issues/684)). Also added  ``_repr_png_`` and ``_repr_svg_`` methods for automatic display in the IPython Notebook, with ``png`` and ``svg`` properties for direct access.
* Added `__str__` method to `skbio.stats.ordination.OrdinationResults`.
* Added ``skbio.stats.distance.anosim`` and ``skbio.stats.distance.permanova`` functions, which replace the ``skbio.stats.distance.ANOSIM`` and ``skbio.stats.distance.PERMANOVA`` classes. These new functions provide simpler procedural interfaces to running these statistical methods. They also provide more convenient access to results by returning a ``pandas.Series`` instead of a ``CategoricalStatsResults`` object. These functions have more extensive documentation than their previous versions. If significance tests are suppressed, p-values are returned as ``np.nan`` instead of ``None`` for consistency with other statistical methods in scikit-bio. [#754](https://github.com/scikit-bio/scikit-bio/issues/754)
* Added `skbio.stats.power` for performing empirical power analysis. The module uses existing datasets and iteratively draws samples to estimate the number of samples needed to see a significant difference for a given critical value.
* Added `skbio.stats.isubsample` for subsampling from an unknown number of values. This method supports subsampling from multiple partitions and does not require that all items be stored in memory, requiring approximately `O(N*M)`` space where `N` is the number of partitions and `M` is the maximum subsample size.
* Added ``skbio.stats.subsample_counts``, which replaces ``skbio.stats.subsample``. See deprecation section below for more details ([#770](https://github.com/scikit-bio/scikit-bio/issues/770)).

### Bug fixes
* Fixed issue where SSW wouldn't compile on i686 architectures ([#409](https://github.com/scikit-bio/scikit-bio/issues/409)).

### Deprecated functionality
* Deprecated ``skbio.stats.p_value_to_str``. This function will be removed in scikit-bio 0.3.0. Permutation-based p-values in scikit-bio are calculated as ``(num_extreme + 1) / (num_permutations + 1)``, so it is impossible to obtain a p-value of zero. This function historically existed for correcting the number of digits displayed when obtaining a p-value of zero. Since this is no longer possible, this functionality will be removed.
* Deprecated ``skbio.stats.distance.ANOSIM`` and ``skbio.stats.distance.PERMANOVA`` in favor of ``skbio.stats.distance.anosim`` and ``skbio.stats.distance.permanova``, respectively.
* Deprecated ``skbio.stats.distance.CategoricalStatsResults`` in favor of using ``pandas.Series`` to store statistical method results. ``anosim`` and ``permanova`` return ``pandas.Series`` instead of ``CategoricalStatsResults``.
* Deprecated ``skbio.stats.subsample`` in favor of ``skbio.stats.subsample_counts``, which provides an identical interface; only the function name has changed. ``skbio.stats.subsample`` will be removed in scikit-bio 0.3.0.

### Backward-incompatible changes
* Deprecation warnings are now raised using ``DeprecationWarning`` instead of ``UserWarning`` ([#774](https://github.com/scikit-bio/scikit-bio/issues/774)).

### Miscellaneous
* The ``pandas.DataFrame`` returned by ``skbio.stats.distance.pwmantel`` now stores p-values as floats and does not convert them to strings with a specific number of digits. p-values that were previously stored as "N/A" are now stored as ``np.nan`` for consistency with other statistical methods in scikit-bio. See note in "Deprecated functionality" above regarding ``p_value_to_str`` for details.
* scikit-bio now supports versions of IPython < 2.0.0 ([#767](https://github.com/scikit-bio/scikit-bio/issues/767)).

## Version 0.2.1 (2014-10-27)

This is an alpha release of scikit-bio. At this stage, major backwards-incompatible API changes can and will happen. Unified I/O with the scikit-bio I/O registry was the focus of this release.

### Features
* Added ``strict`` and ``lookup`` optional parameters to ``skbio.stats.distance.mantel`` for handling reordering and matching of IDs when provided ``DistanceMatrix`` instances as input (these parameters were previously only available in ``skbio.stats.distance.pwmantel``).
* ``skbio.stats.distance.pwmantel`` now accepts an iterable of ``array_like`` objects. Previously, only ``DistanceMatrix`` instances were allowed.
* Added ``plot`` method to ``skbio.stats.ordination.OrdinationResults`` for creating basic 3-D matplotlib scatterplots of ordination results, optionally colored by metadata in a ``pandas.DataFrame`` (see [#518](https://github.com/scikit-bio/scikit-bio/issues/518)). Also added  ``_repr_png_`` and ``_repr_svg_`` methods for automatic display in the IPython Notebook, with ``png`` and ``svg`` properties for direct access.
* Added ``skbio.stats.ordination.assert_ordination_results_equal`` for comparing ``OrdinationResults`` objects for equality in unit tests.
* ``BiologicalSequence`` (and its subclasses) now optionally store Phred quality scores. A biological sequence's quality scores are stored as a 1-D ``numpy.ndarray`` of nonnegative integers that is the same length as the biological sequence. Quality scores can be provided upon object instantiation via the keyword argument ``quality``, and can be retrieved via the ``BiologicalSequence.quality`` property. ``BiologicalSequence.has_quality`` is also provided for determining whether a biological sequence has quality scores or not. See [#616](https://github.com/scikit-bio/scikit-bio/issues/616) for more details.
* Added ``BiologicalSequence.sequence`` property for retrieving the underlying string representing the sequence characters. This was previously (and still is) accessible via ``BiologicalSequence.__str__``. It is provided via a property for convenience and explicitness.
* Added ``BiologicalSequence.equals`` for full control over equality testing of biological sequences. By default, biological sequences must have the same type, underlying sequence of characters, identifier, description, and quality scores to compare equal. These properties can be ignored via the keyword argument ``ignore``. The behavior of ``BiologicalSequence.__eq__``/``__ne__`` remains unchanged (only type and underlying sequence of characters are compared).
* Added ``BiologicalSequence.copy`` for creating a copy of a biological sequence, optionally with one or more attributes updated.
* ``BiologicalSequence.__getitem__`` now supports specifying a sequence of indices to take from the biological sequence.
* Methods to read and write taxonomies are now available under ``skbio.tree.TreeNode.from_taxonomy`` and ``skbio.tree.TreeNode.to_taxonomy`` respectively.
* Added ``SequenceCollection.update_ids``, which provides a flexible way of updating sequence IDs on a ``SequenceCollection`` or ``Alignment`` (note that a new object is returned, since instances of these classes are immutable). Deprecated ``SequenceCollection.int_map`` in favor of this new method; it will be removed in scikit-bio 0.3.0.
* Added ``skbio.util.cardinal_to_ordinal`` for converting a cardinal number to ordinal string (e.g., useful for error messages).
* New I/O Registry: supports multiple file formats, automatic file format detection when reading, unified procedural ``skbio.io.read`` and ``skbio.io.write`` in addition to OOP interfaces (``read/write`` methods) on the below objects. See ``skbio.io`` for more details.
    - Added "clustal" format support:
        * Has sniffer
        * Readers: ``Alignment``
        * Writers: ``Alignment``
    - Added "lsmat" format support:
        * Has sniffer
        * Readers: ``DissimilarityMatrix``, ``DistanceMatrix``
        * Writers: ``DissimilarityMatrix``, ``DistanceMatrix``
    - Added "ordination" format support:
        * Has sniffer
        * Readers: ``OrdinationResults``
        * Writers: ``OrdinationResults``
    - Added "newick" format support:
        * Has sniffer
        * Readers: ``TreeNode``
        * Writers: ``TreeNode``
    - Added "phylip" format support:
        * No sniffer
        * Readers: None
        * Writers: ``Alignment``
    - Added "qseq" format support:
        * Has sniffer
        * Readers: generator of ``BiologicalSequence`` or its subclasses, ``SequenceCollection``, ``BiologicalSequence``, ``NucleotideSequence``, ``DNASequence``, ``RNASequence``, ``ProteinSequence``
        * Writers: None
    - Added "fasta"/QUAL format support:
        * Has sniffer
        * Readers: generator of ``BiologicalSequence`` or its subclasses, ``SequenceCollection``, ``Alignment``, ``BiologicalSequence``, ``NucleotideSequence``, ``DNASequence``, ``RNASequence``, ``ProteinSequence``
        * Writers: same as readers
    - Added "fastq" format support:
        * Has sniffer
        * Readers: generator of ``BiologicalSequence`` or its subclasses, ``SequenceCollection``, ``Alignment``, ``BiologicalSequence``, ``NucleotideSequence``, ``DNASequence``, ``RNASequence``, ``ProteinSequence``
        * Writers: same as readers

### Bug fixes

* Removed ``constructor`` parameter from ``Alignment.k_word_frequencies``, ``BiologicalSequence.k_words``, ``BiologicalSequence.k_word_counts``, and ``BiologicalSequence.k_word_frequencies`` as it had no effect (it was never hooked up in the underlying code). ``BiologicalSequence.k_words`` now returns a generator of ``BiologicalSequence`` objects instead of strings.
* Modified the ``Alignment`` constructor to verify that all sequences have the same length, if not, raise an ``AlignmentError`` exception.  Updated the method ``Alignment.subalignment`` to calculate the indices only once now that identical sequence length is guaranteed.

### Deprecated functionality
* Deprecated ``constructor`` parameter in ``Alignment.majority_consensus`` in favor of having users call ``str`` on the returned ``BiologicalSequence``. This parameter will be removed in scikit-bio 0.3.0.

* Existing I/O functionality deprecated in favor of I/O registry, old functionality will be removed in scikit-bio 0.3.0. All functionality can be found at ``skbio.io.read``, ``skbio.io.write``, and the methods listed below:
    * Deprecated the following "clustal" readers/writers:
        - ``write_clustal`` -> ``Alignment.write``
        - ``parse_clustal`` -> ``Alignment.read``

    * Deprecated the following distance matrix format ("lsmat") readers/writers:
        - ``DissimilarityMatrix.from_file`` -> ``DissimilarityMatrix.read``
        - ``DissimilarityMatrix.to_file`` -> ``DissimilarityMatrix.write``
        - ``DistanceMatrix.from_file`` -> ``DistanceMatrix.read``
        - ``DistanceMatrix.to_file`` -> ``DistanceMatrix.write``

    * Deprecated the following ordination format ("ordination") readers/writers:
        - ``OrdinationResults.from_file`` -> ``OrdinationResults.read``
        - ``OrdinationResults.to_file`` -> ``OrdinationResults.write``

    * Deprecated the following "newick" readers/writers:
        - ``TreeNode.from_file`` -> ``TreeNode.read``
        - ``TreeNode.from_newick`` -> ``TreeNode.read``
        - ``TreeNode.to_newick`` -> ``TreeNode.write``

    * Deprecated the following "phylip" writers:
        - ``Alignment.to_phylip`` -> ``Alignment.write``

    * Deprecated the following "fasta"/QUAL readers/writers:
        - ``SequenceCollection.from_fasta_records`` -> ``SequenceCollection.read``
        - ``SequenceCollection.to_fasta`` -> ``SequenceCollection.write``
        - ``fasta_from_sequences`` -> ``skbio.io.write(obj, into=<file>, format='fasta')``
        - ``fasta_from_alignment`` -> ``Alignment.write``
        - ``parse_fasta`` -> ``skbio.io.read(<fasta>, format='fasta')``
        - ``parse_qual`` -> ``skbio.io.read(<fasta>, format='fasta', qual=<file>)``
        - ``BiologicalSequence.to_fasta`` -> ``BiologicalSequence.write``

    * Deprecated the following "fastq" readers/writers:
        - ``parse_fastq`` -> ``skbio.io.read(<fastq>, format='fastq')``
        - ``format_fastq_record`` -> ``skbio.io.write(<fastq>, format='fastq')``

### Backward-incompatible changes

* ``skbio.stats.distance.mantel`` now returns a 3-element tuple containing correlation coefficient, p-value, and the number of matching rows/cols in the distance matrices (``n``). The return value was previously a 2-element tuple containing only the correlation coefficient and p-value.
* ``skbio.stats.distance.mantel`` reorders input ``DistanceMatrix`` instances based on matching IDs (see optional parameters ``strict`` and ``lookup`` for controlling this behavior). In the past, ``DistanceMatrix`` instances were treated the same as ``array_like`` input and no reordering took place, regardless of ID (mis)matches. ``array_like`` input behavior remains the same.
* If mismatched types are provided to ``skbio.stats.distance.mantel`` (e.g., a ``DistanceMatrix`` and ``array_like``), a ``TypeError`` will be raised.

### Miscellaneous

* Added git timestamp checking to checklist.py, ensuring that when changes are made to Cython (.pyx) files, their corresponding generated C files are also updated.
* Fixed performance bug when instantiating ``BiologicalSequence`` objects. The previous runtime scaled linearly with sequence length; it is now constant time when the sequence is already a string. See [#623](https://github.com/scikit-bio/scikit-bio/issues/623) for details.
* IPython and six are now required dependencies.

## Version 0.2.0 (2014-08-07)

This is an initial alpha release of scikit-bio. At this stage, major backwards-incompatible API changes can and will happen. Many backwards-incompatible API changes were made since the previous release.

### Features

* Added ability to compute distances between sequences in a ``SequenceCollection`` object ([#509](https://github.com/scikit-bio/scikit-bio/issues/509)), and expanded ``Alignment.distance`` to allow the user to pass a function for computing distances (the default distance metric is still ``scipy.spatial.distance.hamming``) ([#194](https://github.com/scikit-bio/scikit-bio/issues/194)).
* Added functionality to not penalize terminal gaps in global alignment. This functionality results in more biologically relevant global alignments (see [#537](https://github.com/scikit-bio/scikit-bio/issues/537) for discussion of the issue) and is now the default behavior for global alignment.
* The python global aligners (``global_pairwise_align``, ``global_pairwise_align_nucleotide``, and ``global_pairwise_align_protein``) now support aligning pairs of sequences, pairs of alignments, and a sequence and an alignment (see [#550](https://github.com/scikit-bio/scikit-bio/issues/550)). This functionality supports progressive multiple sequence alignment, among other things such as adding a sequence to an existing alignment.
* Added ``StockholmAlignment.to_file`` for writing Stockholm-formatted files.
* Added ``strict=True`` optional parameter to ``DissimilarityMatrix.filter``.
* Added ``TreeNode.find_all`` for finding all tree nodes that match a given name.


### Bug fixes

* Fixed bug that resulted in a ``ValueError`` from ``local_align_pairwise_nucleotide`` (see [#504](https://github.com/scikit-bio/scikit-bio/issues/504)) under many circumstances. This would not generate incorrect results, but would cause the code to fail.

### Backward-incompatible changes

* Removed ``skbio.math``, leaving ``stats`` and ``diversity`` to become top level packages. For example, instead of ``from skbio.math.stats.ordination import PCoA`` you would now import ``from skbio.stats.ordination import PCoA``.
* The module ``skbio.math.gradient`` as well as the contents of ``skbio.math.subsample`` and ``skbio.math.stats.misc`` are now found in ``skbio.stats``. As an example, to import subsample: ``from skbio.stats import subsample``; to import everything from gradient: ``from skbio.stats.gradient import *``.
* The contents of ``skbio.math.stats.ordination.utils`` are now in ``skbio.stats.ordination``.
* Removed ``skbio.app`` subpackage (i.e., the *application controller framework*) as this code has been ported to the standalone [burrito](https://github.com/biocore/burrito) Python package. This code was not specific to bioinformatics and is useful for wrapping command-line applications in general.
* Removed ``skbio.core``, leaving ``alignment``, ``genetic_code``, ``sequence``, ``tree``, and ``workflow`` to become top level packages. For example, instead of ``from skbio.core.sequence import DNA`` you would now import ``from skbio.sequence import DNA``.
* Removed ``skbio.util.exception`` and ``skbio.util.warning`` (see [#577](https://github.com/scikit-bio/scikit-bio/issues/577) for the reasoning behind this change). The exceptions/warnings were moved to the following locations:
 - ``FileFormatError``, ``RecordError``, ``FieldError``, and ``EfficiencyWarning`` have been moved to ``skbio.util``
 - ``BiologicalSequenceError`` has been moved to ``skbio.sequence``
 - ``SequenceCollectionError`` and ``StockholmParseError`` have been moved to ``skbio.alignment``
 - ``DissimilarityMatrixError``, ``DistanceMatrixError``, ``DissimilarityMatrixFormatError``, and ``MissingIDError`` have been moved to ``skbio.stats.distance``
 - ``TreeError``, ``NoLengthError``, ``DuplicateNodeError``, ``MissingNodeError``, and ``NoParentError`` have been moved to ``skbio.tree``
 - ``FastqParseError`` has been moved to ``skbio.parse.sequences``
 - ``GeneticCodeError``, ``GeneticCodeInitError``, and ``InvalidCodonError`` have been moved to ``skbio.genetic_code``
* The contents of ``skbio.genetic_code`` formerly ``skbio.core.genetic_code`` are now in ``skbio.sequence``. The ``GeneticCodes`` dictionary is now a function ``genetic_code``. The functionality is the same, except that because this is now a function rather than a dict, retrieving a genetic code is done using a function call rather than a lookup (so, for example, ``GeneticCodes[2]`` becomes ``genetic_code(2)``.
* Many submodules have been made private with the intention of simplifying imports for users. See [#562](https://github.com/scikit-bio/scikit-bio/issues/562) for discussion of this change. The following list contains the previous module name and where imports from that module should now come from.
 - ``skbio.alignment.ssw`` to ``skbio.alignment``
 - ``skbio.alignment.alignment`` to ``skbio.alignment``
 - ``skbio.alignment.pairwise`` to ``skbio.alignment``
 - ``skbio.diversity.alpha.base`` to ``skbio.diversity.alpha``
 - ``skbio.diversity.alpha.gini`` to ``skbio.diversity.alpha``
 - ``skbio.diversity.alpha.lladser`` to ``skbio.diversity.alpha``
 - ``skbio.diversity.beta.base`` to ``skbio.diversity.beta``
 - ``skbio.draw.distributions`` to ``skbio.draw``
 - ``skbio.stats.distance.anosim`` to ``skbio.stats.distance``
 - ``skbio.stats.distance.base`` to ``skbio.stats.distance``
 - ``skbio.stats.distance.permanova`` to ``skbio.stats.distance``
 - ``skbio.distance`` to ``skbio.stats.distance``
 - ``skbio.stats.ordination.base`` to ``skbio.stats.ordination``
 - ``skbio.stats.ordination.canonical_correspondence_analysis`` to ``skbio.stats.ordination``
 - ``skbio.stats.ordination.correspondence_analysis`` to ``skbio.stats.ordination``
 - ``skbio.stats.ordination.principal_coordinate_analysis`` to ``skbio.stats.ordination``
 - ``skbio.stats.ordination.redundancy_analysis`` to ``skbio.stats.ordination``
 - ``skbio.tree.tree`` to ``skbio.tree``
 - ``skbio.tree.trie`` to ``skbio.tree``
 - ``skbio.util.misc`` to ``skbio.util``
 - ``skbio.util.testing`` to ``skbio.util``
 - ``skbio.util.exception`` to ``skbio.util``
 - ``skbio.util.warning`` to ``skbio.util``
* Moved ``skbio.distance`` contents into ``skbio.stats.distance``.

### Miscellaneous

* Relaxed requirement in ``BiologicalSequence.distance`` that sequences being compared are of equal length. This is relevant for Hamming distance, so the check is still performed in that case, but other distance metrics may not have that requirement. See [#504](https://github.com/scikit-bio/scikit-bio/issues/507)).
* Renamed ``powertrip.py`` repo-checking script to ``checklist.py`` for clarity.
* ``checklist.py`` now ensures that all unit tests import from a minimally deep API. For example, it will produce an error if ``skbio.core.distance.DistanceMatrix`` is used over ``skbio.DistanceMatrix``.
* Extra dimension is no longer calculated in ``skbio.stats.spatial.procrustes``.
* Expanded documentation in various subpackages.
* Added new scikit-bio logo. Thanks [Alina Prassas](http://cargocollective.com/alinaprassas)!

## Version 0.1.4 (2014-06-25)

This is a pre-alpha release. At this stage, major backwards-incompatible API changes can and will happen.

### Features

* Added Python implementations of Smith-Waterman and Needleman-Wunsch alignment as ``skbio.core.alignment.pairwise.local_pairwise_align`` and ``skbio.core.alignment.pairwise.global_pairwise_align``. These are much slower than native C implementations (e.g., ``skbio.core.alignment.local_pairwise_align_ssw``) and as a result raise an ``EfficencyWarning`` when called, but are included as they serve as useful educational examples as they’re simple to experiment with.
* Added ``skbio.core.diversity.beta.pw_distances`` and ``skbio.core.diversity.beta.pw_distances_from_table``. These provide convenient access to the ``scipy.spatial.distance.pdist`` *beta diversity* metrics from within scikit-bio. The ``skbio.core.diversity.beta.pw_distances_from_table`` function will only be available temporarily, until the ``biom.table.Table`` object is merged into scikit-bio (see [#489](https://github.com/scikit-bio/scikit-bio/issues/489)), at which point ``skbio.core.diversity.beta.pw_distances`` will be updated to use that.
* Added ``skbio.core.alignment.StockholmAlignment``, which provides support for parsing [Stockholm-formatted alignment files](http://sonnhammer.sbc.su.se/Stockholm.html) and working with those alignments in the context RNA secondary structural information.
* Added ``skbio.core.tree.majority_rule`` function for computing consensus trees from a list of trees.

### Backward-incompatible changes

* Function ``skbio.core.alignment.align_striped_smith_waterman`` renamed to ``local_pairwise_align_ssw`` and now returns an ``Alignment`` object instead of an ``AlignmentStructure``
* The following keyword-arguments for ``StripedSmithWaterman`` and ``local_pairwise_align_ssw`` have been renamed:
    * ``gap_open`` -> ``gap_open_penalty``
    * ``gap_extend`` -> ``gap_extend_penalty``
    * ``match`` -> ``match_score``
    * ``mismatch`` -> ``mismatch_score``
* Removed ``skbio.util.sort`` module in favor of [natsort](https://pypi.python.org/pypi/natsort) package.

### Miscellaneous

* Added powertrip.py script to perform basic sanity-checking of the repo based on recurring issues that weren't being caught until release time; added to Travis build.
* Added RELEASE.md with release instructions.
* Added intersphinx mappings to docs so that "See Also" references to numpy, scipy, matplotlib, and pandas are hyperlinks.
* The following classes are no longer ``namedtuple`` subclasses (see [#359](https://github.com/scikit-bio/scikit-bio/issues/359) for the rationale):
    * ``skbio.math.stats.ordination.OrdinationResults``
    * ``skbio.math.gradient.GroupResults``
    * ``skbio.math.gradient.CategoryResults``
    * ``skbio.math.gradient.GradientANOVAResults``
* Added coding guidelines draft.
* Added new alpha diversity formulas to the ``skbio.math.diversity.alpha`` documentation.

## Version 0.1.3 (2014-06-12)

This is a pre-alpha release. At this stage, major backwards-incompatible API changes can and will happen.

### Features

* Added ``enforce_qual_range`` parameter to ``parse_fastq`` (on by default, maintaining backward compatibility). This allows disabling of the quality score range-checking.
* Added ``skbio.core.tree.nj``, which applies neighbor-joining for phylogenetic reconstruction.
* Added ``bioenv``, ``mantel``, and ``pwmantel`` distance-based statistics to ``skbio.math.stats.distance`` subpackage.
* Added ``skbio.math.stats.misc`` module for miscellaneous stats utility functions.
* IDs are now optional when constructing a ``DissimilarityMatrix`` or ``DistanceMatrix`` (monotonically-increasing integers cast as strings are automatically used).
* Added ``DistanceMatrix.permute`` method for randomly permuting rows and columns of a distance matrix.
* Added the following methods to ``DissimilarityMatrix``: ``filter``, ``index``, and ``__contains__`` for ID-based filtering, index lookup, and membership testing, respectively.
* Added ``ignore_comment`` parameter to ``parse_fasta`` (off by default, maintaining backward compatibility). This handles stripping the comment field from the header line (i.e., all characters beginning with the first space) before returning the label.
* Added imports of ``BiologicalSequence``, ``NucleotideSequence``, ``DNA``, ``DNASequence``, ``RNA``, ``RNASequence``, ``Protein``, ``ProteinSequence``, ``DistanceMatrix``, ``align_striped_smith_waterman``, `` SequenceCollection``, ``Alignment``, ``TreeNode``, ``nj``, ``parse_fasta``, ``parse_fastq``, ``parse_qual``, ``FastaIterator``, ``FastqIterator``, ``SequenceIterator`` in ``skbio/__init__.py`` for convenient importing. For example, it's now possible to ``from skbio import Alignment``, rather than ``from skbio.core.alignment import Alignment``.

### Bug fixes

* Fixed a couple of unit tests that could fail stochastically.
* Added missing ``__init__.py`` files to a couple of test directories so that these tests won't be skipped.
* ``parse_fastq`` now raises an error on dangling records.
* Fixed several warnings that were raised while running the test suite with Python 3.4.

### Backward-incompatible changes

* Functionality imported from ``skbio.core.ssw`` must now be imported from ``skbio.core.alignment`` instead.

### Miscellaneous

* Code is now flake8-compliant; added flake8 checking to Travis build.
* Various additions and improvements to documentation (API, installation instructions, developer instructions, etc.).
* ``__future__`` imports are now standardized across the codebase.
* New website front page and styling changes throughout. Moved docs site to its own versioned subdirectories.
* Reorganized alignment data structures and algorithms (e.g., SSW code, ``Alignment`` class, etc.) into an ``skbio.core.alignment`` subpackage.

## Version 0.1.1 (2014-05-16)

Fixes to setup.py. This is a pre-alpha release. At this stage, major backwards-incompatible API changes can and will happen.

## Version 0.1.0 (2014-05-15)

Initial pre-alpha release. At this stage, major backwards-incompatible API changes can and will happen.
