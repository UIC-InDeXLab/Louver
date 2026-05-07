# Methods in `benchmark_area/quick_pruning`

This note tracks all concrete clustering and enclosing implementations currently present under `clusterings/` and `enclosings/`.

Status tags:
- `default`: exposed through the main registries in `clusterings/__init__.py` or `enclosings/__init__.py`
- `implemented`: code exists in the tree, but it is not enabled in the main default registry

Helper-only modules such as `clusterings/_balanced_utils.py` and `clusterings/_lp_utils.py` are intentionally omitted.

## Clustering Methods

### General-purpose centroid and distortion methods

- `kmeans` (`default`): standard Lloyd k-means in the original key space.
- `kmeans_pp` (`default`): standard k-means with k-means++ seeding.
- `spherical_kmeans` (`implemented`): k-means on L2-normalized keys, targeting cosine similarity.
- `gmm_diag` (`implemented`): diagonal-covariance Gaussian mixture model with hard-EM updates.
- `random_partition` (`default`): assigns points uniformly at random; useful as a structure-free baseline.
- `random_proj` (`default`): LSH-style random-projection hashing followed by center refinement.
- `pca_kmeans` (`implemented`): runs k-means in a low-rank global PCA subspace, then rebuilds original-space centroids.

### K-center and ball-oriented methods

- `kcenter` (`default`): greedy farthest-point seeding under L2 followed by centroid refinement.
- `kcenter_lp` (`default`): `L_p`-aware k-center with `L_p` reassignment and simple `L_p`-aware recentering.
- `kcenter_meb` (`default`): uses `kcenter` assignments, then replaces means with approximate minimum-enclosing-ball centers.
- `kcenter_minimax` (`default`): k-center with repeated minimax-style 1-center recentering instead of Lloyd mean updates.
- `ball_ratio_kmeans` (`default`): optimizes a ball-radius-versus-center-norm objective intended for ball gates.
- `ball_ratio_kcenter` (`default`): k-center-style variant of the same radius-to-center-ratio objective.
- `ray_kmeans` (`default`): clusters on ray features mixing direction and log-norm.
- `ray_kcenter` (`default`): farthest-point clustering on the same ray features.
- `ray_kcenter_meb` (`default`): ray-based k-center assignments with per-cluster MEB-style centers.

### Direction and norm-aware methods

- `direction_kmeans` (`default`): clusters on L2-normalized directions, then recomputes original-space centroids.
- `shell_kmeans` (`default`): partitions by norm shells, then clusters directions within each shell.
- `dirnorm_pq` (`default`): PQ-style clustering on direction features augmented with a scaled norm feature.

### AABB- and span-oriented methods

- `linf_kmeans` (`default`): Chebyshev-distance k-means with midrange centers, directly targeting worst-axis spread.
- `span_kmeans` (`default`): assigns keys to minimize AABB span extension instead of L2 distortion.
- `pq_linf` (`default`): PQ initialization followed by `L_inf`-oriented refinement.
- `pq_l2` (`default`): PQ initialization followed by ordinary L2 refinement.
- `pq_span` (`default`): PQ initialization followed by span-aware refinement.
- `pq_span_refine` (`default`): PQ initialization followed by weighted box-extension reassignment.
- `pq_balanced_span` (`default`): balanced-capacity version of the PQ span refinement.
- `pca_axis_chunk` (`default`): exact-size contiguous chunking along the first PCA axis, followed by span refinement.
- `pca_morton_span` (`default`): PCA projection, Morton ordering, then balanced span refinement.

### Product-quantization and whitening families

- `pq_subspace` (`default`): product-quantization-inspired clustering by splitting dimensions into subspaces.
- `pca_pq` (`default`): full PCA rotation followed by PQ-style subspace clustering.
- `whitened_pq` (`default`): PQ-style clustering in globally variance-whitened coordinates.
- `whitened_pq_kpp` (`implemented`): whitened PQ with k-means++ initialization in each subspace.
- `whitened_pq_span` (`implemented`): whitened PQ initialization followed by span-aware refinement.

### Balanced and exact-capacity partitioners

- `balanced_kmeans` (`default`): capacity-constrained Lloyd updates with exact near-`bf` cluster sizes.
- `balanced_kcenter` (`default`): farthest-point seeding followed by balanced exact-capacity reassignment.
- `balanced_pca_tree` (`default`): recursively splits along dominant local axes while matching final leaf capacities.
- `balanced_ray_kmeans` (`default`): balanced clustering on ray features.
- `balanced_ray_kcenter` (`default`): balanced farthest-point clustering on ray features.
- `pca_morton_chunk` (`default`): PCA projection, Morton ordering, and exact-size contiguous chunking.
- `pca_tree` (`implemented`): balanced PCA-tree partitioning in a low-rank global PCA basis.
- `pca_bisect` (`implemented`): recursive PCA bisection with balanced splits.

### Nearest-neighbor grouping methods

- `batch_nn` (`default`): greedy grouping from L2 nearest-neighbor candidate balls.
- `batch_nn_l1` (`default`): nearest-neighbor grouping scored with L1 distance.
- `batch_nn_linf` (`default`): nearest-neighbor grouping scored with `L_inf` distance.
- `batch_nn_aabb_aware` (`default`): weighted-L1 nearest-neighbor grouping that approximates AABB slack.
- `batch_nn_lp` (`default`): generic `L_p` nearest-neighbor grouping.

## Enclosing Methods

### Ball-family bounds

- `ball_centroid` (`default`): centroid-centered L2 ball with radius equal to the farthest assigned point.
- `l1_ball` (`default`): centroid-centered `L_1` ball using the exact dual-norm support bound.
- `lp_ball` (`implemented`): generic centroid-centered `L_p` ball with dual-norm gate; used directly by `comparison_lp_ball.py`.
- `min_enclosing_ball` (`default`): approximate minimum-enclosing-ball centering via iterative farthest-point shifts.
- `span_ball` (`default`): ball centered at the AABB midrange with radius from the AABB half-span vector.
- `outlier_ball_centroid` (`default`): removes one farthest point, fits a centroid ball to the core, and checks the outlier directly.
- `outlier_span_ball` (`default`): removes one farthest point, fits a span-ball to the core, and checks the outlier directly.
- `multi_ball` (`implemented`): covers each cluster with a small union of balls and uses the best anchor.

### Axis-aligned and box-family bounds

- `aabb` (`default`): per-cluster axis-aligned bounding box with max-corner support evaluation.
- `outlier_aabb` (`default`): removes one outlier per cluster, builds an AABB on the core, and checks the outlier directly.
- `split_aabb` (`implemented`): splits each cluster into two axis-based halves and uses one AABB per half.
- `bisect_aabb` (`implemented`): performs a small 2-means-like split, builds one AABB per half, and intersects with a ball bound.
- `quad_aabb` (`implemented`): recursively bisects each cluster twice to build four sub-boxes.
- `pca_obb` (`default`): rotates into a global PCA basis and builds an oriented bounding box there.
- `topk_aabb_residual` (`default`): exact AABB on a small set of salient dimensions plus a residual L2 bound.
- `global_pca_box` (`implemented`): global PCA subspace box plus orthogonal residual ball.
- `subspace_box` (`implemented`): per-cluster low-rank oriented box plus orthogonal residual.

### Directional and low-rank interval bounds

- `cone` (`default`): angular cone bound using mean direction, half-angle, and max norm.
- `centerline` (`default`): rank-1 cluster-local directional bound with an orthogonal residual ball.
- `dual_centerline` (`implemented`): rank-2 extension of `centerline` with a second cheap residual axis.
- `axis_interval` (`default`): one local axis interval plus orthogonal residual radius.
- `dual_axis_interval` (`default`): two local axis intervals plus residual radius.

### Ellipsoidal, slab, and hybrid intersections

- `ellipsoid` (`default`): diagonal Mahalanobis-style ellipsoid with a ball fallback.
- `slab_bundle` (`default`): several random-direction interval slabs intersected together, plus a ball safeguard.
- `hybrid` (`implemented`): intersection of ball, AABB, and cone bounds.
- `hybrid_plus` (`implemented`): 5-way intersection of ball, AABB, cone, ellipsoid, and centerline.
- `tight_hybrid` (`implemented`): minimum of several complementary support bounds, including ball, AABB, subspace box, and multi-ball ideas.
