from .kmeans import cluster_kmeans
from .random_projection import cluster_random_projection
from .pq_subspace import cluster_pq_subspace
from .kcenter import cluster_kcenter
from .pca_pq import cluster_pca_pq
from .whitened_pq import cluster_whitened_pq
from .batch_nn import cluster_batch_nn
from .fast_balanced_nn import cluster_fast_balanced_nn
from .l1_nn_pairing import cluster_l1_nn
from .l1_batch_nn import cluster_l1_batch_nn
from .linf_nn_pairing import cluster_linf_nn, cluster_weighted_l1_nn
from .balanced_kcenter import cluster_balanced_kcenter
from .fast_nn_pairing import cluster_block_nn
from .min_weight_matching import cluster_min_weight_matching

CLUSTERING_METHODS = {
    "kmeans": cluster_kmeans,
    "random_proj": cluster_random_projection,
    "pq_subspace": cluster_pq_subspace,
    "kcenter": cluster_kcenter,
    "pca_pq": cluster_pca_pq,
    "whitened_pq": cluster_whitened_pq,
    "batch_nn": cluster_batch_nn,
    "fast_balanced_nn": cluster_fast_balanced_nn,
    "fast_nn_pairing": cluster_block_nn,
    "l1_nn": cluster_l1_nn,
    "l1_batch_nn": cluster_l1_batch_nn,
    "linf_nn": cluster_linf_nn,
    "weighted_l1_nn": cluster_weighted_l1_nn,
    "balanced_kcenter": cluster_balanced_kcenter,
    "min_weight_matching": cluster_min_weight_matching,
}
