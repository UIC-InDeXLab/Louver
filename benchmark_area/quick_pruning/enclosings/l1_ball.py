"""L1-ball enclosure centered at the cluster centroid."""

from __future__ import annotations

from .lp_ball import enclose_lp_ball


def enclose_l1_ball(keys, assign, centers, K, bf):
    return enclose_lp_ball(keys, assign, centers, K, bf, p=1.0)
