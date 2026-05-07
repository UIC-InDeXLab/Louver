from .ball_centroid import enclose_ball_centroid
from .min_enclosing_ball import enclose_min_ball
from .aabb import enclose_aabb
from .ellipsoid import enclose_ellipsoid
from .outlier_aabb import enclose_outlier_aabb
from .outlier_ball_centroid import enclose_outlier_ball_centroid

from .span_ball import enclose_span_ball
from .fp16_aabb import enclose_fp16_aabb
from .partial_aabb import (
    enclose_partial_aabb_d8,
    enclose_partial_aabb_d16,
    enclose_partial_aabb_d32,
    enclose_partial_aabb_d64,
)

ENCLOSING_METHODS = {
    "ball_centroid": enclose_ball_centroid,
    "min_enclosing_ball": enclose_min_ball,
    "aabb": enclose_aabb,
    "ellipsoid": enclose_ellipsoid,
    "outlier_aabb": enclose_outlier_aabb,
    "outlier_ball_centroid": enclose_outlier_ball_centroid,
    "span_ball": enclose_span_ball,
    "partial_aabb_d8": enclose_partial_aabb_d8,
    "partial_aabb_d16": enclose_partial_aabb_d16,
    "partial_aabb_d32": enclose_partial_aabb_d32,
    "partial_aabb_d64": enclose_partial_aabb_d64,
    "fp16_aabb": enclose_fp16_aabb,
}
