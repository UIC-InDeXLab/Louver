// attention v4.3 — v4 bitmask full-AND kernel with parallel mask construction.
#define HIRA_V4_ATTEND_FN attend_v4_3
#define HIRA_V4_PARENTS_PER_TILE 128
#define HIRA_V4_PARALLEL_PASS_BLOCKS
#include "_attention_v4_bitmask_common.h"
