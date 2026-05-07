### Main
Here is the problem: I want to get good pruning of the points (keys) in this space by first grouping them, then enclosing each group. in the query time I receive a halfspace, pair of (query, threshold), and I want to filter out those enclosings that have no intersection.

The clustering (or aka groupings) give me groups. Each enclosing has a gate cost G. This cost should not be larger than bf, because on average I want the intersection-checking (gating) of all clusters (groups) to be cheaper than full dot product calculation of all the points, as the baseline bruteforce setting. this means that I want to achieve speedup, asymptotically.

Currently, there are different clustering/enclosing methods implemented. But their problem is that, for those cheap enclosings (cheap gating cost), the pruning power is small and for those enclosings with good pruning powers, the gate cost is large, or according to this gate cost I need to go for larger bf, that results in bad pruning.

I need to achieve good speed up using the above constraints. Here are the high level directions:
- having a good clustering (grouping) algorithm that gets good pruning for enclosings that are not cheap in gating. In other words, for large bf, I need to get good prunings on those enclosings. (remember bf should be larger than g to make sense).
- having good enclosing method that is cheap in gating, or even reimplementing current enclosings to achieve a faster gating.
- Adaptively using the information about the queries, or the queries observed so far to restructure the index (clustering or enclosings). We are observing something by queries. Can we use this observation to our advantage?
- Having a good grouping in terms of disjoint groups while within each group the points are close.

First think about these ideas and write your ideas in a new .md file. If you need, you can try implementing them and checking if they work, and continue this way until getting improvements.

use ~/venv/bin

### More Added
- Are all the previous instructions (ideas) checked? (previously I thought gate cost of aabb is 1.5 which was not, so recheck them)
- Can we augment some auxiliary points to make the pruning better?
- Can we materialize something? Like queries?
- Do the top-k points for the queries intersect? How diverse they are? If they are not diverse, it is a good sign.
- What if I already know something about the threshold? It is not any random scalar. Like a smaller interval.