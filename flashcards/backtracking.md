---
recall: header
---

### How should you think about the time complexity of backtracking?

Backtracking traverses the tree of possible solutions.
 
So it's ultimately a question of how many nodes there are in the tree.
 
Recall that `num_nodes` == `num_branches`<sup>`tree_height`</sup> - 1
 
So to measure the time complexity, you must determine:
 
1. How many branching options there are in each backtrack
2. The maximum recursion depth
