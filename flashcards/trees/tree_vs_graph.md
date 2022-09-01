---
recall: header
---

### What 2 properties distinguish a tree from a graph?
A tree is...

1) Totally connected
1) Acyclic


### How can you quickly check if a graph *might* be a tree?

```python
def is_maybe_tree(num_edges, num_nodes):
  return num_edges == (num_nodes - 1)
```
