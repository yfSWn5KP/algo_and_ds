---
recall: header
---

### When should you consider using a Disjoint Set Union?

Consider using a DSU if you find yourself wanting to join separate entities via bridging.


### Complete the Disjoint Set Union by Size:
```python
class DSU:
  def __init__(self, n):
    self.graph = {}
    self.size = {}
  
  def ensure_node(self, key):
    if key not in self.graph:
      'COMPLETE ME'
  
  def find_root(self, key):
    if self.graph[key] != key:
      self.graph[key] = 'COMPLETE ME'
    return self.graph[key]

  def union(self, key_a, key_b):
    root_a = self.find_root(key_a)
    root_b = self.find_root(key_b)
    if root_a != root_b:
      'COMPLETE ME'
```

```python
class DSU:
  def __init__(self, n):
    self.graph = {}
    self.size = {}
  
  def ensure_node(self, key):
    if key not in self.graph:
      self.graph[key] = key
      self.size[key] = 1
  
  def find_root(self, key):
    if self.graph[key] != key:
      self.graph[key] = self.find_root(self.graph[key])
    return self.graph[key]

  def union(self, key_a, key_b):
    root_a = self.find_root(key_a)
    root_b = self.find_root(key_b)
    if root_a != root_b:
      if self.size[root_a] > self.size[root_b]:
        root_a, root_b = root_b, root_a
      self.graph[root_a] = root_b
      self.size[root_b] += self.size[root_a]
```
 
[further reading](https://leetcode.com/discuss/general-discussion/1072418/Disjoint-Set-Union-(DSU)Union-Find-A-Complete-Guide)
