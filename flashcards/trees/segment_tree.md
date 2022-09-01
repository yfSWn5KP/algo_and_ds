---
recall: header
---

### Complete this Segment Tree:
 
```python
 # type is just an indicative placeholder
 # it can be Any that `aggregate_method` supports
VALUE_TYPE = int


class Node(object):
  def __init__(self, left_i: int, right_i: int):
    self.left_i = left_i
    self.right_i = right_i
    self.value: VALUE_TYPE = None
    self.left: Node = None
    self.right: Node = None


class SegmentTree(object):
  def __init__(self, values, aggregate_method):
    self.aggregate_method = aggregate_method
    self.root = self._init_tree(values, 0, len(values) - 1)
  
  def update(self, idx, val):
    self._update(self.root, idx, val)
  
  def query_range(self, left_i, right_i) -> VALUE_TYPE:
    return self._query_range(self.root, left_i, right_i)
  
  def _init_tree(self, values, left_i, right_i) -> Optional[Node]:
    'COMPLETE ME'
  
  def _update(self, root: Node, idx, val):
    'COMPLETE ME'
  
  def _query_range(self, root: Node, left_i, right_i):
    'COMPLETE ME'
```

```python
 # type is just an indicative placeholder
 # it can be Any that `aggregate_method` supports
VALUE_TYPE = int


class Node(object):
  def __init__(self, left_i: int, right_i: int):
    '''
    Range is inclusive
    '''
    self.left_i = left_i
    self.right_i = right_i
    self.value: VALUE_TYPE = None
    self.left: Node = None
    self.right: Node = None


class SegmentTree(object):
  def __init__(self, values, aggregate_method):
    '''
    Time = O(n)
    '''
    self.aggregate_method = aggregate_method
    self.root = self._init_tree(values, 0, len(values) - 1)
  
  def update(self, idx, val):
    '''
    Time = O(log(n))
    '''
    self._update(self.root, idx, val)
  
  def query_range(self, left_i, right_i) -> VALUE_TYPE:
    '''
    Time = O(log(n))
    Range is inclusive
    '''
    return self._query_range(self.root, left_i, right_i)
  
  def _init_tree(self, values, left_i, right_i) -> Optional[Node]:
    if left_i > right_i:
      return None
    
    root = Node(left_i, right_i)
    if left_i == right_i:
      root.value = values[left_i]
    else:
      mid_i = (left_i + right_i) // 2
      root.left = self._init_tree(values, left_i, mid_i)
      root.right = self._init_tree(values, mid_i + 1, right_i)
      root.value = self.aggregate_method(
        root.left.value,
        root.right.value
      )
    return root
  
  def _update(self, root: Node, idx, val):
    if root.left_i == root.right_i:
      root.value = val
      return
  
    mid_i = (root.left_i + root.right_i) // 2
    if idx <= mid_i:
      self._update(root.left, idx, val)
    else:
      self._update(root.right, idx, val)
    root.value = self.aggregate_method(
      root.left.value,
      root.right.value
    )
  
  def _query_range(self, root: Node, left_i, right_i):
    if (root.left_i == left_i) and (root.right_i == right_i):
      return root.value
    
    mid_i = (root.left_i + root.right_i) // 2
    if right_i <= mid_i:
      return self._query_range(root.left, left_i, right_i)
    elif left_i >= mid_i + 1:
      return self._query_range(root.right, left_i, right_i)
    else:
      return self.aggregate_method(
        self._query_range(root.left, left_i, mid_i),
        self._query_range(root.right, mid_i + 1, right_i)
      )
```
