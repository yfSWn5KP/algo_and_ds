---
recall: header
---

### Write a recursive Inorder Traversal of a Binary Tree in 4 loc

```python
def inorder_recursive(node):
  if node:
    yield from inorder_recursive(node.left)
    yield node
    yield from inorder_recursive(node.right)
```
