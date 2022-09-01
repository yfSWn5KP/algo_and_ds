---
recall: header
---

### What is a Permutation?

A Permutation is a unique ordering of a set.
```python
permutations([1, 2, 3]) == [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
```


### How many Permutations does a set have?

num_permutations == n!
>
Where `n` is the number of elements in the set


### Generate all Permutations using Backtracking
>
```python
def permutations(nums):
  def iter_permutations(num_i):
    if 'COMPLETE ME':
      yield list(nums)
      return
    for iter_num_i in range(num_i, len(nums)):
      'COMPLETE ME'
  return list(iter_permutations(0))
```

```python
def permutations(nums):
  def iter_permutations(num_i):
    if num_i == len(nums):
      yield list(nums)
      return
    for iter_num_i in range(num_i, len(nums)):
      nums[num_i], nums[iter_num_i] = nums[iter_num_i], nums[num_i]
      yield from iter_permutations(num_i + 1)
      nums[num_i], nums[iter_num_i] = nums[iter_num_i], nums[num_i]
  return list(iter_permutations(0))
```


### What is an Anagram?

An Anagram is a single Permutation.


### What is a Combination?

A combination is a unique, unordered subset.
```python
combos([1, 2, 3]) == [[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
```


### How many Combinations does a set have?

<img src="https://render.githubusercontent.com/render/math?math=\color{white}\frac{n!}{k!(n-k)!}#gh-dark-mode-only" width="15%">
<img src="https://render.githubusercontent.com/render/math?math=\color{black}\frac{n!}{k!(n-k)!}#gh-light-mode-only" width="15%">
 
Where `n` is the number of elements in the set and `k` is the number of elements in each subset.


### Generate all Combination Subsets using Backtracking
>
```python
def subsets(nums):
  subset = []
  def iter_subsets(num_i):
    yield list(subset)
    for iter_num_i in range(num_i, len(nums)):
      'COMPLETE ME'
  return list(iter_subsets(0))
```

```python
def subsets(nums):
  subset = []
  def iter_subsets(num_i):
    yield list(subset)
    for iter_num_i in range(num_i, len(nums)):
      subset.append(nums[iter_num_i])
      yield from iter_subsets(iter_num_i + 1)
      del subset[-1]
  return list(iter_subsets(0))
```


### What is `0!` equal to?

`0! == 1` because `0!` is an [empty product](https://en.wikipedia.org/wiki/Empty_product), which equals the multiplicative identity of `1`
