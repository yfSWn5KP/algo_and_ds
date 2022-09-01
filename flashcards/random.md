---
recall: header
---

### Complete this Reservoir Sampling method:
 
```python
def reservoir_sample(nums, target):
  idx = None
  cnt = 0
  for num_i, num in enumerate(nums):
    'COMPLETE ME'
  return idx
```

```python
def reservoir_sample(nums, target):
  idx = None
  cnt = 0
  for num_i, num in enumerate(nums):
    if target == num:
      cnt += 1
      if 0 == random.randrange(cnt):
        idx = num_i
  return idx
```


### What is the Time & Space Complexity of Reservoir Sampling?

Time is `O(n)`
 
Space is `O(1)`


### When is Reservoir Sampling Useful?

You randomly sample from an input stream, without knowing the number of occurrences in advance.
