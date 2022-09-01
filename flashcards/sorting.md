---
recall: header
---

### Complete this Binary Search method:
 
```python
def search(needle, haystack):
  left_i = 0
  right_i = 'COMPLETE ME'
  while 'COMPLETE ME':
    mid_i = (left_i + right_i) // 2
    'COMPLETE ME'
  return None
```

```python
def search(needle, haystack):
  left_i = 0
  right_i = len(haystack) - 1
  while left_i <= right_i:
    mid_i = (left_i + right_i) // 2
    if needle < haystack[mid_i]:
      right_i = mid_i - 1
    elif needle > haystack[mid_i]:
      left_i = mid_i + 1
    else:
      return mid_i
  return None
```

### Complete this Bisect method:
 
```python
def bisect(needle, haystack, left=True):
  compare = 'COMPLETE ME'
  left_i = 0
  right_i = 'COMPLETE ME'
  while 'COMPLETE ME':
    mid_i = (left_i + right_i) // 2
    if compare(needle, haystack):
      'COMPLETE ME'
    else:
      'COMPLETE ME'
  return left_i
```

```python
def bisect(needle, haystack, left=True):
  compare = int.__lteq__ if left else int.__lt__
  left_i = 0
  right_i = len(haystack)
  while left_i < right_i:
    mid_i = (left_i + right_i) // 2
    if compare(needle, haystack[mid_i]):
      right_i = mid_i
    else:
      left_i = mid_i + 1
  return left_i
```


### Complete this Radix Sorting method:
 
```python
def sort_array(nums):
  for bit_idx in range('COMPLETE ME'):
    nums = list(iter_radix_sort(nums, bit_idx))
  return list(iter_sort_by_sign(nums))

def iter_radix_sort(nums, bit_idx):
  'COMPLETE ME'

def iter_sort_by_sign(nums):
  'COMPLETE ME'
```

```python
def sort_array(nums):
  max_abs_num = abs(max(nums, key=lambda n: abs(n)))
  bit_len = len(bin(max_abs_num)) - 2
  for bit_idx in range(bit_len):
    nums = list(iter_radix_sort(nums, bit_idx))
  return list(iter_sort_by_sign(nums))

def get_bit(num, bit_idx):
  return 1 & (num >> bit_idx)

def iter_radix_sort(nums, bit_idx):
  one_bits = []
  for num in nums:
    if get_bit(num, bit_idx):
      one_bits.append(num)
    else:
      yield num
  yield from one_bits

def iter_sort_by_sign(nums):
  positive_nums = []
  for num in nums:
    if num < 0:
      yield num
    else:
      positive_nums.append(num)
  yield from positive_nums
```


### What is the Time & Space Complexity for Radix Sort?

Time is `O(n)` ≈ `O(constant_num_bits * n)`
 
Space is `O(n)` (you create a new array)


### Complete this Selection Sort method:
 
```python
def selection_sort(nums):
  for num_i in range(len(nums)):
    'COMPLETE ME'
```

```python
def selection_sort(nums):
  for num_i in range(len(nums)):
    min_i = next_min_idx(nums, num_i)
    nums[num_i], nums[min_i] = nums[min_i], nums[num_i]

def next_min_idx(nums, start_idx):
  return min(
    range(start_idx, len(nums)),
    key=lambda num_i: nums[num_i]
  )
```
 
[further reading](https://www.interviewcake.com/concept/python3/selection-sort)


### What is the Time & Space Complexity of Selection Sort?

Time is `O(n^2)`
 
Space is `O(1)`


### Complete this Insertion Sort method:
 
```python
def insertion_sort(nums):
  for num_i in range(1, len(nums)):
    'COMPLETE ME'
  return nums
```

```python
def insertion_sort(nums):
  for num_i in range(1, len(nums)):
    while num_i and (nums[num_i - 1] > nums[num_i]):
      nums[num_i - 1], nums[num_i] = nums[num_i], nums[num_i - 1]
      num_i -= 1
  return nums
```
 
[further reading](https://www.interviewcake.com/concept/python3/insertion-sort)


### What is the Time & Space Complexity of Insertion Sort?

Time is `O(n^2)` with possible best case of `O(n)`
 
Space is `O(1)`


### Complete this Merge Sort method:
 
```python
def merge_sort(nums):
  if len(nums) > 1:
    mid_i = len(nums) // 2
    'COMPLETE ME'
  return nums

def ordered_merge(left, right):
  merged = []
  
  left_i = right_i = 0
  while (left_i < len(left)) and (right_i < len(right)):
    'COMPLETE ME'
  
  while left_i < len(left):
    'COMPLETE ME'
  
  while right_i < len(right):
    'COMPLETE ME'
  
  return merged
```

```python
def merge_sort(nums):
  if len(nums) > 1:
    mid_i = len(nums) // 2
    left = merge_sort(nums[:mid_i])
    right = merge_sort(nums[mid_i:])
    nums = ordered_merge(left, right)
  return nums

def ordered_merge(left, right):
  merged = []
  
  left_i = right_i = 0
  while (left_i < len(left)) and (right_i < len(right)):
    if left[left_i] < right[right_i]:
      merged.append(left[left_i])
      left_i += 1
    else:
      merged.append(right[right_i])
      right_i += 1
  
  while left_i < len(left):
    merged.append(left[left_i])
    left_i += 1
  
  while right_i < len(right):
    merged.append(right[right_i])
    right_i += 1
  
  return merged
```
 
[further reading](https://www.interviewcake.com/concept/python3/merge-sort)


### What is the Time & Space Complexity of Merge Sort?

Time is `O(n⋅log(n))`
 
Space is `O(n)`


### Complete this Quick Sort method:
 
```python
def quick_sort(nums):
  partition_and_sort(nums, 0, len(nums) - 1)
  return nums

def partition_and_sort(nums, left_i, right_i):
  if left_i >= right_i:
    return
  'COMPLETE ME'

def set_partition(nums, left_i, right_i):
  'COMPLETE ME'
  
  while left_i <= right_i:
    'COMPLETE ME'
  
  swap_elems(nums, left_i, compare_i)
  return left_i

def swap_elems(nums, i, j):
  nums[i], nums[j] = nums[j], nums[i]
```

```python
def quick_sort(nums):
  partition_and_sort(nums, 0, len(nums) - 1)
  return nums

def partition_and_sort(nums, left_i, right_i):
  if left_i >= right_i:
    return
  part_i = set_partition(nums, left_i, right_i)
  partition_and_sort(nums, left_i, part_i - 1)
  partition_and_sort(nums, part_i + 1, right_i)

def set_partition(nums, left_i, right_i):
  swap_elems(nums, random.randint(left_i, right_i), right_i)
  compare_i = right_i
  right_i -= 1
  
  while left_i <= right_i:
    if nums[left_i] <= nums[compare_i]:
      left_i += 1
    elif nums[right_i] >= nums[compare_i]:
      right_i -= 1
    else:
      swap_elems(nums, left_i, right_i)
  
  swap_elems(nums, left_i, compare_i)
  return left_i

def swap_elems(nums, i, j):
  nums[i], nums[j] = nums[j], nums[i]
```
 
[further reading](https://www.interviewcake.com/concept/python3/quicksort)


### What is the Time & Space Complexity of Quick Sort?

Time is `O(n⋅log(n))` on average but has worst case `O(n^2)` when input is already sorted (because the non-random pivot is always the last element)
 
Space is `O(n⋅log(n))`


### How does Quick Select work?

It's like Quick *Sort*, except you only traverse the partition containing your answer.
 
Result is found once the partition index points at your answer.


### Complete this Quick Select Method:
 
```python
def quick_select(arr, target_i):
  left_i = 0
  right_i = len(arr) - 1
  while True:
    'COMPLETE ME'

def set_partition(nums, left_i, right_i):
  '''
  This method is identical to the one used in Quick Sort
  '''
  swap_elems(nums, random.randint(left_i, right_i), right_i)
  compare_i = right_i
  right_i -= 1
  
  while left_i <= right_i:
    if nums[left_i] <= nums[compare_i]:
      left_i += 1
    elif nums[right_i] >= nums[compare_i]:
      right_i -= 1
    else:
      swap_elems(nums, left_i, right_i)
  
  swap_elems(nums, left_i, compare_i)
  return left_i

def swap_elems(nums, i, j):
  nums[i], nums[j] = nums[j], nums[i]
```

```python
def quick_select(arr, target_i):
  left_i = 0
  right_i = len(arr) - 1
  while True:
    part_i = set_partition(arr, left_i, right_i)
    if part_i < target_i:
      left_i = part_i + 1
    elif part_i > target_i:
      right_i = part_i - 1
    else:
      break

def set_partition(nums, left_i, right_i):
  '''
  This method is identical to the one used in Quick Sort
  '''
  swap_elems(nums, random.randint(left_i, right_i), right_i)
  compare_i = right_i
  right_i -= 1
  
  while left_i <= right_i:
    if nums[left_i] <= nums[compare_i]:
      left_i += 1
    elif nums[right_i] >= nums[compare_i]:
      right_i -= 1
    else:
      swap_elems(nums, left_i, right_i)
  
  swap_elems(nums, left_i, compare_i)
  return left_i

def swap_elems(nums, i, j):
  nums[i], nums[j] = nums[j], nums[i]
```


### What is the Time & Space Complexity of Quick Select?

Time is `O(n)` -- due to "median of medians"
 
Space is `O(1)`
