---
recall: header
---

### The goal of 2-Pointer is to reduce time complexity by narrowing the search window.
 
Sometimes, the left and right pointers are enough, but what should you consider if they're not?

Consider using a current/running pointer.
 
For example, the [Dutch Flag problem](https://leetcode.com/problems/sort-colors/) uses a current/runner to iterate over the input while swapping againt left or right as necessary.
