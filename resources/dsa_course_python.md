# Data Structures and Algorithms (DSA) in Python — Beginner to Interview-Ready

This is a **complete DSA course in Python** designed for absolute beginners and interview prep. It focuses on:

- Building strong fundamentals (DS + Big-O + clean Python)
- Learning **problem-solving patterns** (so you can generalize to new problems)
- Practicing with curated problem sets (LeetCode/HackerRank style)

If you are following this repository primarily for Machine Learning: treat this as an **optional but highly recommended** track for interviews and for writing efficient data-processing code.

---

## Course Objectives

- Build strong DSA fundamentals using Python
- Develop a **pattern-based problem-solving** mindset
- Write clean, optimized, interview-ready code
- Prepare for product-based company interviews and competitive programming
- Build a solid base for advanced CS/ML and ML system design

## Who This Course Is For

- Beginners starting DSA in Python
- Working professionals preparing for interviews
- Students aiming for software engineering or ML roles
- Anyone who wants to improve logical thinking & problem solving

## Prerequisites

- Comfortable with Python basics (variables, loops, functions, lists/dicts)
- Recommended: `00-prerequisites/01-python-basics.md` (especially the Big-O section)

---

## How to Use This Course (Beginner-Friendly)

For each topic:

1. **Understand the core operations** (what it does, what it costs)
2. **Memorize one clean template** (minimal code you can reproduce)
3. **Simulate by hand** on small inputs (the “trace tables” below)
4. **Solve 5–15 problems** before moving on

### A Simple Problem-Solving Checklist

- **Step 1**: What is the input size? What’s the time limit? (choose `O(n)` vs `O(n log n)`)
- **Step 2**: Can you restate the task as a pattern? (two pointers, sliding window, BFS…)
- **Step 3**: What data structure makes the “key operation” fast?
- **Step 4**: Write a first correct solution, then optimize
- **Step 5**: Add edge cases (empty, 1 element, duplicates, negative numbers)

---

## Table of Contents

- [1. Introduction to DSA & Python for Problem Solving](#1-introduction-to-dsa-python-for-problem-solving)
- [2. Arrays and Lists](#2-arrays-and-lists)
- [3. Strings](#3-strings)
- [4. Linked Lists](#4-linked-lists)
- [5. Stacks](#5-stacks)
- [6. Queues and Deques](#6-queues-and-deques)
- [7. Searching](#7-searching)
- [8. Sorting](#8-sorting)
- [9. Recursion](#9-recursion)
- [10. Hashing (Hash Maps & Hash Sets)](#10-hashing-hash-maps-hash-sets)
- [11. Patterns & Problem Solving](#11-patterns-problem-solving)
- [12. Trees](#12-trees)
- [13. Graphs](#13-graphs)
- [14. Heaps and Priority Queues](#14-heaps-and-priority-queues)
- [15. Dynamic Programming](#15-dynamic-programming)
- [Appendix A: Interview Templates (Copy-Paste Friendly)](#appendix-a-interview-templates-copy-paste-friendly)
- [Appendix B: Practice Plan (6-8 Weeks)](#appendix-b-practice-plan-6-8-weeks)
- [Appendix C: Curated Problem Set](#appendix-c-curated-problem-set)

---

## 1. Introduction to DSA & Python for Problem Solving

### What is DSA (and why it matters)?

DSA is about **choosing the right way to store data** (data structures) and **the right steps to process it** (algorithms).

In interviews, most problems are testing:

- Can you recognize a pattern?
- Can you write correct code?
- Can you hit the required complexity?

### Time Complexity (Big-O) — the 80/20

- `O(1)`: constant (hash lookup average)
- `O(log n)`: binary search
- `O(n)`: one pass
- `O(n log n)`: sorting, many divide-and-conquer
- `O(n^2)`: nested loops (often too slow when `n` is big)

### Example: “Is this `O(n)` or `O(n^2)`?”

```python
def has_duplicate(nums):
    seen = set()
    for x in nums:          # n iterations
        if x in seen:       # O(1) average
            return True
        seen.add(x)         # O(1) average
    return False
```

- **Time**: `O(n)` average
- **Space**: `O(n)`

### Python essentials for coding interviews

- Use `dict`/`set` for fast membership
- Use `collections.deque` for queues and sliding windows
- Use `heapq` for priority queues
- Prefer readable variable names and small helper functions

---

## 2. Arrays and Lists

Arrays/lists are the most common interview input form.

### Core operations (list)

- **Index access**: `O(1)`
- **Append**: `O(1)` amortized
- **Insert/Delete in middle**: `O(n)`
- **Membership test (`x in list`)**: `O(n)`

### Prefix sums (pattern)

Used when you need **range sums** quickly.

```python
def build_prefix(nums):
    prefix = [0]
    for x in nums:
        prefix.append(prefix[-1] + x)
    return prefix

def range_sum(prefix, left, right):
    # sum(nums[left:right+1])
    return prefix[right + 1] - prefix[left]
```

#### Simulation (trace)

For `nums = [2, -1, 3, 4]`:

- `prefix[0] = 0`
- `prefix[1] = 2`
- `prefix[2] = 1`
- `prefix[3] = 4`
- `prefix[4] = 8`

Range sum `left=1, right=3`:

- `prefix[4] - prefix[1] = 8 - 2 = 6` → `(-1 + 3 + 4) = 6`

### Sliding window (pattern)

Used when you need the **best/longest** subarray that satisfies a condition.

Example: max sum of size `k`.

```python
def max_sum_fixed_k(nums, k):
    window_sum = sum(nums[:k])
    best = window_sum

    for r in range(k, len(nums)):
        window_sum += nums[r]
        window_sum -= nums[r - k]
        best = max(best, window_sum)
    return best
```

#### Simulation (trace)

`nums = [1, 4, 2, 10, 2]`, `k=3`

- window `[1,4,2]` sum=7 best=7
- slide: add 10 remove 1 → sum=16 best=16 (window `[4,2,10]`)
- slide: add 2 remove 4 → sum=14 best=16 (window `[2,10,2]`)

---

## 3. Strings

Strings often combine:

- Frequency counting (`dict`)
- Two pointers
- Sliding window

### Character frequency

```python
def char_freq(s):
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    return freq
```

### Two pointers: palindrome check

```python
def is_palindrome(s):
    l, r = 0, len(s) - 1
    while l < r:
        if s[l] != s[r]:
            return False
        l += 1
        r -= 1
    return True
```

#### Simulation (trace)

`s="racecar"`

- l=0 r=6 compare r==r ok
- l=1 r=5 compare a==a ok
- l=2 r=4 compare c==c ok
- stop

### Anagrams (frequency compare)

```python
def is_anagram(a, b):
    if len(a) != len(b):
        return False
    return char_freq(a) == char_freq(b)
```

---

## 4. Linked Lists

Linked lists are about **pointers** (references).

### Node definition

```python
class ListNode:
    def __init__(self, val=0, nxt=None):
        self.val = val
        self.next = nxt
```

### Fast & slow pointers (pattern)

#### Find middle

```python
def middle_node(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

#### Detect cycle (Floyd’s)

```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False
```

#### Simulation (trace)

If there is a cycle, `fast` laps `slow` and they eventually meet.

---

## 5. Stacks

Stack = Last-In-First-Out (LIFO).

### Python stack

```python
stack = []
stack.append(10)   # push
stack.append(20)
x = stack.pop()    # pop -> 20
```

### Valid parentheses (classic)

```python
def is_valid_parentheses(s):
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for ch in s:
        if ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
        else:
            stack.append(ch)
    return not stack
```

#### Simulation (trace)

`s="([])"`

- read `(` stack: `[(]`
- read `[` stack: `[(,[]`
- read `]` pop `[` ok stack: `[(]`
- read `)` pop `(` ok stack: `[]`

### Monotonic stack (pattern)

Used for “next greater element”, “daily temperatures”, “largest rectangle in histogram”, etc.

**Idea**: keep a stack that is monotonic (increasing or decreasing). When a new value breaks the monotonic rule, you pop and resolve answers.

#### Next greater element (to the right)

```python
def next_greater(nums):
    n = len(nums)
    ans = [-1] * n
    stack = []  # indices, nums[stack] is decreasing

    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:
            j = stack.pop()
            ans[j] = x
        stack.append(i)
    return ans
```

#### Simulation (trace)

`nums = [2, 1, 2, 4, 3]`

- i=0 x=2 stack=[0] ans=[-1,-1,-1,-1,-1]
- i=1 x=1 stack=[0,1] (2 > 1)
- i=2 x=2: pop 1 (nums[1]=1 < 2) → ans[1]=2, stack=[0]; push 2 → stack=[0,2]
- i=3 x=4: pop 2 → ans[2]=4, pop 0 → ans[0]=4; push 3 → stack=[3]
- i=4 x=3: push 4 → stack=[3,4]

Final: `ans = [4, 2, 4, -1, -1]`

---

## 6. Queues and Deques

Queue = First-In-First-Out (FIFO).

### Use `deque` (fast)

```python
from collections import deque

q = deque()
q.append(1)       # enqueue
q.append(2)
x = q.popleft()   # dequeue -> 1
```

### Sliding window maximum (deque)

Maintain a deque of indices with values in decreasing order.

```python
from collections import deque

def max_sliding_window(nums, k):
    dq = deque()  # indices
    out = []

    for i, x in enumerate(nums):
        # remove indices out of window
        while dq and dq[0] <= i - k:
            dq.popleft()

        # maintain decreasing values
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)

        if i >= k - 1:
            out.append(nums[dq[0]])
    return out
```

#### Simulation (trace)

`nums=[1, 3, -1, -3, 5]`, `k=3`

- i=0 x=1 dq=[0]
- i=1 x=3 pop 0 (1 <= 3) dq=[1]
- i=2 x=-1 dq=[1,2] → output nums[1]=3
- i=3 x=-3 remove none, dq=[1,2,3] → output nums[1]=3
- i=4 x=5 remove out-of-window index 1, then pop 3,2 (<=5), dq=[4] → output nums[4]=5

Outputs: `[3, 3, 5]`

### Circular queue (concept)

Used when you have a fixed-size buffer (ring). It’s common in system design and low-level problems.

```python
class CircularQueue:
    def __init__(self, k):
        self.k = k
        self.q = [None] * k
        self.head = 0
        self.size = 0

    def enqueue(self, x):
        if self.size == self.k:
            return False
        tail = (self.head + self.size) % self.k
        self.q[tail] = x
        self.size += 1
        return True

    def dequeue(self):
        if self.size == 0:
            return None
        x = self.q[self.head]
        self.q[self.head] = None
        self.head = (self.head + 1) % self.k
        self.size -= 1
        return x
```

### Queue using two stacks (classic)

```python
class QueueWithStacks:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def push(self, x):
        self.in_stack.append(x)

    def pop(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack.pop() if self.out_stack else None
```

---

## 7. Searching

### Linear search

```python
def linear_search(nums, target):
    for i, x in enumerate(nums):
        if x == target:
            return i
    return -1
```

### Binary search (sorted array)

```python
def binary_search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target:
            return m
        if nums[m] < target:
            l = m + 1
        else:
            r = m - 1
    return -1
```

#### Simulation (trace)

`nums=[1,3,4,7,9]`, `target=7`

- l=0 r=4 m=2 nums[m]=4 < 7 → l=3
- l=3 r=4 m=3 nums[m]=7 == 7 → return 3

### Binary search on answer (pattern)

Used when the answer is numeric and you can check feasibility.

Example pattern:

```python
def first_true(lo, hi, pred):
    # find smallest x in [lo, hi] where pred(x) is True
    ans = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if pred(mid):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans
```

### Searching in sorted structures (Python `bisect`)

When you want “first position where `x` can be inserted”:

```python
import bisect

def lower_bound(nums, x):
    return bisect.bisect_left(nums, x)

def upper_bound(nums, x):
    return bisect.bisect_right(nums, x)
```

### Binary search on answer: worked example

Problem type: “minimize the maximum”, “minimum capacity”, “minimum days”, etc.

Example: **Minimum ship capacity** (classic).  
Given `weights` and `days`, find the minimum ship capacity so all weights can be shipped in `days` (in order).

```python
def can_ship(weights, days, cap):
    used_days = 1
    current = 0
    for w in weights:
        if w > cap:
            return False
        if current + w <= cap:
            current += w
        else:
            used_days += 1
            current = w
    return used_days <= days

def min_ship_capacity(weights, days):
    lo, hi = max(weights), sum(weights)
    return first_true(lo, hi, lambda cap: can_ship(weights, days, cap))
```

#### Simulation (trace)

`weights=[3,2,2,4,1,4]`, `days=3`

- lo=4 hi=16
- mid=10 → can_ship=True → hi=9 ans=10
- mid=6  → can_ship=True → hi=5 ans=6
- mid=4  → can_ship=False → lo=5
- mid=5  → can_ship=True → hi=4 ans=5

Answer: `5`

---

## 8. Sorting

In interviews, you usually use Python’s built-in sort, but you must understand the ideas.

### Python sort (Timsort)

```python
nums = [3, 1, 2]
nums.sort()                   # in-place
sorted_nums = sorted(nums)    # new list
```

### Merge sort (concept + implementation)

```python
def merge_sort(nums):
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])
    return merge(left, right)

def merge(a, b):
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
```

### Other common sorts (what to remember)

| Algorithm | Best | Average | Worst | Stable | Notes |
|----------|------|---------|-------|--------|------|
| Bubble | O(n) | O(n^2) | O(n^2) | Yes | Mostly for learning |
| Selection | O(n^2) | O(n^2) | O(n^2) | No | Few swaps, still slow |
| Insertion | O(n) | O(n^2) | O(n^2) | Yes | Great when nearly sorted |
| Merge | O(n log n) | O(n log n) | O(n log n) | Yes | Extra memory |
| Quick | O(n log n) | O(n log n) | O(n^2) | No | Fast in practice |
| Counting | O(n + k) | O(n + k) | O(n + k) | Yes | Only for small integer ranges (`k` = value range) |

### Insertion sort (good for nearly-sorted)

```python
def insertion_sort(nums):
    a = nums[:]
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a
```

### Quick sort (recursive, educational)

In practice, use `sorted()` / `.sort()`.

```python
def quick_sort(nums):
    if len(nums) <= 1:
        return nums
    pivot = nums[len(nums) // 2]
    left = [x for x in nums if x < pivot]
    mid = [x for x in nums if x == pivot]
    right = [x for x in nums if x > pivot]
    return quick_sort(left) + mid + quick_sort(right)
```

### Counting sort (integers in a small range)

```python
def counting_sort(nums):
    if not nums:
        return []
    if min(nums) < 0:
        raise ValueError("counting_sort assumes non-negative integers")

    m = max(nums)
    count = [0] * (m + 1)
    for x in nums:
        count[x] += 1

    out = []
    for value, freq in enumerate(count):
        out.extend([value] * freq)
    return out
```

---

## 9. Recursion

Recursion = a function that calls itself. Always define:

- **Base case** (stop)
- **Recursive case** (progress)

### Factorial

```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
```

### Memoization (avoid repeated work)

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
```

### Recursion tree (how to think)

For `fib(4)` (without memoization), the calls expand like:

- `fib(4)` calls `fib(3)` and `fib(2)`
- `fib(3)` calls `fib(2)` and `fib(1)`
- `fib(2)` calls `fib(1)` and `fib(0)`

The key issue is **repeated subproblems** (`fib(2)` happens multiple times), which memoization fixes.

### Backtracking (generate all solutions)

Backtracking is recursion that explores choices and then “undoes” them.

Example: generate all subsets.

```python
def subsets(nums):
    out = []
    path = []

    def backtrack(i):
        if i == len(nums):
            out.append(path[:])
            return
        # choice 1: skip nums[i]
        backtrack(i + 1)
        # choice 2: take nums[i]
        path.append(nums[i])
        backtrack(i + 1)
        path.pop()

    backtrack(0)
    return out
```

#### Simulation (trace)

`nums=[1,2]`

- i=0 skip 1 → i=1 skip 2 → add []
- i=1 take 2 → add [2]
- i=0 take 1 → i=1 skip 2 → add [1]
- i=1 take 2 → add [1,2]

---

## 10. Hashing (Hash Maps & Hash Sets)

### Two-sum (classic)

```python
def two_sum(nums, target):
    seen = {}  # value -> index
    for i, x in enumerate(nums):
        need = target - x
        if need in seen:
            return [seen[need], i]
        seen[x] = i
    return None
```

### Subarray sum equals K (prefix sum + hash map)

```python
def subarray_sum_equals_k(nums, k):
    count = 0
    prefix = 0
    freq = {0: 1}

    for x in nums:
        prefix += x
        count += freq.get(prefix - k, 0)
        freq[prefix] = freq.get(prefix, 0) + 1
    return count
```

#### Simulation (trace)

`nums=[1,2,1,2]`, `k=3`

- start: prefix=0 freq={0:1} count=0
- x=1: prefix=1, need=-2 (0), freq[1]=1
- x=2: prefix=3, need=0 (+1) count=1, freq[3]=1
- x=1: prefix=4, need=1 (+1) count=2, freq[4]=1
- x=2: prefix=6, need=3 (+1) count=3, freq[6]=1

Answer: `3` subarrays sum to 3 → `[1,2]`, `[2,1]`, `[1,2]`

### Hash set (fast membership)

```python
def has_duplicate(nums):
    seen = set()
    for x in nums:
        if x in seen:
            return True
        seen.add(x)
    return False
```

### Frequency counting (common in strings)

```python
from collections import Counter

def most_common_char(s):
    c = Counter(s)
    ch, _ = c.most_common(1)[0]
    return ch
```

---

## 11. Patterns & Problem Solving

This section consolidates the “big patterns” you’ll reuse constantly:

- Sliding window
- Two pointers
- Prefix sums
- Fast & slow pointers
- Binary search pattern
- Greedy basics

Each pattern becomes a **template** + **practice set**.

### 11.1 Sliding window (variable) — example: longest substring without repeating

```python
def longest_unique_substring(s):
    seen = set()
    l = 0
    best = 0
    for r, ch in enumerate(s):
        while ch in seen:
            seen.remove(s[l])
            l += 1
        seen.add(ch)
        best = max(best, r - l + 1)
    return best
```

#### Simulation (trace)

`s="abba"`

- r=0 'a' seen={a} l=0 best=1
- r=1 'b' seen={a,b} l=0 best=2
- r=2 'b' already in seen → remove s[l]='a' (l=1), still 'b' in seen → remove s[l]='b' (l=2), now add 'b' → seen={b}, best=2
- r=3 'a' add → seen={b,a} best=2

### 11.2 Two pointers (same direction) — remove duplicates in sorted array

```python
def remove_duplicates(nums):
    if not nums:
        return 0
    write = 1
    for read in range(1, len(nums)):
        if nums[read] != nums[read - 1]:
            nums[write] = nums[read]
            write += 1
    return write  # new length
```

### 11.3 Prefix sum (range queries and subarray counting)

- Range sum: use a prefix array
- Counting subarrays with sum `k`: prefix sum + hash map (Section 10)

### 11.4 Fast & slow pointers (cycle / middle)

- Cycle detection: Floyd’s algorithm (Section 4)
- Middle of linked list: move `fast` twice as quickly (Section 4)

### 11.5 Binary search pattern

- Only works when the condition is **monotonic** (false…false…true…true)
- Classic: search in sorted array
- “On answer”: minimize capacity/days/maximum (Section 7)

### 11.6 Greedy basics (local best choice)

Greedy works when making the best local choice leads to a global optimum.

Example: activity selection (choose the earliest finishing activity).

```python
def max_non_overlapping(intervals):
    intervals = sorted(intervals, key=lambda x: x[1])  # sort by end time
    taken = 0
    last_end = float("-inf")
    for start, end in intervals:
        if start >= last_end:
            taken += 1
            last_end = end
    return taken
```

---

## 12. Trees

Trees show up everywhere (and in ML: decision trees, ASTs, hierarchical data).

### Basic binary tree node

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### DFS traversals (recursive)

```python
def inorder(root):
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)
```

### Tree traversals (the “big 4”)

#### Preorder (root, left, right)

```python
def preorder(root):
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)
```

#### Postorder (left, right, root)

```python
def postorder(root):
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]
```

#### Level-order (BFS)

```python
from collections import deque

def level_order(root):
    if not root:
        return []
    q = deque([root])
    out = []
    while q:
        node = q.popleft()
        out.append(node.val)
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return out
```

#### Simulation (trace)

For this tree:

```
    2
   / \
  1   3
```

- inorder: `[1, 2, 3]`
- preorder: `[2, 1, 3]`
- postorder: `[1, 3, 2]`
- level-order: `[2, 1, 3]`

### Binary Search Tree (BST)

BST property: `left < root < right` (for unique values).

```python
def bst_search(root, target):
    cur = root
    while cur:
        if cur.val == target:
            return True
        if target < cur.val:
            cur = cur.left
        else:
            cur = cur.right
    return False
```

### Lowest Common Ancestor (LCA) in a BST

In a BST, you can walk down from the root:

```python
def lca_bst(root, p, q):
    cur = root
    while cur:
        if p < cur.val and q < cur.val:
            cur = cur.left
        elif p > cur.val and q > cur.val:
            cur = cur.right
        else:
            return cur
    return None
```

### N-ary trees (general trees)

```python
class NaryNode:
    def __init__(self, val=0, children=None):
        self.val = val
        self.children = children or []

def preorder_nary(root):
    if not root:
        return []
    out = [root.val]
    for child in root.children:
        out.extend(preorder_nary(child))
    return out
```

### Tree DP (pattern): diameter of binary tree

Tree DP often means: compute something for each node using results from children.

```python
def diameter(root):
    best = 0

    def height(node):
        nonlocal best
        if not node:
            return 0
        lh = height(node.left)
        rh = height(node.right)
        best = max(best, lh + rh)
        return 1 + max(lh, rh)

    height(root)
    return best
```

---

## 13. Graphs

Graphs are about relationships. Most problems use:

- BFS (shortest path in unweighted graphs)
- DFS (explore / detect cycles)
- Dijkstra (weighted shortest path)
- DSU (connectivity)

### Adjacency list

```python
from collections import defaultdict

graph = defaultdict(list)
graph[0].append(1)
graph[1].append(2)
```

### BFS template

```python
from collections import deque

def bfs(start, graph):
    q = deque([start])
    seen = {start}
    order = []
    while q:
        node = q.popleft()
        order.append(node)
        for nei in graph[node]:
            if nei not in seen:
                seen.add(nei)
                q.append(nei)
    return order
```

### Adjacency matrix vs adjacency list

- **Adjacency matrix**: `matrix[u][v] = 1/weight`  
  - Fast edge check `O(1)`, high memory `O(V^2)`
- **Adjacency list**: `graph[u] = [v1, v2, ...]` or `[(v,w), ...]`  
  - Memory `O(V+E)`, best for sparse graphs (most interview problems)

### Shortest path in an unweighted grid (state space graph)

Grid problems are graphs where each cell is a node.

```python
from collections import deque

def shortest_path_grid(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    sr, sc = start
    gr, gc = goal

    q = deque([(sr, sc)])
    dist = {(sr, sc): 0}

    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            return dist[(r, c)]

        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                if (nr, nc) not in dist:
                    dist[(nr, nc)] = dist[(r, c)] + 1
                    q.append((nr, nc))

    return None
```

#### Simulation (trace)

`0` = free cell, `1` = blocked.

```
grid =
0 0 1
0 0 0
1 0 0
start=(0,0) goal=(2,2)
```

BFS expands layer-by-layer:

- distance 0: (0,0)
- distance 1: (1,0), (0,1)
- distance 2: (1,1)
- distance 3: (2,1), (1,2)
- distance 4: (2,2) reached

Answer: `4`

### Weighted shortest path: Dijkstra (when edges have weights)

```python
import heapq

def dijkstra(start, graph):
    # graph[u] = [(v, w), ...]
    dist = {start: 0}
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist.get(u, float("inf")):
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist
```

### Other shortest path algorithms (when to use)

- **0-1 BFS**: when edge weights are only 0 or 1 (faster than Dijkstra)
- **Bellman-Ford**: when negative edge weights exist (can detect negative cycles)
- **Floyd-Warshall**: all-pairs shortest paths for small graphs (`O(V^3)`)

### DSU (Union-Find): connectivity / components

Use DSU when edges come in any order and you need “are these connected?” quickly.

```python
def count_components(n, edges):
    dsu = DSU(n)
    comps = n
    for a, b in edges:
        if dsu.union(a, b):
            comps -= 1
    return comps
```

### Minimum Spanning Tree (Kruskal)

```python
def kruskal_mst_weight(n, edges):
    # edges = [(w, u, v), ...]
    edges = sorted(edges)
    dsu = DSU(n)
    total = 0
    used = 0
    for w, u, v in edges:
        if dsu.union(u, v):
            total += w
            used += 1
            if used == n - 1:
                break
    return total if used == n - 1 else None
```

### DAGs (Directed Acyclic Graphs)

If the graph has **no cycles**, you can do:

- Topological sort (Appendix A.9)
- DP on DAGs (longest path in DAG, counting paths)

---

## 14. Heaps and Priority Queues

Heaps are used when you repeatedly need:

- the **smallest** item (“min”) or the **largest** item (“max”)
- the **top K** items
- the **next best** choice (scheduling, shortest path, merging streams)

In Python, heaps are implemented with `heapq` and are **min-heaps** by default.

### Core operations (Python `heapq`)

```python
import heapq

heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 2)
heapq.heappush(heap, 8)

smallest = heapq.heappop(heap)   # 2
```

- `heappush`: `O(log n)`
- `heappop`: `O(log n)`
- `heap[0]` (peek min): `O(1)`
- `heapify(list)`: `O(n)`

### Min-heap vs max-heap

Python has only a min-heap. For a max-heap you typically **negate values**:

```python
import heapq

nums = [3, 1, 4]
max_heap = [-x for x in nums]
heapq.heapify(max_heap)

largest = -heapq.heappop(max_heap)   # 4
```

### Pattern 1: Top K largest elements (`O(n log k)`)

This is one of the most common heap interview patterns.

```python
import heapq

def top_k_largest(nums, k):
    heap = []
    for x in nums:
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heapreplace(heap, x)
    return sorted(heap, reverse=True)
```

#### Simulation (trace)

`nums=[5, 1, 9, 2, 7]`, `k=2`

- push 5 → heap=[5]
- push 1 → heap=[1,5] (min is 1)
- x=9 > 1 → replace → heap=[5,9]
- x=2 not > 5 → ignore → heap=[5,9]
- x=7 > 5 → replace → heap=[7,9]

Answer: `[9, 7]`

### Pattern 2: K-th largest (min-heap of size K)

```python
def kth_largest(nums, k):
    heap = nums[:k]
    heapq.heapify(heap)
    for x in nums[k:]:
        if x > heap[0]:
            heapq.heapreplace(heap, x)
    return heap[0]
```

### Pattern 3: Merge K sorted lists (k-way merge)

Use a heap to always pick the smallest “front” element among lists.

```python
import heapq

def merge_k_sorted(lists):
    heap = []
    out = []

    for li, arr in enumerate(lists):
        if arr:
            heapq.heappush(heap, (arr[0], li, 0))  # (value, list_index, element_index)

    while heap:
        val, li, ei = heapq.heappop(heap)
        out.append(val)
        ni = ei + 1
        if ni < len(lists[li]):
            heapq.heappush(heap, (lists[li][ni], li, ni))

    return out
```

#### Simulation (trace)

`lists=[[1,4,7],[2,5],[3,6,9]]`

- heap starts with (1,A0), (2,B0), (3,C0) → pop 1, push 4
- heap has 2,3,4 → pop 2, push 5
- heap has 3,4,5 → pop 3, push 6

Output begins: `[1,2,3,...]` and continues sorted.

### Pattern 4: Scheduling (meeting rooms / CPU tasks)

Classic idea: keep a **min-heap of end times**.

```python
import heapq

def min_meeting_rooms(intervals):
    intervals.sort()  # sort by start
    ends = []         # min-heap of end times

    for start, end in intervals:
        if ends and ends[0] <= start:
            heapq.heapreplace(ends, end)  # reuse room
        else:
            heapq.heappush(ends, end)     # new room

    return len(ends)
```

#### Simulation (trace)

`intervals=[(0,30),(5,10),(15,20)]`

- start 0: push 30 → ends=[30]
- start 5: 30 > 5 → push 10 → ends=[10,30]
- start 15: 10 <= 15 → replace with 20 → ends=[20,30]

Answer: `2`

### Where heaps show up in ML/engineering

- Top-K predictions / Top-K features
- Streaming “best so far” metrics
- Dijkstra / A* (shortest path)
- Scheduling batch jobs

---

## 15. Dynamic Programming

Dynamic Programming (DP) is recursion + caching, or building answers bottom-up.

DP works when a problem has:

- **Overlapping subproblems** (you solve the same smaller problem many times)
- **Optimal substructure** (best solution uses best solutions of subproblems)

### How to recognize DP

If you see:

- “maximum/minimum number of …”
- “count the number of ways …”
- “best score / longest / shortest …”
- “choose or skip …”
- sequences/substrings/subarrays with constraints

…DP is often a strong candidate.

### DP styles

- **Top-down (memoization)**: write recursion, cache results
- **Bottom-up (tabulation)**: build a `dp[]` array/table iteratively

---

### 15.1 1D DP: Climbing Stairs (count ways)

Problem: if you can take 1 or 2 steps, how many ways to reach step `n`?

Recurrence:

- `dp[i] = dp[i-1] + dp[i-2]`
- base: `dp[0]=1`, `dp[1]=1`

```python
def climb_stairs(n):
    if n <= 1:
        return 1
    a, b = 1, 1  # dp[i-2], dp[i-1]
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

#### Simulation (trace)

`n=5`

- i=2: (a,b)=(1,2)
- i=3: (a,b)=(2,3)
- i=4: (a,b)=(3,5)
- i=5: (a,b)=(5,8)

Answer: `8`

---

### 15.2 1D DP: House Robber (max sum with no adjacent)

You can’t pick adjacent houses.

Recurrence:

- `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`

```python
def house_robber(nums):
    prev2 = 0  # dp[i-2]
    prev1 = 0  # dp[i-1]
    for x in nums:
        cur = max(prev1, prev2 + x)
        prev2, prev1 = prev1, cur
    return prev1
```

#### Simulation (trace)

`nums=[2,7,9,3,1]`

- x=2: prev1=2
- x=7: prev1=7
- x=9: prev1=11
- x=3: prev1=11
- x=1: prev1=12

Answer: `12` (2 + 9 + 1)

---

### 15.3 DP on grids (2D): Unique Paths

From top-left to bottom-right, only right/down moves.

Recurrence:

- `dp[r][c] = dp[r-1][c] + dp[r][c-1]`

```python
def unique_paths(rows, cols):
    dp = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        dp[r][0] = 1
    for c in range(cols):
        dp[0][c] = 1

    for r in range(1, rows):
        for c in range(1, cols):
            dp[r][c] = dp[r - 1][c] + dp[r][c - 1]
    return dp[rows - 1][cols - 1]
```

#### Simulation (trace)

`rows=3, cols=4`

DP table becomes:

- row 0: `1 1 1 1`
- row 1: `1 2 3 4`
- row 2: `1 3 6 10`

Answer: `10`

---

### 15.4 Coin Change (classic DP)

Problem: minimum coins to make amount `A`.

```python
def coin_change_min(coins, amount):
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0

    for a in range(1, amount + 1):
        for coin in coins:
            if a - coin >= 0:
                dp[a] = min(dp[a], dp[a - coin] + 1)

    return dp[amount] if dp[amount] != float("inf") else None
```

#### Simulation (trace)

`coins=[1,3,4]`, `amount=6`

- dp[0]=0
- dp[1]=1 (1)
- dp[2]=2 (1+1)
- dp[3]=1 (3)
- dp[4]=1 (4)
- dp[5]=2 (4+1)
- dp[6]=2 (3+3 or 4+1+1)

Answer: `2`

---

### 15.5 0/1 Knapsack (2D DP idea)

You can take each item at most once.

`dp[i][w]` = best value using first `i` items with capacity `w`.

```python
def knapsack_01(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        wi = weights[i - 1]
        vi = values[i - 1]
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]  # skip
            if w - wi >= 0:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - wi] + vi)  # take
    return dp[n][capacity]
```

---

### 15.6 Longest Increasing Subsequence (LIS)

Two common solutions:

#### `O(n^2)` DP (easy to explain)

```python
def lis_n2(nums):
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
```

#### `O(n log n)` (binary search / patience sorting idea)

Maintain `tails[k]` = smallest possible tail of an increasing subsequence of length `k+1`.

```python
import bisect

def lis_nlogn(nums):
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)
```

#### Simulation (trace)

`nums=[10, 9, 2, 5, 3, 7, 101, 18]`

- tails=[10]
- x=9  → tails=[9]
- x=2  → tails=[2]
- x=5  → tails=[2,5]
- x=3  → tails=[2,3]
- x=7  → tails=[2,3,7]
- x=101→ tails=[2,3,7,101]
- x=18 → tails=[2,3,7,18]

Answer length: `4`

---

### DP mindset: write the recurrence first

For DP problems, a strong workflow is:

1. Define state (`dp[i]` or `dp[r][c]`)
2. Write recurrence (how to compute from smaller states)
3. Choose base cases
4. Pick top-down or bottom-up
5. Optimize space if needed

---

## Appendix A: Interview Templates (Copy-Paste Friendly)

This appendix is intentionally “template-heavy” so you can practice writing from memory and recognize patterns quickly.

### A.1 Two pointers (opposite ends)

Common for sorted arrays, removing duplicates, and palindrome-like problems.

```python
def two_pointers_opposite(nums):
    l, r = 0, len(nums) - 1
    while l < r:
        # use nums[l] and nums[r]
        if nums[l] + nums[r] < 0:
            l += 1
        else:
            r -= 1
```

### A.2 Sliding window (fixed size)

```python
def sliding_window_fixed(nums, k):
    window = sum(nums[:k])
    best = window
    for r in range(k, len(nums)):
        window += nums[r]
        window -= nums[r - k]
        best = max(best, window)
    return best
```

### A.3 Sliding window (variable size)

Typical use: “longest subarray/substring with at most K …”

```python
def sliding_window_variable(nums, k):
    l = 0
    best = 0
    current_sum = 0

    for r, x in enumerate(nums):
        current_sum += x

        while l <= r and current_sum > k:
            current_sum -= nums[l]
            l += 1

        best = max(best, r - l + 1)
    return best
```

### A.4 Prefix sum + hash map (count subarrays)

```python
def count_subarrays_sum_k(nums, k):
    count = 0
    prefix = 0
    freq = {0: 1}

    for x in nums:
        prefix += x
        count += freq.get(prefix - k, 0)
        freq[prefix] = freq.get(prefix, 0) + 1
    return count
```

### A.5 Binary search (classic)

```python
def binary_search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target:
            return m
        if nums[m] < target:
            l = m + 1
        else:
            r = m - 1
    return -1
```

### A.6 Binary search on answer (first true)

```python
def first_true(lo, hi, pred):
    ans = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if pred(mid):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans
```

### A.7 BFS on graph (unweighted shortest path)

```python
from collections import deque

def bfs_shortest_paths(start, graph):
    dist = {start: 0}
    q = deque([start])
    while q:
        node = q.popleft()
        for nei in graph[node]:
            if nei not in dist:
                dist[nei] = dist[node] + 1
                q.append(nei)
    return dist
```

### A.8 DFS on graph (recursive)

```python
def dfs(node, graph, seen, order):
    seen.add(node)
    order.append(node)
    for nei in graph[node]:
        if nei not in seen:
            dfs(nei, graph, seen, order)
```

### A.9 Topological sort (Kahn’s algorithm)

```python
from collections import deque

def topo_sort(n, edges):
    graph = [[] for _ in range(n)]
    indeg = [0] * n
    for a, b in edges:        # a -> b
        graph[a].append(b)
        indeg[b] += 1

    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        x = q.popleft()
        order.append(x)
        for nei in graph[x]:
            indeg[nei] -= 1
            if indeg[nei] == 0:
                q.append(nei)

    return order if len(order) == n else None
```

### A.10 DSU (Union-Find)

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True
```

### A.11 Dijkstra (weighted shortest path)

```python
import heapq

def dijkstra(start, graph):
    # graph[u] = [(v, w), ...]
    dist = {start: 0}
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist.get(u, float("inf")):
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist
```

---

## Appendix B: Practice Plan (6-8 Weeks)

Suggested pacing:

- **Week 1**: Big-O + Arrays + Prefix sums
- **Week 2**: Strings + Two pointers + Sliding window
- **Week 3**: Linked lists + Stacks + Queues
- **Week 4**: Binary search + Sorting + Recursion
- **Week 5**: Hashing patterns + Heaps (top-k, k-way merge, scheduling)
- **Week 6**: Dynamic Programming (1D, 2D/grid, knapsack, LIS)
- **Week 7**: Trees (traversals, BST, tree DP)
- **Week 8**: Graphs (BFS/DFS, shortest path) + mixed mocks + weak areas

---

## Appendix C: Curated Problem Set

See also: `resources/practice_platforms.md`.

- Arrays: two-sum, best time to buy/sell stock, product except self
- Strings: valid anagram, longest substring without repeating
- Linked list: reverse list, detect cycle, merge two sorted lists
- Stack: valid parentheses, daily temperatures, next greater element
- Queue/deque: sliding window maximum
- Binary search: first/last position, search insert, rotated array
- Hashing: subarray sum equals k, top k frequent
- Heaps: kth largest element, merge k sorted lists, meeting rooms II
- DP: climbing stairs, house robber, coin change, LIS, unique paths, LCS
- Trees: level order, max depth, validate BST, LCA
- Graphs: number of islands, course schedule, shortest path in grid

