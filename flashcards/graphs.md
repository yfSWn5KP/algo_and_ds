---
recall: header
---

### What is Manhattan Distance?

Also known as Taxicab Geometry, it is the distance between points when moving between cells on a grid.
 
You can move 90° North, South, East, or West.
 
This is opposed to Euclidean Geometry where the distance is the shortest path between two points at any angle.
 
<img src="https://upload.wikimedia.org/wikipedia/commons/0/08/Manhattan_distance.svg">
 
In the above graphic, the green path is Euclidean distance; the other paths are Manhattan distance.


### Complete this Dijkstra Shortest Path method:
 
```python
def dijkstra(adj_list_costs, src, dst):
  priority_queue = [ (0, -1, src) ]  # cost, steps, node
  while priority_queue:
    'COMPLETE ME'
```

```python
def dijkstra(adj_list_costs, src, dst):
  priority_queue = [ (0, -1, src) ]  # cost, steps, node
  while priority_queue:
    cur_cost, steps, node = heapq.heappop(priority_queue)
    if node == dst:
      return cur_cost
    for neighbor, cost in adj_list_costs[node]:
      heapq.heappush(priority_queue, (cur_cost + cost, steps + 1, neighbor))
```
 
[further reading](https://leetcode.com/explore/learn/card/graph/622/single-source-shortest-path-algorithm/3862/)


### What is the Time & Space Complexity of Dijkstra Shortest Path?

Time is `O( E⋅log(V) )`
 
Space is `O(V^2)`


### Complete this Bellman-Ford Propagation Cost method:
 
```python
def bellman_ford(nodes, weighted_edges, start_node):
  costs = 'COMPLETE ME'
  for _ in range(len(nodes) - 1):
    'COMPLETE ME'
  max_cost = max(costs.values())
  'COMPLETE ME'
```

```python
def bellman_ford(nodes, weighted_edges, start_node):
  costs = {node: float('inf') for node in nodes}
  costs[start_node] = 0
  for _ in range(len(nodes) - 1):
    for src, dst, cost in weighted_edges:
      costs[dst] = min(costs[dst], costs[src] + cost)
  max_cost = max(costs.values())
  if max_cost == float('inf'):
    raise ImpossibleToPropagate
  return max_cost
```
 
[further reading](https://leetcode.com/explore/learn/card/graph/622/single-source-shortest-path-algorithm/3864/)


### What is the Time & Space Complexity of Bellman-Ford Propagation Cost?

Time is `O(E⋅V)`
 
Space is `O(V)`


### How do you use Bellman-Ford Propagation Cost to detect a Negative Cycle?

Run through Bellman-Ford twice.
 
If the cost shrinks between runs, then there is a negative cycle.


### How does SPFA improve on Bellman-Ford?

SPFA relaxes edges proximal to the start, i.e. optimizes for only calculating costs that have evolved from the starting node.


### Complete this Shortest Path Fastest Algo (SPFA) Propagation Cost method:
 
```python
def spfa(nodes, weighted_edges, start_node):
  costs = 'COMPLETE ME'
  edge_weights = 'COMPLETE ME'
  queue = deque([start_node])
  while queue:
    'COMPLETE ME'
  max_cost = max(costs.values())
  'COMPLETE ME'
```

```python
def spfa(nodes, weighted_edges, start_node):
  costs = {node: float('inf') for node in nodes}
  costs[start_node] = 0
  edge_weights = defaultdict(dict)
  for src, dst, cost in weighted_edges:
    edge_weights[src][dst] = cost
  queue = deque([start_node])
  while queue:
    src = queue.popleft()
    for dst, edge_cost in edge_weights[src].items():
      if costs[src] + edge_cost < costs[dst]:
        costs[dst] = costs[src] + edge_cost
        queue.append(dst)
  max_cost = max(costs.values())
  if max_cost == float('inf'):
    raise ImpossibleToPropagate
  return max_cost
```
 
[further reading](https://leetcode.com/explore/learn/card/graph/622/single-source-shortest-path-algorithm/3865/)


### What is the Time & Space Complexity of SPFA Propagation Cost?

Time is average of `O(E)` but with worst case of `O(E⋅V)`
 
Space is `O(E + V)`


### Complete this Floyd-Warshall Propagation Cost method:
 
```python
def floyd_warshall(nodes, weighted_edges, start_node):
  costs = 'COMPLETE ME'
  for stopover in nodes:
    'COMPLETE ME'
  max_cost = max(costs[start_node])
  'COMPLETE ME'
```

```python
def floyd_warshall(nodes, weighted_edges, start_node):
  costs = defaultdict(dict)
  for src in nodes:
    for dst in nodes:
      costs[src][dst] = float('inf')
  for src in nodes:
    costs[src][src] = 0
  for src, dst, cost in weighted_edges:
    costs[src][dst] = cost
  for stopover in nodes:
    for src in nodes:
      for dst in nodes:
        costs[src][dst] = min(
          costs[src][dst],
          costs[src][stopover] + costs[stopover][dst]
        )
  max_cost = max(costs[start_node].values())
  if max_cost == float('inf'):
    raise ImpossibleToPropagate
  return max_cost
```


### What is the Time & Space Complexity of Floyd-Warshall Propagation Cost?

Time is `O(V^3)`
 
Space is `O(V^2)`


### What are the Pros/Cons of Floyd-Warshall vs. SPFA or Bellman-Ford for calculating Propagation Cost?

Pro: F-W can be distributed/multithreaded
 
Con: F-W has worse overall time & space complexity


### Complete this Tarjan's SCC (Strongly Connected Components) method:
 
```python
def tarjan(adj_dict: Dict['Node', List['Node']]):
  min_ranks = {}
  def iter_tarjan_dfs(cur_rank, node, parent):
    'COMPLETE ME'
  
  node_a = next(k for k in adj_dict)
  return list(iter_tarjan_dfs(0, node_a, None))
```

```python
def tarjan(adj_dict: Dict['Node', List['Node']]):
  min_ranks = {}
  def iter_tarjan_dfs(cur_rank, node, parent):
    min_ranks[node] = cur_rank
    cur_rank += 1
    for neighbor in adj_dict[node]:
      if neighbor == parent:
        continue
      if neighbor not in min_ranks:
        yield from iter_tarjan_dfs(cur_rank, neighbor, node)
      if min_ranks[neighbor] == cur_rank:
        yield (node, neighbor)
      min_ranks[node] = min(min_ranks[node], min_ranks[neighbor])
  
  node_a = next(k for k in adj_dict)
  return list(iter_tarjan_dfs(0, node_a, None))
```


### What is the Time & Space Complexity of Tarjan's SCC (Strongly Connected Components)?

Time is `O(E + V)`
 
Space is `O(V)`


### What is the formula for Row-Order Traversal of a 2d Graph?

`idx = row_i * num_rows + col_i`


### What is the formula for Column-Order Traversal of a 2d Graph?

`idx = col_i * num_cols + row_i`


### Complete this Trie Data Structure:
 
```python
class Trie:
  def __init__(self):
    self.root = 'COMPLETE ME'
    self.terminal_val = ''
  
  def insert(self, word):
    'COMPLETE ME'
```

```python
class Trie:
  def __init__(self):
    node_factory = lambda: defaultdict(node_factory)
    self.root = node_factory()
    self.terminal_val = ''
  
  def insert(self, word):
    node = self.root
    for char in word:
      node = node[char]
    node[self.terminal_val] = None
```
 
[further reading](https://www.interviewcake.com/concept/python3/trie)


### What is the Time & Space Complexity of a Trie Data Structure?

Time is `O(k)`, where `k` == `num_chars` in an insert/lookup word
 
Space is `O(n⋅k)`, where `n` == `num_words` in the trie


### Complete this Topological Ordering method:
 
```python
def topo_order(nodes, dependencies):
  dependency_cnts = 'COMPLETE ME'
  out_degrees = 'COMPLETE ME'
  
  dq = deque()
  for node in nodes:
    'COMPLETE ME'
  
  topo_res = []
  while dq:
    'COMPLETE ME'
  
  if dependency_cnts:
    raise UnmetDependencies
  return topo_res
```

```python
def topo_order(nodes, dependencies):
  dependency_cnts = defaultdict(int)
  out_degrees = defaultdict(set)
  for in_node, out_node in dependencies:
    dependency_cnts[in_node] += 1
    out_degrees[out_node].add(in_node)
  
  dq = deque()
  for node in nodes:
    if node not in dependency_cnts:
      dq.append(node)
  
  topo_res = []
  while dq:
    node = dq.popleft()
    topo_res.append(node)
    for in_node in out_degrees[node]:
      dependency_cnts[in_node] -= 1
      if not dependency_cnts[in_node]:
        del dependency_cnts[in_node]
        dq.append(in_node)
  
  if dependency_cnts:
    raise UnmetDependencies
  return topo_res
```


### What is the Time & Space Complexity of Topological Ordering?

Time is `O(E + V)`
 
Space is `O(E + V)`


### Complete this Kruskal's Minimum Spanning Tree:
 
```python
def kruskal(weighted_edges):
  weighted_edges = 'COMPLETE_ME'
  dsu = DSU()
  for cost, node_a, node_b in weighted_edges:
    'COMPLETE ME'
```

```python
def kruskal(weighted_edges):
  weighted_edges = sorted(weighted_edges)
  dsu = DSU()
  for cost, node_a, node_b in weighted_edges:
    if dsu.union(node_a, node_b):
      yield cost, node_a, node_b
```


### What is the Time & Space Complexity of Kruskal's Minimum Spanning Tree?

Time is `O(E⋅log(V))`
 
Space is `O(E)`


### Complete this Prim's Minimum Spanning Tree:
 
```python
def prims(adj_weights: Dict[Node, List[List[int, Node]]]):
  priority_weighted_nodes = 'COMPLETE ME'
  visited = set()
  while len(visited) != len(adj_weights):
    node_cost, node = heapq.heappop(priority_weighted_nodes)
    if node in visited:
      continue
    visited.add(node)
    
    'COMPLETE ME'
```

```python
def prims(adj_weights):
  node = next(k for k in adj_weights)
  priority_weighted_nodes = [(0, node)]
  visited = set()
  while len(visited) != len(adj_weights):
    node_cost, node = heapq.heappop(priority_weighted_nodes)
    if node in visited:
      continue
    visited.add(node)
    
    yield node_cost, node
    for neighbor_cost, neighbor in adj_weights[node]:
      if neighbor not in visited:
        heapq.heappush(priority_weighted_nodes, (neighbor_cost, neighbor))
```


### What is the Time & Space Complexity of Prim's Minimum Spanning Tree?

Time is `O(V^2)`
 
Space is `O(E)`


### What situations work with Dijkstra's Algo?

Finding the shortest path between 2 points in a:
* Directed or Undirected graph
* with edge weight/cost >= 0


### What situations work with Bellman-Ford, SPFA, and Floyd-Warshall?

Calculating the propagation cost from a starting node to all other nodes in a:
* Directed graph
* with positive and/or negative edge weight/cost


### What situations work with Kruskal's Minimum Spanning Tree?

Finding the propagation cost across the whole network (starting at any node) in an:
* Undirected graph
* that is either Connected or Disconnected


### What situations work with Prim's Minimum Spanning Tree?

Finding the propagation cost across the whole network (starting at any node) in an:
* Undirected graph
* that is totally Connected
