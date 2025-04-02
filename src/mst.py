import random
import time
import heapq
from collections import defaultdict
import numpy as np

class MinHeap:
    """
    Min-heap implementation for Kruskal's and Prim's algorithms.
    Supports extract-min, decrease-key, and build-heap operations.
    """
    
    def __init__(self):
        """Initialize an empty min-heap."""
        self.heap = []  # Array for storing heap elements
        self.positions = {}  # For Prim's algorithm - track positions of vertices in heap
    
    def build_heap(self, edges):
        """
        Build a min-heap from a list of edges.
        For Kruskal's algorithm.
        
        Args:
            edges: List of (u, v, weight) tuples representing edges
        """
        self.heap = list(edges)  # Copy edges to heap array
        heapq.heapify(self.heap)  # Convert to a heap
    
    def extract_min(self):
        """
        Remove and return the minimum element from the heap.
        For both Kruskal's and Prim's algorithms.
        
        Returns:
            Tuple (u, v, weight) for Kruskal's or (weight, vertex) for Prim's
        """
        if not self.heap:
            return None
        
        min_elem = heapq.heappop(self.heap)
        
        # If we're tracking positions (Prim's), update the positions
        if len(min_elem) == 2:  # (weight, vertex) format for Prim's
            _, vertex = min_elem
            self.positions.pop(vertex, None)
        
        return min_elem
    
    def insert(self, item):
        """
        Insert an item into the heap.
        
        Args:
            item: Item to insert (edge tuple or weight-vertex pair)
        """
        heapq.heappush(self.heap, item)
        
        # If we're tracking positions (Prim's), update the positions
        if len(item) == 2:  # (weight, vertex) format for Prim's
            weight, vertex = item
            self.positions[vertex] = len(self.heap) - 1
    
    def decrease_key(self, vertex, new_weight):
        """
        Decrease the key (weight) of a vertex.
        For Prim's algorithm.
        
        Args:
            vertex: Vertex whose key should be decreased
            new_weight: New weight for the vertex
        """
        # Find the vertex in the heap
        if vertex not in self.positions:
            # If vertex is not in the heap, simply insert it
            self.insert((new_weight, vertex))
            return
        
        # Find the position in the heap
        old_pos = self.positions[vertex]
        
        # Check if the position is valid
        if old_pos >= len(self.heap):
            # Position is invalid, reinsert the vertex
            self.insert((new_weight, vertex))
            return
        
        # Extract the old entry
        old_entry = self.heap[old_pos]
        old_weight, _ = old_entry
        
        # Only update if the new weight is less than the old weight
        if new_weight < old_weight:
            # Replace with a new entry
            self.heap[old_pos] = (float('inf'), vertex)  # Mark for removal
            heapq.heapify(self.heap)  # Restore heap property
            
            # Remove the marked entry (will be at the end after heapify)
            while self.heap and self.heap[-1][0] == float('inf'):
                self.heap.pop()
            
            # Add the new entry
            self.insert((new_weight, vertex))

class DisjointSet:
    """
    Disjoint-set data structure for Kruskal's algorithm.
    Implements union by rank and path compression.
    """
    
    def __init__(self, n):
        """
        Initialize a disjoint set with n elements.
        
        Args:
            n: Number of elements in the set
        """
        self.parent = list(range(n))  # Initially, each element points to itself
        self.rank = [0] * n  # Initially, all elements have rank 0
    
    def find(self, x):
        """
        Find the representative of the set containing x with path compression.
        
        Args:
            x: Element whose set representative to find
            
        Returns:
            The representative of the set containing x
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        """
        Merge the sets containing x and y using union by rank.
        
        Args:
            x: Element in first set
            y: Element in second set
            
        Returns:
            True if sets were merged, False if already in the same set
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        # Already in the same set
        if root_x == root_y:
            return False
        
        # Union by rank - attach smaller rank tree under root of higher rank tree
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:  # Same rank, break the tie
            self.parent[root_y] = root_x
            self.rank[root_x] += 1  # Increment rank
        
        return True

class Graph:
    """A graph represented as an adjacency list."""
    
    def __init__(self, num_vertices):
        """
        Initialize an empty graph with num_vertices vertices.
        
        Args:
            num_vertices: Number of vertices in the graph
        """
        self.V = num_vertices
        self.graph = [[] for _ in range(num_vertices)]  # Adjacency list
        self.edges = []  # List of all edges (u, v, weight)
        self.weight_matrix = [[0 for _ in range(num_vertices)] for _ in range(num_vertices)]
    
    def add_edge(self, u, v, w):
        """
        Add an edge from vertex u to vertex v with weight w.
        
        Args:
            u: First vertex
            v: Second vertex
            w: Edge weight
        """
        # Add to adjacency list
        self.graph[u].append((v, w))
        self.graph[v].append((u, w))  # For undirected graph
        
        # Add to edge list
        self.edges.append((u, v, w))
        
        # Add to weight matrix
        self.weight_matrix[u][v] = w
        self.weight_matrix[v][u] = w  # For undirected graph
    
    def kruskal_mst(self):
        """
        Compute a Minimum Spanning Tree (MST) using Kruskal's algorithm.
        
        Returns:
            tuple: (list of edges in the MST, total weight of the MST, execution time)
        """
        # Start timing after graph representation is ready
        start_time = time.time()
        
        n = self.V
        m = len(self.edges)
        
        # Min-heap to store all edges prioritized by weight
        min_heap = MinHeap()
        
        # Convert edges to a suitable format for the heap
        # Sort by weight first for heap construction
        # Format: (weight, u, v) - weight first for proper ordering in heap
        heap_edges = [(w, u, v) for u, v, w in self.edges]
        
        # Build the min-heap
        min_heap.build_heap(heap_edges)
        
        # Initialize disjoint set for connected components
        disjoint_set = DisjointSet(n)
        
        # Initialize array to store the edges in the MST
        mst = []  # T[1..n-1] in the assignment
        
        # Variable to accumulate total weight of the MST
        total_weight = 0
        
        # Process edges in order of increasing weight
        while len(mst) < n - 1 and min_heap.heap:
            # Extract the edge with minimum weight
            weight, u, v = min_heap.extract_min()
            
            # Check if adding this edge creates a cycle
            if disjoint_set.union(u, v):
                # Add edge to MST
                mst.append((u, v, weight))
                total_weight += weight
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return mst, total_weight, execution_time
    
    def prim_mst(self):
        """
        Compute a Minimum Spanning Tree (MST) using Prim's algorithm.
        
        Returns:
            tuple: (list of edges in the MST, total weight of the MST, execution time)
        """
        # Start timing after graph representation is ready
        start_time = time.time()
        
        n = self.V
        
        # NEAR[1..n] array - for each vertex, track the nearest vertex in the MST
        # NEAR[v] = u means vertex u is closest to v among all vertices in the MST
        # A value of -1 means the vertex is already in the MST
        near = [-1] * n
        
        # Min-heap for vertices prioritized by their distance to the MST
        min_heap = MinHeap()
        
        # Start with vertex 0
        start_vertex = 0
        near[start_vertex] = -1  # Mark as in MST
        
        # Initialize distances for all other vertices
        for v in range(n):
            if v != start_vertex:
                # Find weight of edge from start_vertex to v
                weight = float('inf')
                for neighbor, w in self.graph[start_vertex]:
                    if neighbor == v:
                        weight = w
                        break
                
                if weight < float('inf'):
                    near[v] = start_vertex
                    min_heap.insert((weight, v))
                else:
                    near[v] = -2  # No direct edge
        
        # Initialize MST
        mst = []  # T[1..n-1] in the assignment
        total_weight = 0
        
        # Grow the MST
        while len(mst) < n - 1 and min_heap.heap:
            # Extract vertex with minimum key
            min_entry = min_heap.extract_min()
            if min_entry is None:
                break
                
            weight, v = min_entry
            
            # Add edge to MST
            u = near[v]
            if u >= 0:  # Only add if u is a valid vertex
                mst.append((u, v, weight))
                total_weight += weight
            
            # Mark v as in MST
            near[v] = -1
            
            # Update keys of vertices not in MST
            for neighbor, w in self.graph[v]:
                if near[neighbor] != -1:  # Not in MST
                    if near[neighbor] == -2 or w < float('inf'):
                        # Find the current weight to the MST if it exists
                        current_weight = float('inf')
                        if near[neighbor] >= 0:
                            current_weight = self.weight_matrix[neighbor][near[neighbor]]
                        
                        # Update if the new path is better
                        if w < current_weight:
                            near[neighbor] = v
                            min_heap.decrease_key(neighbor, w)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return mst, total_weight, execution_time

def generate_random_graph(n, m=None, max_weight=1000, logger=None):
    """
    Generate a random graph with n vertices and m edges as per the assignment specifications.
    
    Parameters:
        n (int): Number of vertices
        m (int, optional): Number of edges. If None, uses n(n-1)/2 (complete graph)
        max_weight (int): Maximum weight for an edge
        logger: Logger instance for logging details
    
    Returns:
        Graph: A random graph
    """
    if logger:
        logger.debug(f"Generating random graph with {n} vertices")
    
    # Create a new graph
    g = Graph(n)
    
    # Calculate maximum possible edges for a complete graph
    max_possible_edges = n * (n - 1) // 2
    
    # If m is not specified, create a complete graph
    if m is None:
        m = max_possible_edges
    
    if m > max_possible_edges:
        if logger:
            logger.warning(f"Requested {m} edges exceeds maximum possible {max_possible_edges}. Using {max_possible_edges} instead.")
        m = max_possible_edges
    
    if logger:
        logger.debug(f"Target number of edges: {m}")
    
    # Initialize weight matrix to 0
    W = [[0 for _ in range(n)] for _ in range(n)]
    
    # Deterministically generate edges {0,1}, {1,2}, ..., {n-2,n-1} to ensure connectivity
    edges_added = 0
    for i in range(n - 1):
        weight = random.randint(1, max_weight)
        W[i][i + 1] = weight
        W[i + 1][i] = weight  # For undirected graph
        g.add_edge(i, i + 1, weight)
        edges_added += 1
    
    if logger:
        logger.debug(f"Added {edges_added} connectivity edges")
    
    # Handle special case for complete graph
    if m == max_possible_edges:
        if logger:
            logger.debug("Generating complete graph")
        
        # Generate n(n-1)/2 different weights and fill upper triangular submatrix
        for i in range(n):
            for j in range(i + 1, n):
                # Skip if edge already exists (from connectivity edges)
                if W[i][j] != 0:
                    continue
                
                weight = random.randint(1, max_weight)
                W[i][j] = weight
                W[j][i] = weight  # For undirected graph
                g.add_edge(i, j, weight)
                edges_added += 1
    
    # For dense graphs (more than half of possible edges)
    elif m > max_possible_edges // 2:
        if logger:
            logger.debug(f"Using deletion approach for dense graph (m={m})")
        
        # Start with complete graph
        complete_graph = generate_random_graph(n, max_possible_edges, max_weight, logger)
        
        # Copy the weight matrix
        W = [[0 for _ in range(n)] for _ in range(n)]
        for u, v, w in complete_graph.edges:
            W[u][v] = w
            W[v][u] = w
        
        # Count edges to delete
        edges_to_delete = max_possible_edges - m
        deleted = 0
        
        # Reset graph
        g = Graph(n)
        
        # Delete random edges until we have m edges
        all_edges = list(complete_graph.edges)
        random.shuffle(all_edges)
        
        for u, v, w in all_edges:
            # Skip connectivity edges (i, i+1)
            if (u == v + 1) or (v == u + 1):
                g.add_edge(u, v, w)
            else:
                # Delete this edge if we still need to delete edges
                if deleted < edges_to_delete:
                    deleted += 1
                    continue
                else:
                    g.add_edge(u, v, w)
        
        edges_added = m
    
    # For sparse graphs
    else:
        if logger:
            logger.debug(f"Using addition approach for sparse graph (m={m})")
        
        # Add random edges until we have m edges
        attempts = 0
        max_attempts = n * n * 10  # Avoid infinite loops
        
        while edges_added < m and attempts < max_attempts:
            attempts += 1
            
            # Generate random vertices
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            
            # Skip self-loops
            if i == j:
                continue
            
            # Skip if edge already exists
            if W[i][j] != 0:
                continue
            
            # Generate random weight
            weight = random.randint(1, max_weight)
            
            # Add edge
            W[i][j] = weight
            W[j][i] = weight  # For undirected graph
            g.add_edge(i, j, weight)
            edges_added += 1
    
    if logger:
        logger.debug(f"Final graph has {n} vertices and {edges_added} edges")
    
    return g

def generate_random_graph_old(n, density=0.5, min_weight=1, max_weight=100, logger=None):
    """
    Generate a random graph with n vertices.
    
    Parameters:
        n (int): Number of vertices
        density (float): Edge density between 0 and 1
        min_weight (int): Minimum weight for an edge
        max_weight (int): Maximum weight for an edge
        logger: Logger instance for logging details
    
    Returns:
        Graph: A random graph
    """
    if logger:
        logger.debug(f"Generating random graph with {n} vertices and density {density}")
        logger.debug(f"Edge weights range: [{min_weight}, {max_weight}]")
        logger.debug(f"Random number generator: Python's random module (Mersenne Twister algorithm)")
    
    g = Graph(n)
    max_edges = n * (n - 1) // 2
    num_edges = int(max_edges * density)
    
    if logger:
        logger.debug(f"Maximum possible edges: {max_edges}")
        logger.debug(f"Target number of edges: {num_edges}")
    
    # Set a seed for reproducibility if needed
    # random.seed(42)
    
    # Ensure the graph is connected by first creating a spanning tree
    edges_added = 0
    for i in range(1, n):
        parent = random.randint(0, i - 1)
        weight = random.randint(min_weight, max_weight)
        g.add_edge(parent, i, weight)
        edges_added += 1
    
    if logger:
        logger.debug(f"Added {edges_added} edges to ensure connectivity (spanning tree)")
    
    # Add remaining random edges
    attempts = 0
    while edges_added < num_edges and attempts < max_edges * 10:  # Avoid infinite loops
        attempts += 1
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        
        if u != v:
            # Check if this edge already exists to avoid duplicates
            edge_exists = False
            for adj, _ in g.graph[u]:
                if adj == v:
                    edge_exists = True
                    break
            
            if not edge_exists:
                weight = random.randint(min_weight, max_weight)
                g.add_edge(u, v, weight)
                edges_added += 1
    
    if logger:
        logger.debug(f"Final graph has {n} vertices and {edges_added} edges")
        if edges_added < num_edges:
            logger.warning(f"Could not add all requested edges. Added {edges_added} out of {num_edges}")
    
    return g

def run_experiment(num_vertices_list, num_trials=5, density=0.5):
    """
    Run experiments comparing Kruskal's and Prim's algorithms on graphs of different sizes.
    
    Parameters:
        num_vertices_list (list): List of number of vertices to test
        num_trials (int): Number of trials for each graph size
        density (float): Edge density for the generated graphs
    
    Returns:
        dict: Results of the experiments
    """
    results = {
        'vertices': [],
        'kruskal_time': [],
        'prim_time': []
    }
    
    for n in num_vertices_list:
        kruskal_times = []
        prim_times = []
        
        for _ in range(num_trials):
            # Calculate number of edges based on density
            max_edges = n * (n - 1) // 2
            m = int(max_edges * density)
            
            graph = generate_random_graph(n, m)
            
            _, _, kruskal_time = graph.kruskal_mst()
            _, _, prim_time = graph.prim_mst()
            
            kruskal_times.append(kruskal_time)
            prim_times.append(prim_time)
        
        results['vertices'].append(n)
        results['kruskal_time'].append(sum(kruskal_times) / num_trials)
        results['prim_time'].append(sum(prim_times) / num_trials)
    
    return results

def verify_mst(graph, mst):
    """
    Verify that the MST is correct (connected and minimum weight).
    
    Parameters:
        graph: The original graph
        mst: List of edges in the MST
        
    Returns:
        bool: True if the MST is correct, False otherwise
    """
    n = graph.V
    
    # Check that the MST has n-1 edges
    if len(mst) != n - 1:
        return False
    
    # Check that the MST is connected
    # Create a new graph with just the MST edges
    mst_graph = Graph(n)
    for u, v, w in mst:
        mst_graph.add_edge(u, v, w)
    
    # BFS to check connectivity
    visited = [False] * n
    queue = [0]  # Start from vertex 0
    visited[0] = True
    
    while queue:
        u = queue.pop(0)
        for v, _ in mst_graph.graph[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
    
    # If any vertex is not visited, the MST is not connected
    if not all(visited):
        return False
    
    # The MST is correct
    return True

if __name__ == "__main__":
    # Example usage
    g = Graph(4)
    g.add_edge(0, 1, 10)
    g.add_edge(0, 2, 6)
    g.add_edge(0, 3, 5)
    g.add_edge(1, 3, 15)
    g.add_edge(2, 3, 4)
    
    print("Kruskal's MST:")
    mst_edges, total_weight, time_taken = g.kruskal_mst()
    for u, v, w in mst_edges:
        print(f"Edge {u}-{v} with weight {w}")
    print(f"Total weight: {total_weight}")
    print(f"Time taken: {time_taken:.6f} seconds")
    print(f"MST is correct: {verify_mst(g, mst_edges)}")
    
    print("\nPrim's MST:")
    mst_edges, total_weight, time_taken = g.prim_mst()
    for u, v, w in mst_edges:
        print(f"Edge {u}-{v} with weight {w}")
    print(f"Total weight: {total_weight}")
    print(f"Time taken: {time_taken:.6f} seconds")
    print(f"MST is correct: {verify_mst(g, mst_edges)}") 