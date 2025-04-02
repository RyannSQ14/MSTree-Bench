import random
import time
import heapq

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
        
        # Sort edges by weight directly instead of using MinHeap
        sorted_edges = sorted(self.edges, key=lambda x: x[2])
        
        # Initialize disjoint set for connected components
        disjoint_set = DisjointSet(n)
        
        # Initialize array to store the edges in the MST
        mst = []  # T[1..n-1] in the assignment
        
        # Variable to accumulate total weight of the MST
        total_weight = 0
        
        # Process edges in order of increasing weight
        for u, v, weight in sorted_edges:
            # Stop if we've found n-1 edges
            if len(mst) == n - 1:
                break
                
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
        
        # Use a more efficient priority queue implementation
        # Use direct array for key values and standard Python heapq
        key = [float('inf')] * n
        parent = [-1] * n
        in_mst = [False] * n
        
        # Start with vertex 0
        key[0] = 0
        
        # Use a simple heap instead of our custom implementation
        # Format: (key, vertex)
        heap = [(0, 0)]  # (weight, vertex)
        
        mst = []
        total_weight = 0
        
        while heap:
            # Extract vertex with minimum key
            k, u = heapq.heappop(heap)
            
            # Skip if already processed
            if in_mst[u]:
                continue
                
            # Mark as processed
            in_mst[u] = True
            
            # Add edge to MST if not the first vertex
            if parent[u] != -1:
                mst.append((parent[u], u, k))
                total_weight += k
            
            # Update keys of adjacent vertices
            for v, weight in self.graph[u]:
                # Only process unvisited vertices with better weight
                if not in_mst[v] and weight < key[v]:
                    # Update parent and key
                    parent[v] = u
                    key[v] = weight
                    
                    # Add to heap
                    heapq.heappush(heap, (weight, v))
        
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