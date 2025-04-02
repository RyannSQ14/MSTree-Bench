import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import os
import math
from mst import Graph, generate_random_graph

def run_experiments_by_size(logger):
    """Run experiments varying the number of vertices."""
    logger.info("Running experiments with varying graph sizes...")
    
    # Vertex sizes to test
    vertex_sizes = [10, 50, 100, 200, 300, 400, 500, 750, 1000]
    # Run experiments
    results = run_experiment(vertex_sizes, num_trials=5, density=0.5, logger=logger)
    
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Save results to CSV
    with open("reports/size_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Vertices", "Kruskal Time (s)", "Prim Time (s)"])
        for i in range(len(results['vertices'])):
            writer.writerow([
                results['vertices'][i],
                results['kruskal_time'][i],
                results['prim_time'][i]
            ])
    
    logger.info("Generating plots for graph size experiments...")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results['vertices'], results['kruskal_time'], 'o-', label="Kruskal's Algorithm")
    plt.plot(results['vertices'], results['prim_time'], 's-', label="Prim's Algorithm")
    plt.xlabel('Number of Vertices')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('MST Algorithm Performance vs Graph Size')
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/size_comparison.png", dpi=300, bbox_inches='tight')
    
    # Log-log plot to better visualize asymptotic behavior
    plt.figure(figsize=(10, 6))
    plt.loglog(results['vertices'], results['kruskal_time'], 'o-', label="Kruskal's Algorithm")
    plt.loglog(results['vertices'], results['prim_time'], 's-', label="Prim's Algorithm")
    plt.xlabel('Number of Vertices (log scale)')
    plt.ylabel('Average Execution Time (seconds, log scale)')
    plt.title('MST Algorithm Performance vs Graph Size (Log-Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/size_comparison_log.png", dpi=300, bbox_inches='tight')
    
    logger.info("Size experiments completed.")
    return results

def run_experiments_by_density(logger):
    """Run experiments varying the density of the graph."""
    logger.info("Running experiments with varying graph densities...")
    
    # Fixed number of vertices
    n = 500
    # Densities to test (from sparse to dense)
    densities = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = {
        'densities': densities,
        'kruskal_time': [],
        'prim_time': []
    }
    
    num_trials = 5
    max_edges = n * (n - 1) // 2
    
    for density in densities:
        logger.info(f"Testing density: {density}")
        kruskal_times = []
        prim_times = []
        
        # Calculate number of edges based on density
        m = int(max_edges * density)
        
        for trial in range(num_trials):
            logger.debug(f"  Trial {trial+1}/{num_trials}")
            graph = generate_random_graph(n, m=m, logger=logger)
            
            _, _, kruskal_time = graph.kruskal_mst()
            _, _, prim_time = graph.prim_mst()
            
            kruskal_times.append(kruskal_time)
            prim_times.append(prim_time)
        
        avg_kruskal = sum(kruskal_times) / num_trials
        avg_prim = sum(prim_times) / num_trials
        logger.debug(f"  Average Kruskal: {avg_kruskal:.6f}s, Average Prim: {avg_prim:.6f}s")
        
        results['kruskal_time'].append(avg_kruskal)
        results['prim_time'].append(avg_prim)
    
    # Save results to CSV
    with open("reports/density_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Density", "Kruskal Time (s)", "Prim Time (s)"])
        for i in range(len(results['densities'])):
            writer.writerow([
                results['densities'][i],
                results['kruskal_time'][i],
                results['prim_time'][i]
            ])
    
    logger.info("Generating plots for graph density experiments...")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results['densities'], results['kruskal_time'], 'o-', label="Kruskal's Algorithm")
    plt.plot(results['densities'], results['prim_time'], 's-', label="Prim's Algorithm")
    plt.xlabel('Graph Density')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('MST Algorithm Performance vs Graph Density (n=500)')
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/density_comparison.png", dpi=300, bbox_inches='tight')
    
    logger.info("Density experiments completed.")
    return results

def run_comparison_plot(logger):
    """
    Run experiments for the comparison plot as specified in the assignment.
    
    For each n in [16, 32, 64, 128, 256], test edge counts m = n, 2n, 4n, 8n, 16n, ..., n(n-1)/2
    Plot with density (m/n) as x-axis with logarithmic scale.
    """
    logger.info("Running experiments for comparison plot...")
    
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Vertex counts to test
    vertex_counts = [16, 32, 64, 128, 256, 512]
    num_trials = 5
    
    # Store all results
    results = {
        'n': [],          # Vertex count
        'density': [],    # Density (m/n)
        'kruskal': [],    # Kruskal's algorithm time
        'prim': []        # Prim's algorithm time
    }
    
    # For each vertex count
    for n in vertex_counts:
        logger.info(f"Testing graphs with {n} vertices...")
        
        # Calculate max edges for complete graph
        max_edges = n * (n - 1) // 2
        
        # Calculate edge counts to test
        edge_counts = []
        m = n  # Start with m = n edges
        while m <= max_edges:
            edge_counts.append(m)
            m *= 2  # Double the edge count
        
        # Add the maximum edge count if not already included
        if max_edges not in edge_counts:
            edge_counts.append(max_edges)
        
        # For each edge count
        for m in edge_counts:
            # Calculate density as m/n where m is edge count
            density_m_n = m / n
            
            logger.info(f"  Testing with {m} edges (density m/n = {density_m_n:.2f})")
            
            kruskal_times = []
            prim_times = []
            
            # Run multiple trials
            for trial in range(num_trials):
                logger.debug(f"    Trial {trial+1}/{num_trials}")
                
                # Generate graph with specified vertex count and edge count
                graph = generate_random_graph(n, m=m, logger=logger)
                
                # Time Kruskal's algorithm
                _, _, kruskal_time = graph.kruskal_mst()
                kruskal_times.append(kruskal_time)
                
                # Time Prim's algorithm
                _, _, prim_time = graph.prim_mst()
                prim_times.append(prim_time)
            
            # Calculate average times
            avg_kruskal = sum(kruskal_times) / num_trials
            avg_prim = sum(prim_times) / num_trials
            
            logger.debug(f"    Average times - Kruskal: {avg_kruskal:.6f}s, Prim: {avg_prim:.6f}s")
            
            # Store results
            results['n'].append(n)
            results['density'].append(density_m_n)
            results['kruskal'].append(avg_kruskal)
            results['prim'].append(avg_prim)
    
    # Save results to CSV
    with open("reports/comparison_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Vertices (n)", "Density (m/n)", "Kruskal Time (s)", "Prim Time (s)"])
        for i in range(len(results['n'])):
            writer.writerow([
                results['n'][i],
                results['density'][i],
                results['kruskal'][i],
                results['prim'][i]
            ])
    
    # Generate the comparison plot
    logger.info("Generating comparison plot...")
    plt.figure(figsize=(12, 8))
    
    # Define markers and colors
    markers = ['o', 's', 'd', '^', 'v', 'p']
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    
    # For each vertex count, plot a line for each algorithm
    for i, n in enumerate(vertex_counts):
        # Filter data for this vertex count
        indices = [j for j, v in enumerate(results['n']) if v == n]
        
        if indices:
            # Extract data for this vertex count
            densities = [results['density'][j] for j in indices]
            kruskal_times = [results['kruskal'][j] for j in indices]
            prim_times = [results['prim'][j] for j in indices]
            
            # Plot data for Kruskal's algorithm
            plt.plot(densities, kruskal_times, 
                    marker=markers[i % len(markers)], 
                    color=colors[i % len(colors)], 
                    linestyle='-', 
                    label=f'Kruskal (n={n})')
            
            # Plot data for Prim's algorithm
            plt.plot(densities, prim_times, 
                    marker=markers[i % len(markers)], 
                    color=colors[i % len(colors)], 
                    linestyle='--', 
                    label=f'Prim (n={n})')
    
    # Set logarithmic scale for x-axis
    plt.xscale('log', base=2)
    
    # Add labels and title
    plt.xlabel('Density (m/n) - Log Scale')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('MST Algorithm Performance vs Graph Density')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig("reports/comparison_plot.png", dpi=300, bbox_inches='tight')
    
    logger.info("Comparison plot experiments completed.")
    return results

def find_crossover_point(logger):
    """
    Find the approximate graph size where Prim's algorithm becomes faster than Kruskal's.
    Uses bisection to narrow down the range.
    """
    logger.info("Finding crossover point where Prim's algorithm becomes faster than Kruskal's...")
    
    lower_bound = 10   # Assume Kruskal is faster for small graphs
    upper_bound = 2000  # Assume Prim is faster for large graphs
    num_trials = 5
    precision = 20   # Stop when range is this small
    
    crossover_points = []
    
    # Do several runs to get an average
    for run in range(3):
        logger.info(f"Crossover run {run+1}/3")
        current_lower = lower_bound
        current_upper = upper_bound
        
        while current_upper - current_lower > precision:
            mid_point = (current_lower + current_upper) // 2
            logger.info(f"  Testing size {mid_point}...")
            
            kruskal_times = []
            prim_times = []
            
            for trial in range(num_trials):
                logger.debug(f"    Trial {trial+1}/{num_trials}")
                
                # Calculate number of edges based on 50% density
                max_edges = mid_point * (mid_point - 1) // 2
                m = max_edges // 2
                
                graph = generate_random_graph(mid_point, m=m, logger=logger)
                
                _, _, kruskal_time = graph.kruskal_mst()
                _, _, prim_time = graph.prim_mst()
                
                kruskal_times.append(kruskal_time)
                prim_times.append(prim_time)
            
            avg_kruskal = sum(kruskal_times) / num_trials
            avg_prim = sum(prim_times) / num_trials
            
            logger.info(f"    Average times - Kruskal: {avg_kruskal:.6f}s, Prim: {avg_prim:.6f}s")
            
            if avg_kruskal < avg_prim:
                # Kruskal is still faster, so crossover point is higher
                current_lower = mid_point
                logger.debug(f"    Kruskal faster, increasing lower bound to {current_lower}")
            else:
                # Prim is faster, so crossover point is lower
                current_upper = mid_point
                logger.debug(f"    Prim faster, decreasing upper bound to {current_upper}")
        
        crossover_point = (current_lower + current_upper) // 2
        crossover_points.append(crossover_point)
        logger.info(f"Run {run+1}: Crossover point approximately at {crossover_point} vertices")
    
    avg_crossover = sum(crossover_points) / len(crossover_points)
    logger.info(f"Average crossover point: {avg_crossover} vertices")
    
    # Save results
    with open("reports/crossover_point.txt", "w") as f:
        f.write(f"Individual crossover points: {crossover_points}\n")
        f.write(f"Average crossover point: {avg_crossover} vertices\n")
    
    return avg_crossover

def theoretical_analysis(logger):
    """Generate plots based on theoretical time complexities for comparison."""
    logger.info("Generating theoretical analysis plots...")
    
    # Theoretical complexities
    # Kruskal: O(E log E) where E is O(V²) for dense graphs
    # Prim with binary heap: O(E log V) where E is O(V²) for dense graphs
    
    vertices = np.arange(10, 1001, 10)
    
    # For a dense graph (E ~ V²)
    kruskal_theory = [v**2 * np.log(v**2) for v in vertices]
    prim_theory = [v**2 * np.log(v) for v in vertices]
    
    # Normalize to make them comparable
    max_val = max(max(kruskal_theory), max(prim_theory))
    kruskal_theory = [t / max_val for t in kruskal_theory]
    prim_theory = [t / max_val for t in prim_theory]
    
    plt.figure(figsize=(10, 6))
    plt.plot(vertices, kruskal_theory, 'b-', label="Kruskal's Algorithm (Theory: O(E log E))")
    plt.plot(vertices, prim_theory, 'r-', label="Prim's Algorithm (Theory: O(E log V))")
    plt.xlabel('Number of Vertices')
    plt.ylabel('Normalized Theoretical Time')
    plt.title('Theoretical Time Complexity Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/theoretical_comparison.png", dpi=300, bbox_inches='tight')
    
    # Log-log plot
    plt.figure(figsize=(10, 6))
    plt.loglog(vertices, kruskal_theory, 'b-', label="Kruskal's Algorithm (Theory: O(E log E))")
    plt.loglog(vertices, prim_theory, 'r-', label="Prim's Algorithm (Theory: O(E log V))")
    plt.xlabel('Number of Vertices (log scale)')
    plt.ylabel('Normalized Theoretical Time (log scale)')
    plt.title('Theoretical Time Complexity Comparison (Log-Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/theoretical_comparison_log.png", dpi=300, bbox_inches='tight')
    
    logger.info("Theoretical analysis completed.")

def run_experiment(num_vertices_list, num_trials=5, density=0.5, logger=None):
    """
    Run experiments comparing Kruskal's and Prim's algorithms on graphs of different sizes.
    
    Parameters:
        num_vertices_list (list): List of number of vertices to test
        num_trials (int): Number of trials for each graph size
        density (float): Edge density for the generated graphs
        logger: Logger instance for logging progress
    
    Returns:
        dict: Results of the experiments
    """
    results = {
        'vertices': [],
        'kruskal_time': [],
        'prim_time': []
    }
    
    for n in num_vertices_list:
        if logger:
            logger.info(f"Testing graph with {n} vertices...")
        
        kruskal_times = []
        prim_times = []
        
        for trial in range(num_trials):
            if logger:
                logger.debug(f"  Trial {trial+1}/{num_trials}")
            
            # Calculate number of edges based on density
            max_edges = n * (n - 1) // 2
            m = int(max_edges * density)
            
            graph = generate_random_graph(n, m=m, logger=logger)
            
            _, _, kruskal_time = graph.kruskal_mst()
            _, _, prim_time = graph.prim_mst()
            
            kruskal_times.append(kruskal_time)
            prim_times.append(prim_time)
        
        avg_kruskal = sum(kruskal_times) / num_trials
        avg_prim = sum(prim_times) / num_trials
        
        if logger:
            logger.debug(f"  Average Kruskal: {avg_kruskal:.6f}s, Average Prim: {avg_prim:.6f}s")
        
        results['vertices'].append(n)
        results['kruskal_time'].append(avg_kruskal)
        results['prim_time'].append(avg_prim)
    
    return results

def run_all_experiments(logger=None):
    """Run all experiments and generate a report."""
    logger.info("Starting all experiments...")
    
    logger.info("\nRunning experiments with varying graph sizes...")
    size_results = run_experiments_by_size(logger)
    
    logger.info("\nRunning experiments with varying graph densities...")
    density_results = run_experiments_by_density(logger)
    
    logger.info("\nRunning comparison plot experiments...")
    comparison_results = run_comparison_plot(logger)
    
    logger.info("\nFinding crossover point...")
    crossover = find_crossover_point(logger)
    
    logger.info("\nGenerating theoretical analysis plots...")
    theoretical_analysis(logger)
    
    logger.info("\nAll experiments completed. Results saved to 'reports' directory.")

if __name__ == "__main__":
    from logger import setup_logger
    logger = setup_logger()
    run_all_experiments(logger) 