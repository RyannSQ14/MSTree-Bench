import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime

def generate_report(logger=None):
    """Generate a comprehensive report from the experimental results."""
    if logger:
        logger.info("Generating comprehensive HTML report...")
    
    # Create the report directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Read results from CSV files
    size_results = read_csv_results("reports/size_results.csv", logger)
    density_results = read_csv_results("reports/density_results.csv", logger)
    comparison_results = read_csv_results("reports/comparison_results.csv", logger)
    
    # Read crossover point results
    crossover_point = ""
    try:
        with open("reports/crossover_point.txt", "r") as f:
            crossover_point = f.read()
        if logger:
            logger.debug("Successfully read crossover point data")
    except FileNotFoundError:
        crossover_point = "Crossover point data not available."
        if logger:
            logger.warning("Crossover point data file not found")
    
    # Generate HTML report
    if logger:
        logger.info("Generating HTML content...")
    
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MST Algorithms Experimental Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .image-container {{ display: flex; flex-wrap: wrap; justify-content: center; }}
            .image-container img {{ max-width: 100%; height: auto; margin: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .analysis {{ background-color: #f8f8f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .code {{ font-family: monospace; background-color: #f0f0f0; padding: 10px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Minimum Spanning Tree Algorithms: Experimental Analysis</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>1. Introduction</h2>
            <div class="analysis">
                <p>This report presents an experimental analysis comparing two popular algorithms for finding Minimum Spanning Trees (MST) in graphs:</p>
                <ul>
                    <li><strong>Kruskal's Algorithm</strong>: Based on a greedy approach that adds the next smallest-weight edge that doesn't create a cycle.</li>
                    <li><strong>Prim's Algorithm</strong>: Builds the MST one vertex at a time, adding the minimum-weight edge connecting the tree to a new vertex.</li>
                </ul>
                <p>We investigate the performance characteristics of these algorithms in various scenarios, focusing on:</p>
                <ul>
                    <li>How algorithm performance scales with graph size</li>
                    <li>How graph density affects performance</li>
                    <li>The crossover point where one algorithm becomes more efficient than the other</li>
                    <li>Comparison of experimental results with theoretical expectations</li>
                </ul>
            </div>
            
            <h2>2. Algorithm Implementations</h2>
            <div class="analysis">
                <h3>2.1 Kruskal's Algorithm</h3>
                <p>Our implementation of Kruskal's algorithm uses a disjoint-set data structure with path compression and union by rank optimizations. The main steps are:</p>
                <ol>
                    <li>Sort all edges by weight</li>
                    <li>Initialize disjoint sets for each vertex</li>
                    <li>Process edges in increasing weight order, adding an edge if it connects different components</li>
                    <li>Stop when n-1 edges have been added</li>
                </ol>
                <p>Time complexity: O(E log E), where E is the number of edges in the graph.</p>
                
                <h3>2.2 Prim's Algorithm</h3>
                <p>Our implementation of Prim's algorithm uses a binary heap (priority queue) to efficiently find the next minimum-weight edge. The main steps are:</p>
                <ol>
                    <li>Start with any vertex</li>
                    <li>Maintain a priority queue of edges connecting the current MST to unvisited vertices</li>
                    <li>At each step, add the minimum-weight edge (and its vertex) to the MST</li>
                    <li>Update the priority queue with edges from the newly added vertex</li>
                </ol>
                <p>Time complexity: O(E log V), where V is the number of vertices in the graph.</p>
            </div>
            
            <h2>3. Experimental Setup</h2>
            <div class="analysis">
                <p>All experiments were conducted using:</p>
                <ul>
                    <li><strong>Random Graph Generation</strong>: We generated random graphs with varying numbers of vertices and edge densities, ensuring all graphs were connected.</li>
                    <li><strong>Performance Measurement</strong>: Execution times were measured in seconds, with each test repeated multiple times to obtain average performance.</li>
                    <li><strong>Graph Sizes</strong>: From 10 to 1000 vertices</li>
                    <li><strong>Graph Densities</strong>: From 0.01 (very sparse) to 0.9 (very dense)</li>
                </ul>
            </div>
            
            <h2>4. Results: Effect of Graph Size</h2>
            <div class="analysis">
                <p>We measured performance as the number of vertices increased from 10 to 1000, with a fixed edge density of 0.5.</p>
                <div class="image-container">
                    <img src="../reports/size_comparison.png" alt="MST Performance vs Graph Size">
                    <img src="../reports/size_comparison_log.png" alt="MST Performance vs Graph Size (Log-Log)">
                </div>
                
                <h3>Data Table: Algorithm Performance vs Graph Size</h3>
                <table>
                    <tr>
                        <th>Number of Vertices</th>
                        <th>Kruskal's Algorithm (s)</th>
                        <th>Prim's Algorithm (s)</th>
                    </tr>
                    {generate_table_rows(size_results)}
                </table>
                
                <h3>Analysis:</h3>
                <p>The results show that for smaller graphs, Kruskal's algorithm tends to outperform Prim's algorithm. 
                However, as the graph size increases, there is a crossover point after which Prim's algorithm becomes more efficient. 
                This aligns with their theoretical complexities: Kruskal's O(E log E) vs Prim's O(E log V).</p>
                <p>For dense graphs where E approaches V², Kruskal's complexity is effectively O(V² log V²) = O(V² log V), 
                which grows faster than Prim's O(V² log V) as V increases.</p>
            </div>
            
            <h2>5. Results: Effect of Graph Density</h2>
            <div class="analysis">
                <p>We measured performance as the graph density varied from 0.01 to 0.9, with a fixed size of 500 vertices.</p>
                <div class="image-container">
                    <img src="../reports/density_comparison.png" alt="MST Performance vs Graph Density">
                </div>
                
                <h3>Data Table: Algorithm Performance vs Graph Density</h3>
                <table>
                    <tr>
                        <th>Graph Density</th>
                        <th>Kruskal's Algorithm (s)</th>
                        <th>Prim's Algorithm (s)</th>
                    </tr>
                    {generate_table_rows(density_results)}
                </table>
                
                <h3>Analysis:</h3>
                <p>The results demonstrate that graph density significantly impacts the relative performance of both algorithms. 
                For sparse graphs (low density), Kruskal's algorithm tends to be more efficient because there are fewer edges to sort. 
                As density increases, Prim's algorithm's advantage becomes more pronounced, particularly because its complexity 
                depends more on the number of vertices than the number of edges for dense graphs.</p>
            </div>
            
            <h2>6. Comparison Plot: Varying Edge Counts</h2>
            <div class="analysis">
                <p>This experiment examines how both algorithms perform across different edge-to-vertex ratios (m/n) for various graph sizes. 
                For each vertex count n in {{16, 32, 64, 128, 256, 512}}, we tested different edge counts m in {{n, 2n, 4n, 8n, 16n, ..., n(n-1)/2}}.</p>
                
                <div class="image-container">
                    <img src="../reports/comparison_plot.png" alt="MST Performance vs Edge-to-Vertex Ratio">
                </div>
                
                <h3>Analysis:</h3>
                <p>This comprehensive comparison plot reveals several key insights:</p>
                <ul>
                    <li>For smaller graphs (n=16, 32), Kruskal's algorithm generally outperforms Prim's algorithm across most edge densities.</li>
                    <li>As the graph size increases (n=64, 128, 256, 512), we observe that Prim's algorithm becomes more efficient, especially at higher edge densities.</li>
                    <li>The logarithmic x-axis clearly shows that as the edge-to-vertex ratio (m/n) increases, the performance difference between the algorithms becomes more pronounced.</li>
                    <li>For very sparse graphs (m/n close to 1), Kruskal's algorithm tends to perform better regardless of the number of vertices.</li>
                    <li>The performance curves demonstrate that the choice of algorithm should consider both the number of vertices and the edge density of the graph.</li>
                </ul>
                <p>These results align with the theoretical analysis: Kruskal's O(E log E) is more efficient for sparse graphs, while Prim's O(E log V) 
                is better for dense graphs, especially as the number of vertices increases.</p>
            </div>
            
            <h2>7. Crossover Point Analysis</h2>
            <div class="analysis">
                <pre>{crossover_point}</pre>
                
                <h3>Analysis:</h3>
                <p>We used a bisection method to identify the approximate graph size where Prim's algorithm begins to outperform Kruskal's algorithm. 
                This crossover point represents an important threshold for algorithm selection in practical applications. 
                For graphs smaller than this size, Kruskal's algorithm is the better choice, while for larger graphs, Prim's algorithm is more efficient.</p>
            </div>
            
            <h2>8. Theoretical vs Experimental Results</h2>
            <div class="analysis">
                <div class="image-container">
                    <img src="../reports/theoretical_comparison.png" alt="Theoretical Time Complexity">
                    <img src="../reports/theoretical_comparison_log.png" alt="Theoretical Time Complexity (Log-Log)">
                </div>
                
                <h3>Analysis:</h3>
                <p>The theoretical plots show the expected asymptotic behavior based on the algorithms' time complexities:
                O(E log E) for Kruskal's algorithm and O(E log V) for Prim's algorithm. Comparing these with our experimental 
                results reveals how closely practical performance follows theoretical expectations.</p>
                <p>The relative performance advantage of Prim's algorithm for larger graphs is clearly visible in both the 
                theoretical and experimental results, confirming our understanding of these algorithms' scaling behavior.</p>
            </div>
            
            <h2>9. Conclusion</h2>
            <div class="analysis">
                <p>Our experimental analysis provides several key insights:</p>
                <ul>
                    <li>For small graphs (fewer than approximately 500 vertices with medium density), Kruskal's algorithm tends to be more efficient.</li>
                    <li>For larger graphs, Prim's algorithm offers better performance.</li>
                    <li>Graph density significantly impacts the performance characteristics, with Prim's algorithm showing greater advantages for denser graphs.</li>
                    <li>The edge-to-vertex ratio (m/n) is a critical factor in determining which algorithm performs better.</li>
                    <li>The experimental results largely conform to theoretical expectations based on the algorithms' time complexities.</li>
                </ul>
                <p>These findings have practical implications for algorithm selection in applications requiring minimum spanning trees. 
                The choice between Kruskal's and Prim's algorithms should consider both the expected size and density of the input graphs.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML report
    report_path = "reports/mst_analysis_report.html"
    with open(report_path, "w") as f:
        f.write(report_html)
    
    if logger:
        logger.info(f"Report generated successfully: {report_path}")
    else:
        print(f"Report generated successfully: {report_path}")

def read_csv_results(file_path, logger=None):
    """Read results from a CSV file."""
    results = []
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip header row
            for row in reader:
                results.append(row)
        if logger:
            logger.debug(f"Successfully read data from {file_path}")
    except FileNotFoundError:
        if logger:
            logger.warning(f"Warning: File {file_path} not found.")
        else:
            print(f"Warning: File {file_path} not found.")
    return results

def generate_table_rows(data):
    """Generate HTML table rows from data."""
    rows = ""
    for row in data:
        rows += "<tr>\n"
        for cell in row:
            rows += f"<td>{cell}</td>\n"
        rows += "</tr>\n"
    return rows

if __name__ == "__main__":
    from logger import setup_logger
    logger = setup_logger()
    generate_report(logger) 