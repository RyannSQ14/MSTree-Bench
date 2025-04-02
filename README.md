# Minimum Spanning Tree Algorithms - Experimental Analysis

<div align="center">
<pre>
    ███╗   ███╗███████╗████████╗
    ████╗ ████║██╔════╝╚══██╔══╝
    ██╔████╔██║███████╗   ██║   
    ██║╚██╔╝██║╚════██║   ██║   
    ██║ ╚═╝ ██║███████║   ██║   
    ╚═╝     ╚═╝╚══════╝   ╚═╝   
                              
 ALGORITHMS EXPERIMENTAL ANALYSIS
</pre>
  <br>
  <em>Comparing Kruskal's and Prim's algorithms through experimental analysis</em>
</div>

<br>

This project implements and analyzes the performance of Kruskal's and Prim's algorithms for finding Minimum Spanning Trees (MST) in graphs.

## Project Structure

```
.
├── docs/                          # Documentation
│   └── HW 4 - Experiments with MST Algorithms-1.pdf  # Assignment specs
│
├── logs/                          # Log files (created when running)
│
├── reports/                       # Generated reports and visualizations
│   ├── mst_analysis_report.html   # HTML report with analysis
│   ├── mst_analysis_report.md     # Markdown version of the report
│   ├── comparison_plot.png        # Performance comparison chart
│   ├── size_comparison.png        # Graph size impact visualization
│   ├── density_comparison.png     # Density impact visualization
│   └── ...                        # Other generated files
│
├── src/                           # Source code
│   ├── main.py                    # Main entry point
│   ├── mst.py                     # MST algorithm implementations
│   ├── experiment.py              # Experimental procedures
│   ├── report_generator.py        # Report generation logic
│   └── logger.py                  # Logging utilities
│
├── venv/                          # Virtual environment (created when running)
│
├── README.md                      # Project documentation
├── requirements.txt               # Project dependencies
└── run.sh                         # Execution script
```

## Requirements

This project requires Python 3.6+ with the following packages:
- matplotlib
- numpy

## How to Run

### Using the Run Script (Recommended)

The easiest way to run the project is by using the included run script:

```bash
./run.sh
```

This script will:
1. Create a virtual environment if it doesn't exist
2. Install required dependencies
3. Create necessary directories
4. Run the main program
5. Deactivate the virtual environment when finished

### Manual Execution

Alternatively, you can run the project manually:

1. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install matplotlib numpy
```

3. Run the main script:
```bash
python src/main.py
```

## Experiment Details

The project runs several experiments:

1. **Effect of Graph Size**: Comparing algorithm performance as the number of vertices increases from 10 to 1000
2. **Effect of Graph Density**: Comparing algorithm performance as the graph density changes from very sparse (0.01) to dense (0.9)
3. **Edge Count Comparison**: Performance across different edge-to-vertex ratios for various graph sizes
4. **Crossover Point Analysis**: Finding where Prim's algorithm becomes more efficient than Kruskal's
5. **Theoretical Analysis**: Comparing experimental results with theoretical time complexities

## Viewing Results

After running the program, you can view the results in two formats:

### HTML Report
A comprehensive HTML report is generated at:
```
reports/mst_analysis_report.html
```

### Markdown Report
A markdown version of the report is available at:
```
reports/mst_analysis_report.md
```

The reports include:
- Detailed descriptions of the algorithms
- Performance data tables
- Visualizations of the results
- Analysis of the findings

## Implementation Details

### Data Structures
- **MinHeap**: Custom implementation for priority queue operations needed by Prim's algorithm
- **DisjointSet**: Implementation with path compression and union by rank for Kruskal's algorithm

### Kruskal's Algorithm
Kruskal's algorithm uses a disjoint-set data structure with path compression and union by rank for efficiency. The implementation sorts all edges and adds them to the MST in order of increasing weight if they don't create a cycle.

Time complexity: O(E log E), where E is the number of edges.

### Prim's Algorithm
Prim's algorithm uses a binary heap (priority queue) to efficiently select the next minimum-weight edge to add to the MST. It starts from a single vertex and grows the MST by adding one vertex at a time.

Time complexity: O(E log V), where V is the number of vertices.

### Random Graph Generation
The graph generation function creates random connected graphs with:
- Specified number of vertices (n)
- Specified number of edges (m)
- Random edge weights
- Guaranteed connectivity

Special handling is implemented for:
- Complete graphs
- Dense graphs (using deletion approach)
- Sparse graphs (using addition approach)

## Logging

Detailed logging is implemented throughout the project, capturing:
- Experiment progress
- Graph generation details
- Algorithm execution
- Performance measurements

Logs are stored in the `logs/` directory for debugging and analysis. 