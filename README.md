# Trust Estimation with Discount Factors

A Python project for estimating human trust dynamics using Bayesian parameter estimation and discount factor analysis. This repository implements models for tracking how trust in autonomous systems evolves based on feedback history, with support for different weighting schemes of historical performance.

## Overview

This project models trust as a dynamic process that evolves based on historical feedback. The key innovation is the use of **discount factors** to determine how much weight past performance carries when estimating current trust. The framework:

1. **Parameterizes trust** using a Beta distribution with four key parameters:
   - `alpha0`, `beta0`: Prior belief parameters
   - `ws`: Weight for successful feedback
   - `wf`: Weight for failed feedback

2. **Uses discount factors** to model how historical feedback decays over time
3. **Optimizes parameters** using BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimization
4. **Searches for optimal discount factors** by fitting models across a range of values

## Project Structure

```
├── SearchForBestDF.py              # Main script to find optimal discount factors
├── ConstantDF.py                   # Script to fit data with a fixed discount factor
├── PlotTrustEstimates.py           # Visualization script for trust estimates vs feedback
├── AggregatedData/
│   └── RandomData.csv              # Sample data with participant feedback
├── classes/
│   ├── DataReader.py               # Reads and loads aggregated data
│   ├── DiscountFactors.py          # Discount factor classes
│   ├── ParamsEstimator.py          # BFGS parameter optimization
│   ├── TrustEstimator.py           # Trust calculation with discount factors
│   └── Utils.py                    # Utility classes and functions
└── images/                         # Output directory for visualizations
```

## Core Components

### Classes

#### `TrustEstimator`
Computes trust estimates using the Beta distribution and discount factors.

- **Input**: Discount factor, trust parameters, performance history
- **Output**: Estimated trust, alpha and beta parameters
- **Key Method**: `get_trust()` - Returns trust estimate for current site

#### `DiscountFactors.ConstantDF`
A discount factor model where the discount value remains constant regardless of site number.

- **Usage**: `df = ConstantDF(discount_factor=0.9)`
- **get_value(site_number)**: Returns the constant discount factor

#### `ParamsEstimatorBFGS`
Optimizes the four trust parameters (α₀, β₀, wₛ, wf) using BFGS optimization by maximizing log-likelihood.

- **Input**: Performance history, trust feedback observations
- **Output**: Optimized parameter values
- **Key Methods**:
  - `neg_log_likelihood()`: Computes negative log-likelihood
  - `gradients()`: Computes gradient for optimization

#### `AggregatedDataReader`
Reads participant feedback data from CSV files.

- **Default path**: `./AggregatedData/RandomData.csv`
- **Expected format**: DataFrame with performance and trust feedback columns

#### `LearnerSettings`
Configuration container for learning parameters and discount factor ranges.

### Scripts

#### `SearchForBestDF.py`
Searches across a range of discount factors to find which produces the best model fit.

**Usage**:
```bash
python SearchForBestDF.py --group-num <N>
```

**Parameters**:
- `START`: Starting discount factor (default: 0.1)
- `END`: Ending discount factor (default: 1.0)
- `STEP_SIZE`: Increment between tested factors (default: 0.01)
- `--group-num`: Divides computation into 13 groups for parallel processing

**Output**: Generates timestamped directory with results for each discount factor and cluster

#### `ConstantDF.py`
Fits the trust model with a single, fixed discount factor.

**Parameters**:
- `DISCOUNT_FACTOR`: The constant discount factor to use (default: 1.0)
- Learning rate, iterations, and tolerance settings

**Output**: Fitted models and trust estimates

#### `PlotTrustEstimates.py`
Generates visualization comparing estimated trust against observed feedback.

**Usage**:
```bash
python PlotTrustEstimates.py --path <path_to_estimates>
```

**Output**: Individual plots for each participant showing trust trajectory

## Key Concepts

### Discount Factors
A discount factor (γ ∈ [0,1]) determines how historical feedback is weighted:
- **γ = 1.0**: All past feedback weighted equally (memory-less)
- **γ = 0.5**: Recent feedback weighted more heavily than distant feedback
- **γ = 0.0**: Only the most recent feedback matters

### Trust Model
Trust is modeled as a Beta distribution: Trust ~ Beta(α, β)

Where:
```
α(t) = α₀ + γ·(α(t-1) - α₀) + wₛ·Performance(t)
β(t) = β₀ + γ·(β(t-1) - β₀) + wf·(1 - Performance(t))
Trust = α / (α + β)
```

### Parameter Optimization
The BFGS optimizer finds parameters that maximize the log-likelihood of observed trust feedback given the performance history.

## Installation

### Requirements
- Python 3.8+
- pandas
- numpy
- scipy
- tqdm
- matplotlib
- seaborn

### Setup
```bash
# Clone the repository
git clone <repository_url>
cd discounting_trust_update

# Create virtual environment (optional but recommended)
conda create -n trust-analysis python=3.10
conda activate trust-analysis

# Install dependencies
pip install pandas numpy scipy tqdm matplotlib seaborn
```

## Usage

### 1. Prepare Your Data
Place your aggregated data in:
```
AggregatedData/RandomData.csv
```

Expected columns: `Performance`, `Trust`, `Participant ID`, `Cluster`, etc.

### 2. Find Optimal Discount Factor
```bash
# Single run
python SearchForBestDF.py

### 3. Fit with a Specific Discount Factor
```bash
python ConstantDF.py
```

### 4. Visualize Results
```bash
python PlotTrustEstimates.py --path <path to BestEstimates.csv>
```

## Output

The pipeline generates:
1. **Model parameters**: Fitted α₀, β₀, wₛ, wf for each discount factor and cluster
2. **Trust estimates**: Point-by-point trust estimates for all participants
3. **Performance metrics**: RMSE and other fit quality measures
4. **Visualizations**: Plots comparing estimated vs actual trust trajectories

## Data Format

**Input (RandomData.csv)**:
- `Participant ID`: Unique identifier
- `Performance`: 0/1 success/failure feedback
- `Trust`: Observed trust ratings
- `Cluster`: Participant group/cluster assignment

**Output (BestEstimates.csv)**:
- All input columns plus:
- `Trust Estimate`: Model-predicted trust
- `Discount Factor`: Discount factor used
- `Alpha`, `Beta`: Beta distribution parameters
