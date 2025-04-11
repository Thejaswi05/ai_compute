# AI Compute Carbon Footprint Analysis

This project provides tools and examples for measuring and estimating the carbon footprint of AI computations, from simple machine learning models to large language models (LLMs).

## Project Components

### 1. Random Forest Carbon Measurement (codecarbon_example.ipynb)
Demonstrates real-time carbon emission tracking during model training using CodeCarbon. The experiment compares three Random Forest configurations:

- Basic RF (100 trees, 10,000 samples):
  - Training time: 1.53 seconds
  - Emissions: 0.000002 kg CO2eq
  - Accuracy: 85.1%

- Intermediate RF (500 trees, 50,000 samples):
  - Training time: 74.29 seconds
  - Emissions: 0.000103 kg CO2eq
  - Accuracy: 92.95%

- Complex RF (1000 trees, 100,000 samples):
  - Training time: 458.54 seconds
  - Emissions: 0.000637 kg CO2eq
  - Accuracy: 96.075%

Hardware: Apple M1 Pro (10 CPU cores)

### 2. LLM Carbon Estimator (llm_carbon_estimate.py)
Estimates carbon emissions for training large language models based on the MLCarbon methodology. Features include:
- Parameter-based throughput calculation
- GPU requirement estimation
- Batch size optimization
- Energy consumption and CO2 emission calculations

Supports various model sizes including:
- GPT-3 (175B parameters)
- GPT-4 (1T parameters)
- BLOOM (176B parameters)
- LLaMA (65B parameters)
- Claude 3.5 Sonnet (125B parameters)
- Gemini 1.5 Pro (125B parameters)

## Key Findings

1. Model Complexity vs. Emissions:
   - Emissions don't scale linearly with model size
   - The complex Random Forest model produced 318.5x more emissions than the basic model
   - Training time increased by 370x from basic to complex configuration

2. Environmental Impact vs. Performance:
   - Significant environmental cost for marginal performance gains
   - 11% accuracy improvement required 318.5x more emissions

## Usage

### CodeCarbon Example
'''python
from codecarbon import EmissionsTracker
tracker = EmissionsTracker()
tracker.start()
Your model training code here
emissions = tracker.stop()
'''

### LLM Carbon Estimator
'''python
from llm_carbon_estimate import estimate_carbon_footprint

results = estimate_carbon_footprint(
    model_params=175e9,  # 175B parameters
    training_hours=24
)
'''

## Requirements
- codecarbon
- scikit-learn
- pandas
- numpy
- matplotlib

## Data
Emissions data is stored in the `emissions/` directory:
- `emissions.csv`: Contains detailed measurements from CodeCarbon
- `llm_carbon_estimates.json`: Contains LLM training emission estimates

