# Based on https://github.com/SotaroKaneda/MLCarbon/
from math import exp
import json
import numpy as np

def calculate_throughput(num_parameters):
    """
    Calculate tensor and pipeline throughput using the regression coefficients from MLCarbon
    """
    # Coefficients from the paper
    tensor_coeffs = {
        'a': -8.82079068e-20,
        'b': 1.68591116e-09,
        'c': 1.33954735e+02
    }
    
    pipeline_coeffs = {
        'a': -5.60233749e-23,
        'b': 8.45435587e-11,
        'c': 1.34546129e+02
    }
    
    # Calculate throughputs using polynomial fit: y = ax² + bx + c
    tensor_throughput = (
        tensor_coeffs['a'] * num_parameters**2 + 
        tensor_coeffs['b'] * num_parameters + 
        tensor_coeffs['c']
    )
    
    pipeline_throughput = (
        pipeline_coeffs['a'] * num_parameters**2 + 
        pipeline_coeffs['b'] * num_parameters + 
        pipeline_coeffs['c']
    )
    
    return tensor_throughput, pipeline_throughput

def calculate_gpu_requirements(num_parameters):
    """
    Calculate total number of GPUs needed using the regression coefficients
    """
    gpu_coeffs = {
        'a': -2.12910565e-21,
        'b': 4.39684339e-09,
        'c': 7.99173057e+02
    }
    
    total_gpus = (
        gpu_coeffs['a'] * num_parameters**2 + 
        gpu_coeffs['b'] * num_parameters + 
        gpu_coeffs['c']
    )
    
    return max(1, int(total_gpus))  # Ensure at least 1 GPU

def calculate_batch_size(num_parameters):
    """
    Calculate optimal batch size using the regression coefficients
    """
    batch_coeffs = {
        'a': -4.29439186e-01,
        'b': 5.21376002e+01,
        'c': 1.43737095e+03
    }
    
    batch_size = (
        batch_coeffs['a'] * num_parameters**2 + 
        batch_coeffs['b'] * num_parameters + 
        batch_coeffs['c']
    )
    
    return max(1, int(batch_size))  # Ensure positive batch size

def estimate_carbon_footprint(model_params, training_hours, power_usage_effectiveness=1.1):
    """
    Estimate the carbon footprint for training an LLM
    """
    # Calculate basic metrics
    tensor_throughput, pipeline_throughput = calculate_throughput(model_params)
    num_gpus = calculate_gpu_requirements(model_params)
    batch_size = calculate_batch_size(model_params)
    
    # Assume A100 GPU power consumption (400W per GPU)
    gpu_power_watts = 400
    
    # Calculate total energy consumption
    total_power_watts = num_gpus * gpu_power_watts
    total_power_kw = total_power_watts / 1000
    
    # Apply PUE
    total_power_with_pue = total_power_kw * power_usage_effectiveness
    
    # Calculate energy consumption in kWh
    energy_consumption_kwh = total_power_with_pue * training_hours
    
    # Assume average grid carbon intensity (500 gCO2/kWh)
    carbon_intensity = 500  # gCO2/kWh
    
    # Calculate total emissions in kg CO2eq
    emissions_kg = (energy_consumption_kwh * carbon_intensity) / 1000
    
    return {
        "model_parameters": model_params,
        "number_of_gpus": num_gpus,
        "batch_size": batch_size,
        "tensor_throughput": tensor_throughput,
        "pipeline_throughput": pipeline_throughput,
        "training_hours": training_hours,
        "total_energy_kwh": energy_consumption_kwh,
        "total_emissions_kg": emissions_kg
    }

def main():
    # Example models
    models = {
        "GPT-3 175B": 175e9,
        "GPT-4 1T": 1e12,
        "BLOOM 176B": 176e9,
        "LLaMA 65B": 65e9,
        "Claude 3.5 Sonnet": 125e9,
        "Gemini 1.5 Pro": 125e9,
        "GPT-4o": 125e9,
        "GPT-4o-mini": 125e9,
        "GPT-4o-2024-08-06": 125e9,
        "GPT-4o-2024-05-13": 125e9,
        "GPT-4o-2024-03-14": 125e9,
        
    }
    
    print("Carbon Footprint Estimates for Different LLMs")
    print("=" * 50)
    
    results = {}
    for model_name, params in models.items():
        # Estimate for different training durations
        training_scenarios = [24, 168, 720]  # 1 day, 1 week, 1 month
        
        model_results = []
        for hours in training_scenarios:
            result = estimate_carbon_footprint(params, hours)
            model_results.append({
                "training_hours": hours,
                "emissions": result["total_emissions_kg"],
                "energy_consumption": result["total_energy_kwh"],
                "gpus_required": result["number_of_gpus"]
            })
            
            print(f"\n{model_name} ({hours} hours training):")
            print(f"- Estimated emissions: {result['total_emissions_kg']:.2f} kg CO2eq")
            print(f"- Energy consumption: {result['total_energy_kwh']:.2f} kWh")
            print(f"- GPUs required: {result['number_of_gpus']}")
            print(f"- Batch size: {result['batch_size']}")
        
        results[model_name] = model_results
    
    # Save results
    with open("emissions/llm_carbon_estimates.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()