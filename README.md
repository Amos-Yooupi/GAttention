# Dynamic System Prediction using Regional Graph Representation

## Project Overview

Real-time and accurate prediction of the long-term behavior of dynamic systems is crucial for identifying risks during unexpected events, while computational efficiency is significantly influenced by the scale of the dynamic system. However, existing neural network models often focus on optimizing network structures to improve accuracy, neglecting key issues like computational efficiency. To address this, we propose a general regional graph representation method, which can be deployed in graph-based models. This approach reduces the graph size in advance and combines fusion graph convolution with lightweight convolution modules to extract the topological information of dynamic systems, providing a feasible solution for real-time computation. Our experimental results show that the introduction of a time-aware expert module significantly enhances model performance, achieving an optimal balance between computation speed and prediction accuracy. This framework holds broad application potential in fields such as intelligent transportation, weather forecasting, and energy management.

![Diagram](figs/architecture.pdf) <!-- Insert your image path here -->

## Model Parameters

Below are the model parameters:

json
{
  "embed_dim": 32,
  "hidden_dim": 32,
  "out_dim": 4,
  "in_len": 24,
  "out_len": 24,
  "num_head_for_time": 8,
  "num_head_for_node": 8,
  "num_moe_layer": 3,
  "num_expert": 5,
  "top_k": 3,
  "att_mode": "linear",
  "activation": "relu",
  "is_fusion": true,
  "num_blocks": 2,
  "embedding_choose": "Traffic",
  "is_load": true,
  "model_path": "model_parameter/traffic",
  "lr": 3e-3,
  "split_ratio": [0.6, 0.2, 0.2],
  "batch_size": 64,
  "epoch": 6000,
  "vehicle_dim": 4,
  "weather_dim": 1,
  "bridge_dim": 6,
  "pier_dim": 3,
  "traffic_dim": 1,
  "span": [5, 6, 7],
  "train": [4, 5, 6, 7],
  "is_region": true,
  "representation": "graph",
  "partial_ratio": 0.5,
  "num_longitude": 384,
  "num_latitude": 512,
  "region_order": 1
}

## Running Instructions
Run the run_this.py file. You can select different datasets through the embedding_choose parameter and choose different graph region orders using the region_order parameter.
## Application Areas
- Intelligent Transportation: Real-time traffic prediction and traffic management optimization.
- Weather Forecasting: Accurate prediction of dynamic weather data.
- Energy Management: Optimizing energy distribution and predicting system load.
