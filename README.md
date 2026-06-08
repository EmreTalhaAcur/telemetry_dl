Project: Telemetry DL Pipeline

```markdown
# Vehicle Telemetry Deep Learning Pipeline

This project is an industry-standard modular **Neural Network** built with PyTorch to predict real-time engine temperature based on engine revolutions per minute (RPM) and engine load data.

## 🏗️ Architecture (Separation of Concerns)
To ensure maintainability, scalability, and clean code standards, the codebase is decoupled into independent modules rather than a monolithic script:
* `src/config.py`: Centralized hub for hyperparameters (Batch Size, Epochs, Learning Rate) and hardware allocation (CPU/GPU - CUDA selection).
* `src/dataset.py`: Logistics layer using PyTorch `Dataset` and `DataLoader` to handle data pipelines and serve mini-batches efficiently.
* `src/model.py`: Neural network architecture incorporating `nn.Linear` layers and `ReLU` activation functions to introduce non-linearity.
* `src/trainer.py`: Core execution engine managing the training loop, including Forward Pass, Loss computation, Backpropagation, and weight updates.
* `main.py`: The entry point that orchestrates the entire pipeline.

## 📉 Training Insights
The network initializes with random weights and consistently decreases the Mean Squared Error (MSE) loss across epochs. Utilizing backpropagation and the Adam optimizer, it successfully captures the underlying regression patterns and exports the optimized weights as `telemetry_model.pt`.

## 🛠️ Tech Stack
* Python 3
* PyTorch (torch.nn, torch.optim)
* NumPy
