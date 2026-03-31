This project involves developing Machine Learning models using Python. The agent must adhere to the following rules and best practices:

## Environment and Stack
*   **Language:** Python 3.12+.
*   **Frameworks:** Use `PyTorch` and `PyTorch Lightning` primarily, with `Hydra` to manage config.
*   **Dependency Management:** Use `uv` for dependency management. Ensure `pyproject.toml` is updated with required packages.

## Code Style
*   **Linting:** Adhere to `flake8` and `black` style guidelines with max line length 88.
*   **Tensor manipulation:** favor using `einops` for any tensor transformation or multiplication for readability.

## ML-Specific Methodology
*   **Experiment Tracking:** Use `wandb` for versioning, tracking experiments.
*   **Data Handling:**
    *   Treat the operational environment with the utmost respect. Never hardcode data paths or credentials.
*   **Model Versioning:** Save all trained models in checkpoint folders with associated experiment IDs, and the experiment folders should be grouped under directory named after the wandb project name. Keep a config.yaml within each folder which contains required information to reinstantiate the model for loading during inference.

## Development
*   **Code Structure:** Develop based on the structure below
    ```
    |- src
        |- data: data related object and utils, prefer to have one data class for each data source
        |- modules: nn.module model or model components
        |- utils
        |- interfaces: high level lightning module classes which correspond to training task types (e.g. ImageGenVQVAE). It usally has a `model` attribute that defines the main model for the task and methods that govern the training for that task.
    |- configs: high level task based main configs e.g. train_{MODEL}_{DATA}_v1.yaml config, with model and data subconfigs. Trainer related stuff in main config only under trainer section.
    |- train.py
    ```
*   **Testing:** First, activate the venv by running source .venv/bin/activate from the repository root. When conduct testing which require artifact checking, save all artifacts in `tmp` directory. Try to be efficient when testing training script (use small subset of data).