# PyTorch to Pega ONNX — pdstools Examples

Convert PyTorch models to Pega-compatible ONNX using the `ONNXModel.from_pytorch()` classmethod.

## Why?

Pega Prediction Studio requires ONNX models with:

- **Fixed-size input tensors** — dynamic batch dimensions are rejected.
- **`pegaMetadata`** — a JSON object embedded in the ONNX file describing
  predictors, output, and model type.
- **2-D inputs** — shape `[1, N]` where N is the number of features.

`ONNXModel.from_pytorch()` handles all of this automatically.

## Quick Start

```python
import torch
from pdstools.infinity.resources.prediction_studio.local_model_utils import (
    Metadata, ONNXModel, OutcomeType, Output, Predictor,
)

# 1. Convert
onnx_model = ONNXModel.from_pytorch(
    my_model,
    torch.randn(1, 5),
    input_names=["features"],
    output_names=["ChurnProbability"],
    fixed_batch_size=True,          # required for Pega
)

# 2. Attach metadata
metadata = Metadata(
    type=OutcomeType.BINARY,
    output=Output(label_name="ChurnProbability"),
    predictor_list=[
        Predictor(name="Age",    index=1, input_name="features"),
        Predictor(name="Tenure", index=2, input_name="features"),
    ],
    model_version="1.0.0",         # enterprise governance field (optional)
    created_by="Falcons",
)
onnx_model.add_metadata(metadata)

# 3. Save & validate
onnx_model.save("model_pega.onnx")
onnx_model.validate()
```

## Enterprise Governance Fields

The `Metadata` model accepts optional Pega 24+ governance fields:

| Field | Description |
|---|---|
| `model_version` | Semantic version string |
| `created_by` | Author / team name |
| `objective` | Business objective |
| `rule_set` / `rule_set_version` | Target Pega RuleSet |
| `baseline_auc` / `baseline_accuracy` | Baseline metrics |
| `training_dataset` | Training data location |
| `experiment_id` | MLflow / experiment-tracker run ID |

These are serialised as camelCase with a `py` prefix in the ONNX metadata
(e.g. `model_version` → `pyModelVersion`) via Pydantic serialization aliases.

## Security Helpers

```python
from pdstools.infinity.resources.prediction_studio.model_security import (
    ModelSecurityValidator, compute_model_hash,
)

validator = ModelSecurityValidator()
result = validator.validate_metadata(metadata_dict)  # allow-list + size checks
sha = compute_model_hash("model.onnx")               # SHA-256 integrity hash
```

## Notebook

See `ONNX_PyTorch_Example.ipynb` for a complete walkthrough.
