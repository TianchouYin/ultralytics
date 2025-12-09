# Ultralytics YOLO AI Coding Agent Instructions

## Project Overview

Ultralytics YOLO (v8.3.225) is a production-grade computer vision framework supporting object detection, segmentation, classification, pose estimation, and OBB tasks. The codebase is designed for both research flexibility and deployment efficiency across 15+ export formats.

## Core Architecture

### Package Structure

```
ultralytics/
├── engine/        # Base training/validation/prediction infrastructure
├── models/        # Task-specific model implementations (detect, segment, classify, pose, obb, sam, rtdetr, nas, yolo, yoloe)
├── nn/            # Neural network modules, layers, and model builders
├── data/          # Dataset loaders, augmentation, and build utilities
├── utils/         # Cross-cutting utilities (exports, benchmarks, torch helpers)
├── cfg/           # Configuration files and CLI entrypoint
└── solutions/     # High-level solutions (heatmaps, tracking, analytics)
```

### The Engine Pattern (Critical)

All tasks follow a consistent inheritance pattern from `ultralytics/engine/`:

- **BaseTrainer** → Task-specific trainer (e.g., `DetectionTrainer`, `ClassificationTrainer`)
- **BaseValidator** → Task-specific validator
- **BasePredictor** → Task-specific predictor

**Key insight**: Custom behavior is implemented by subclassing these base engines, not by modifying the YOLO model class directly. See `ultralytics/models/yolo/detect/train.py` for the canonical example.

### Model Loading Hierarchy

1. **YAML → torch.nn.Module**: `ultralytics/nn/tasks.py` contains model builders (`DetectionModel`, `ClassificationModel`, etc.) that parse YAML configs
2. **High-level API**: `YOLO()` class in `ultralytics/engine/model.py` wraps models and automatically selects the correct trainer/validator/predictor
3. **Task detection**: Model task is inferred from the checkpoint's `model.model[-1]` layer type or explicitly set

## Development Workflows

### Training a Model

```python
from ultralytics import YOLO

# Load pretrained or build from YAML
model = YOLO("yolo11n.pt")  # or "yolo11n.yaml"

# Train with dataset YAML
results = model.train(
    data="coco8.yaml",      # Dataset config (required)
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,               # GPU device or 'cpu'
    project="runs/detect",  # Output directory
    name="experiment"
)
```

**Dataset YAML requirements**: Must include `path`, `train`, `val`, and `names` fields. See `ultralytics/cfg/datasets/` for examples.

### CLI Usage

The CLI (`yolo` command) is defined in `ultralytics/cfg/__init__.py::entrypoint()`:

```bash
# Syntax: yolo TASK MODE ARGS
yolo detect train data=coco8.yaml model=yolo11n.pt epochs=10
yolo segment predict model=yolo11n-seg.pt source=image.jpg
yolo classify val model=yolo11n-cls.pt data=imagenet
```

**Important**: `TASK` is optional (auto-detected from model), `MODE` is required.

### Export for Deployment

```python
model = YOLO("yolo11n.pt")
model.export(format="onnx")  # Creates yolo11n.onnx
```

**Supported formats** (see `ultralytics/engine/exporter.py::export_formats()`):

- **PyTorch**: torchscript
- **ONNX**: onnx
- **TensorRT**: engine
- **CoreML**: coreml, mlmodel (macOS only, no Windows)
- **TensorFlow**: saved_model, pb, tflite, tfjs
- **Edge**: edgetpu (requires int8), ncnn, mnn, rknn, imx
- **Mobile**: paddle (conflicts with ONNX protobuf)

**Export constraints**:

- TensorRT requires ONNX as intermediate format
- CoreML requires torch>=1.11, not supported on Windows or Python 3.13
- TFLite requires Python>=3.10, Linux recommended due to TF install conflicts
- `nms=True` and `dynamic=True` are mutually exclusive for CoreML

### Testing

```bash
# Run tests (pytest)
pytest tests/test_python.py -v
pytest tests/test_cli.py::test_train -k detect

# Benchmarking
yolo benchmark model=yolo11n.pt data=coco8.yaml
```

**Test structure**: `tests/` uses pytest with markers (`@pytest.mark.slow`, `@pytest.mark.skipif`)

## Project-Specific Conventions

### Configuration Management

- **Default config**: `ultralytics/cfg/default.yaml` defines all training/val/predict parameters
- **Loaded as**: `DEFAULT_CFG` (IterableSimpleNamespace) in `ultralytics/utils/__init__.py`
- **Override priority**: CLI args > function kwargs > YAML config > defaults
- **Settings file**: `~/.config/Ultralytics/settings.json` (or `YOLO_CONFIG_DIR` env var)

### Data Augmentation

`ultralytics/data/augment.py` uses `Albumentations` and custom transforms:

- Training: `RandomResizedCrop`, `Mosaic`, `MixUp` (for detection/segmentation)
- Classification: Uses torchvision transforms (`RandomResizedCrop` train, `CenterCrop` val)
- Key classes: `BaseTransform`, `Compose`, `LetterBox`

### Model Architecture YAML

Models defined in `ultralytics/cfg/models/{version}/{model}.yaml`:

```yaml
nc: 80  # number of classes
depth_multiple: 0.33  # model depth scaling
width_multiple: 0.25  # layer channel scaling

backbone:
  - [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  # ...

head:
  - [-1, 1, Detect, [nc]]  # task layer
```

**Parsing**: `ultralytics/nn/tasks.py::parse_model()` converts YAML to torch.nn.Sequential

### Custom Trainers

Extend base engines for custom behavior:

```python
from ultralytics.models.yolo.detect import DetectionTrainer

class CustomTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        # Custom dataset logic
        pass
  
    def get_model(self, cfg, weights=None):
        # Custom model initialization
        pass

# Use with YOLO API
model = YOLO("yolo11n.yaml")
model.train(data="custom.yaml", trainer=CustomTrainer)
```

**Common override points**: `build_dataset`, `get_model`, `get_validator`, `preprocess_batch`

### World Models & Grounding

YOLO-World and YOLO-E support open-vocabulary detection:

- Located in `ultralytics/models/yolo/world/` and `ultralytics/models/yolo/yoloe/`
- Use text embeddings for class names (via `ultralytics/nn/text_model.py`)
- Support grounding datasets with `build_grounding()` from `ultralytics/data/build.py`
- Training from scratch: `WorldTrainerFromScratch`, `YOLOETrainerFromScratch` handle mixed detection + grounding datasets

### Logging & Callbacks

Default integrations (opt-in via `settings.json`):

- **MLflow**, **TensorBoard**: Auto-enabled during training
- **ClearML**, **Comet**, **DVC**, **Neptune**, **WandB**: Via callback registration
- Callback system: `ultralytics/utils/callbacks/` with hooks (on_train_start, on_epoch_end, etc.)

## Common Pitfalls & Anti-Patterns

### ❌ Don't modify YOLO class directly

```python
# WRONG - fragile and loses base functionality
model = YOLO("yolo11n.pt")
model.custom_method = lambda: ...
```

### ✅ Use Trainer/Validator/Predictor subclasses

```python
# CORRECT - proper extension point
class CustomPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs):
        # Custom post-processing
        return super().postprocess(preds, img, orig_imgs)

model.predict(source="img.jpg", predictor=CustomPredictor)
```

### ❌ Don't assume sequential execution

The codebase supports:

- Multi-GPU training (DDP)
- Distributed validation
- Background processes for HUB integration

Always use `RANK` checks: `if RANK in {-1, 0}:` for main process logic

### ❌ Don't hardcode paths

```python
# WRONG
data = "/home/user/datasets/coco8.yaml"
```

### ✅ Use relative paths or YAML configs

```python
# CORRECT
data = "coco8.yaml"  # Searches ultralytics/cfg/datasets/
# or
data = "path/to/custom.yaml"  # Relative to CWD
```

### Device Handling

```python
# WRONG - bypasses device validation
model.to("cuda:0")

# CORRECT - uses select_device with error handling
results = model.train(device=0)  # or device="0,1,2" for multi-GPU
```

## File Naming Patterns

- Training scripts: `train.py` in task directories (e.g., `ultralytics/models/yolo/detect/train.py`)
- Validators: `val.py` in task directories
- Predictors: `predict.py` in task directories
- Model definitions: `ultralytics/cfg/models/{version}/{model}.yaml`
- Dataset configs: `ultralytics/cfg/datasets/{dataset}.yaml`
- Example scripts: `examples/{script}.py` (often prefixed like `xlptrain.py`, `xlpval.py` for custom implementations)

## External Integrations

- **Ultralytics HUB**: Cloud training/deployment via `ultralytics/hub/` (requires API key)
- **Roboflow**: Dataset downloading via `ultralytics/data/explorer/utils.py`
- **SAHI**: Sliced inference for small objects (see `examples/YOLOv8-SAHI-Inference-Video/`)
- **Triton Server**: Inference serving via `ultralytics/utils/triton.py`

## Performance Considerations

- **Half precision**: Use `half=True` for FP16 inference (2x speedup on compatible GPUs)
- **Batch inference**: Set `batch=32` for predict mode to process multiple images
- **TensorRT**: Fastest export format for NVIDIA GPUs, requires calibration dataset for INT8
- **Model size vs speed**: n < s < m < l < x (nano fastest, x-large most accurate)
- **Image size**: `imgsz=640` is default, larger sizes improve accuracy but slow inference

## Quick Reference Commands

```bash
# Install/update
pip install -U ultralytics

# Training
yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640

# Validation
yolo detect val model=yolo11n.pt data=coco8.yaml

# Inference
yolo detect predict model=yolo11n.pt source=path/to/images

# Export
yolo export model=yolo11n.pt format=onnx

# Benchmark
yolo benchmark model=yolo11n.pt data=coco8.yaml imgsz=640

# Settings
yolo settings  # View current settings
yolo settings reset  # Reset to defaults
```

## Documentation References

- **Full docs**: https://docs.ultralytics.com
- **Export guide**: docs/en/modes/export.md
- **Training guide**: docs/en/modes/train.md
- **Custom trainer example**: docs/en/usage/engine.md
- **CLI reference**: docs/en/usage/cli.md
