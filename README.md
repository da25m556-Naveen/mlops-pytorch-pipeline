# mlops-pytorch-pipeline

A PyTorch image classifier (Fashion-MNIST) packaged as two containerized
services — a training job and a FastAPI inference service — and deployed
to Kubernetes with health checks, rolling updates, and CPU-based
autoscaling.

## Architecture

```
                         ┌───────────────────────────┐
                         │   configs/*.yaml          │
                         │   (model/training params) │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
 data/ (FashionMNIST) ───────▶  src/train.py  ───────▶  checkpoints/classifier_v1.pt
   (auto-downloaded)         (src/dataset.py,               │
                              src/model.py: SimpleCNN)      │
                                                            ▼
                                                    src/serve.py (FastAPI)
                                                     GET  /health
                                                     POST /predict
                                                              │
                                                              ▼
                                                     JSON: predicted_class
                                                     + per-class probabilities

Containerized with:
  docker/Dockerfile.train  →  image: mlops-train:v1  (runs train.py to completion)
  docker/Dockerfile.serve  →  image: mlops-serve:v1  (long-running FastAPI server)
```

### Kubernetes topology (namespace: `ml-training`)

```
 ml-training namespace
 ├── ConfigMap  training-config            (mounted into the training Job as training_config.yaml)
 ├── PVC        training-data-pvc          (FashionMNIST download cache)
 ├── PVC        training-checkpoints-pvc   (shared between training Job and serving Deployment)
 ├── Job        model-training             (runs mlops-train:v1 to completion, writes checkpoint to the PVC)
 │
 ├── Deployment model-serving  (2 replicas, RollingUpdate, mlops-serve:v1)
 │     └── mounts training-checkpoints-pvc read-only, liveness/readiness on /health
 ├── Service    model-serving  (ClusterIP :80 → :8080, load-balances across replicas)
 └── HPA        model-serving  (scales the Deployment 2↔5 replicas on CPU > 70%)
```

The training Job and the serving Deployment are decoupled — the Job runs
once, saves a checkpoint to a PersistentVolumeClaim, and exits; the serving
Deployment reads that same checkpoint from the PVC (read-only) and serves
predictions independently of the Job's lifecycle.

## Repository structure

```
src/
  dataset.py   FashionMNIST loading + transforms
  model.py     SimpleCNN architecture + get_model() factory
  train.py     training loop, checkpointing, early stopping
  serve.py     FastAPI app: /health, /predict
configs/
  training_config.yaml       used inside containers/K8s
  local_dev_config.yaml      local override (gitignored)
docker/
  Dockerfile.train           multi-stage build for the training image
  Dockerfile.serve           multi-stage build for the serving image
k8s/
  namespace.yaml             ml-training Namespace
  configmap.yaml              training config mounted into the Job
  training-job.yaml           PVCs + training Job
  serving-deployment.yaml      inference Deployment (2 replicas, probes, resources)
  serving-service.yaml         ClusterIP Service in front of the Deployment
  hpa.yaml                    HorizontalPodAutoscaler (CPU-based)
tests/
  test_model.py               unit tests for model.py
requirements/
  train.txt / serve.txt / test.txt
```

## Prerequisites

- Python 3.11+
- Docker
- (for Kubernetes) `kubectl` and a cluster — these instructions use Minikube
- (for the HPA demo) the cluster's `metrics-server` addon

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/train.txt -r requirements/serve.txt -r requirements/test.txt
```

Run the unit tests:

```bash
pytest tests/
```

Run training locally (writes to `./checkpoints/classifier_v1.pt`):

```bash
python src/train.py
```

Run the serving API locally (expects a checkpoint at `./checkpoints/classifier_v1.pt`):

```bash
cd src && uvicorn serve:app --host 0.0.0.0 --port 8080
```

## Docker

Build and run the training image:

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-train:v1
```

Build and run the serving image:

```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 \
  -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
  mlops-serve:v1
```

Test it:

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_images/bag.png"
```

## Kubernetes deployment (Minikube)

```bash
minikube start
minikube addons enable metrics-server   # required for the HPA to read real CPU usage

# make the locally-built images available to the cluster (no registry needed)
minikube image load mlops-train:v1
minikube image load mlops-serve:v1

# 1. namespace + config
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml

# 2. training job — runs to completion, persists checkpoint to a PVC
kubectl apply -f k8s/training-job.yaml
kubectl wait --for=condition=complete job/model-training -n ml-training --timeout=900s

# 3. serving layer — reads the checkpoint from the same PVC
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# 4. verify
kubectl get pods -n ml-training
kubectl get hpa -n ml-training
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_images/bag.png"
```

> **Note:** the HPA's exact thresholds used : `minReplicas: 2`, `maxReplicas: 5`,
> `target CPU utilization: 70%`

## API reference

- `GET /health` → `{"status": "ok"}`, or `503` if the model checkpoint
  failed to load.
- `POST /predict` (multipart form field `image`) → predicted Fashion-MNIST
  class and per-class softmax probabilities, e.g.:
  ```json
  {
    "predicted_class": "Bag",
    "probabilities": { "T-shirt/top": 0.0, "...": 0.0, "Bag": 1.0 }
  }
  ```

## Configuration

`configs/training_config.yaml` (also embedded in `k8s/configmap.yaml`):

| Key                               | Meaning                                             |
|-----------------------------------|-----------------------------------------------------|
| `model.architecture`              | model factory key (`simple_cnn`)                    |
| `model.num_classes`               | number of output classes                            |
| `training.epochs`                 | max training epochs                                 |
| `training.batch_size`             | batch size for train/val loaders                    |
| `training.learning_rate`          | Adam learning rate                                  |
| `training.early_stopping_patience`| epochs without val-loss improvement before stopping |
| `data.data_dir`                   | FashionMNIST download/cache directory               |
| `output.checkpoint_dir`           | where checkpoints are written                       |
| `output.model_name`               | checkpoint filename                                 |
