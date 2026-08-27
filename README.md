# mlops-pytorch-pipeline

A production-style pipeline that trains a CIFAR-10 image classifier in
PyTorch, packages training and serving as Docker images, and deploys both
to Kubernetes (training as a `Job`, serving as a `Deployment` behind a
`Service` and `HorizontalPodAutoscaler`).

## Architecture

```
                                   ┌────────────────────────────┐
                                   │        GitHub repo          │
                                   │  main ← develop ← feature/* │
                                   │  .github/workflows/ci.yml   │
                                   └──────────────┬───────────────┘
                                                  │ docker build
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             │                             │
                    ▼                             │                             ▼
        docker/Dockerfile.train                   │               docker/Dockerfile.serve
        (multi-stage, pinned deps)                │           (multi-stage, non-root, HEALTHCHECK)
                    │                             │                             │
                    ▼                             │                             ▼
            image: mlops-train:v1                 │               image: mlops-serve:v1
                    │                             │                             │
                    │            Kubernetes namespace: ml-training              │
                    │      ┌───────────────────────┴────────────────────┐      │
                    ▼      ▼                                            ▼      ▼
        ┌───────────────────────────┐                    ┌───────────────────────────────┐
        │  Job: cifar10-training     │                    │  Deployment: model-serving      │
        │  ConfigMap: training-config│  checkpoints-pvc   │  2 replicas, rolling update      │
        │  PVC: training-data-pvc    │ ───(read-only)───▶ │  liveness/readiness → GET /health│
        │  PVC: checkpoints-pvc      │  (writes here)     │  Service: model-serving (:80)    │
        │  CPU 2 / Mem 4Gi           │                    │  HPA: 2-5 replicas @ 70% CPU     │
        └───────────────────────────┘                    └───────────────┬───────────────┘
                                                                           │
                                                                           ▼
                                                              curl -X POST /predict
                                                              (via port-forward or LB)
```

Training writes a checkpoint to a `ReadWriteOnce` PVC; the serving
`Deployment` mounts that same PVC read-only so every replica serves the
same model version. Redeploying with a newer checkpoint is a rollout of
the `Deployment` (rolling update, `maxSurge: 1` / `maxUnavailable: 0`, so
capacity never drops during a rollout).

## Repository layout

```
mlops-pytorch-pipeline/
├── README.md
├── .gitignore
├── .github/workflows/ci.yml     # lint, unit tests, docker build on every PR
├── src/
│   ├── model.py                 # SimpleCNN + CIFAR-adapted ResNet-18
│   ├── dataset.py                # CIFAR-10 transforms + DataLoaders
│   ├── train.py                  # config-driven training loop, JSON logs, early stopping
│   └── serve.py                  # FastAPI app: GET /health, POST /predict
├── configs/training_config.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── training-job.yaml         # PVCs + Job
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
└── tests/test_model.py
```

## Local setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements/train.txt
pip install pytest ruff  # dev-only

pytest tests -v
```

## Docker: training

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-train:v1
```

The first run downloads CIFAR-10 (~170MB) into `./data`. To smoke-test the
image without waiting through full epochs, cap the batches per epoch:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  -e MAX_BATCHES=5 \
  mlops-train:v1
```

## Docker: serving

```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

docker run --rm -p 8080:8080 \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-serve:v1

curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Kubernetes

```bash
# Make the local images visible to your cluster first.
# Minikube:  eval $(minikube docker-env) && docker build ...  (rebuild after switching env)
# kind:      kind load docker-image mlops-train:v1 mlops-serve:v1

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml

kubectl get pods -n ml-training -w        # wait for cifar10-training-xxxxx to Complete
kubectl logs -n ml-training job/cifar10-training

kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml             # requires the metrics-server addon

kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Secrets management

No secrets are checked into this repo (`.gitignore` blocks `.env`,
`secrets.yaml`, and any `**/secrets/*.yaml`). For anything that needs a
real credential (a private model registry, a cloud storage bucket for
checkpoints), create a Kubernetes `Secret` imperatively and reference it
from the manifest instead of writing values into a tracked YAML file:

```bash
kubectl create secret generic registry-credentials \
  --from-literal=username=... --from-literal=password=... \
  -n ml-training
```

## Git workflow

- `main` — always deployable; only updated via merged PRs from `develop`.
- `develop` — integration branch, branched from `main`.
- `feature/*` — one branch per unit of work (e.g. `feature/docker-training`,
  `feature/k8s-deployment`), merged back into `develop` via PR.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `chore:`, ...).

## CI

`.github/workflows/ci.yml` runs on every PR and push to `develop`/`main`:
lints `src/` and `tests/` with `ruff`, runs `pytest`, then builds both
Docker images to catch Dockerfile regressions before merge.
