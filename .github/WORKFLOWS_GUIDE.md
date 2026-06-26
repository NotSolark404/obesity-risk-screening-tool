# GitHub Actions Workflows Guide

---

## File locations

```
.github/
└── workflows/
    ├── ci.yml        → runs on every push and pull request
    ├── cd.yml        → runs on every push to main branch
    └── retrain.yml   → runs on new data, weekly schedule, or manual trigger
```

---

## `ci.yml` — Continuous Integration

### What it does

Every time anyone pushes a commit or opens a pull request, this workflow runs automatically and performs 4 checks in order:

```
1. Lint (ruff)         → checks code style, unused imports, syntax errors
2. Format (black)      → checks code is consistently formatted
3. dvc repro           → rebuilds the full ML pipeline from raw CSV
4. pytest              → runs all 5 test files
```

If any step fails, the workflow stops and shows a red cross. The push or PR is flagged as failing.

### Full file

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  ci:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Lint with ruff
        run: ruff check src/ app/ tests/

      - name: Format check with black
        run: black --check src/ app/ tests/

      - name: Run DVC pipeline
        run: dvc repro --no-run-cache

      - name: Run tests
        run: pytest -v --timeout=60
```

### How to trigger it

```bash
# Just push anything — CI runs automatically
git add .
git commit -m "feat: any change"
git push
```

---

## `cd.yml` — Continuous Deployment

### What it does

Runs only when a commit is pushed to the `main` branch. Streamlit Community Cloud watches your `main` branch and redeploys automatically — this workflow simply confirms the deployment is triggered.

### Full file

```yaml
name: CD

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Confirm deployment
        run: echo "Streamlit Community Cloud auto-deploys on push to main branch."
```

### How to trigger it

```bash
# Push to main branch
git push origin main
```

---

## `retrain.yml` — Automated Retraining

### What it does

Retrains the model when triggered, then commits updated metrics and version log back to the repo.

```
1. Install dependencies
2. dvc repro --no-run-cache    → rebuild full pipeline (preprocess → train → evaluate)
3. pytest                      → validate retrained model meets accuracy threshold
4. python -m src.models.version_log  → log new version to model_version_log.csv
5. git commit + push           → save evaluation_metrics.json, model_version_log.csv, dvc.lock
```

### Three triggers

**Trigger 1 — New data pushed:**
```bash
# Drop a new CSV into data/new_data/ and push
copy some_new_data.csv data\new_data\new_batch.csv
git add data/new_data/new_batch.csv
git commit -m "data: add new batch for retraining"
git push
# retrain.yml fires automatically
```

**Trigger 2 — Weekly schedule:**
Every Monday at 9am UTC — no action needed, GitHub runs it automatically.

**Trigger 3 — Manual button:**
Go to GitHub → Actions → Retrain → Run workflow → click green Run workflow button.

### Full file

```yaml
name: Retrain

on:
  push:
    paths:
      - "data/new_data/**"
  schedule:
    - cron: "0 9 * * 1"
  workflow_dispatch:

jobs:
  retrain:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run full DVC pipeline
        run: dvc repro --no-run-cache

      - name: Run tests to validate retrained model
        run: pytest -v --timeout=60

      - name: Log new model version
        run: python -m src.models.version_log

      - name: Commit updated artifacts and version log
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add artifacts/metrics/evaluation_metrics.json
          git add artifacts/metrics/model_version_log.csv
          git add dvc.lock
          git commit -m "model: auto-retrain $(date +'%Y-%m-%d') — triggered by ${{ github.event_name }} [skip ci]" || echo "nothing to commit"
          git push
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## Setup on GitHub

### Step 1 — Commit the workflow files

```bash
mkdir -p .github/workflows

# Save ci.yml, cd.yml, retrain.yml into .github/workflows/
git add .github/workflows/
git commit -m "ci: add CI, CD, and retrain GitHub Actions workflows"
git push
```

The moment you push, CI fires automatically. No setup required on GitHub.

### Step 2 — Check the Actions tab

Go to your repo on GitHub → click the **Actions** tab at the top.

You will see:
- **CI** — triggered by your push, currently running or complete
- **CD** — also triggered if you pushed to main
- **Retrain** — visible in the left sidebar, waiting for trigger

### Step 3 — Allow workflow write permissions (required for retrain.yml)

`retrain.yml` commits back to the repo using `GITHUB_TOKEN`. You need to enable write permissions:

1. Go to **Repo → Settings → Actions → General**
2. Scroll to **Workflow permissions**
3. Select **Read and write permissions**
4. Click **Save**

Without this, the commit step in `retrain.yml` will fail with a 403 error.

---

## How to verify each workflow is working

### Verify CI

```bash
# Make a small change and push
echo "# test" >> README.md
git add README.md
git commit -m "docs: trigger CI test"
git push
```

Go to **GitHub → Actions → CI**. You should see:
- Green tick next to each step: Checkout, Python setup, Install, Lint, Format, DVC, Tests
- Total run time around 3-5 minutes

If any step fails, click the run → click the failing step → read the log output.

### Verify CD

Push to main branch (same push above if you're on main). Go to **GitHub → Actions → CD**.

You should see:
- The "Confirm deployment" step prints the deployment message
- Green tick on the job

Also check **Streamlit Community Cloud** — the app should redeploy within 1-2 minutes of the push.

### Verify Retrain (manual trigger test)

1. Go to **GitHub → Actions → Retrain** in the left sidebar
2. Click **Run workflow** (top right, blue button)
3. Leave branch as `main`
4. Click the green **Run workflow** button

Wait 3-5 minutes. You should see all steps complete:

```
✓ Checkout code
✓ Set up Python
✓ Install dependencies
✓ Run full DVC pipeline      ← shows preprocess, train, evaluate output
✓ Run tests                  ← shows pytest results
✓ Log new model version      ← prints "logged v1.2"
✓ Commit updated artifacts   ← commits dvc.lock and metrics
```

After it completes:

```bash
git pull
cat artifacts/metrics/model_version_log.csv
# should show a new row added by the workflow
```

### Verify Retrain (data push trigger test)

```bash
# Windows
copy data\raw\ObesityDataSet_raw_and_data_sinthetic.csv data\new_data\trigger_test.csv

# Mac/Linux
cp data/raw/ObesityDataSet_raw_and_data_sinthetic.csv data/new_data/trigger_test.csv

git add data/new_data/trigger_test.csv
git commit -m "data: test retrain trigger with new data file"
git push
```

Go to **GitHub → Actions** — you should see `Retrain` start running automatically within seconds of the push.

---

## Common failures and fixes

| Failure | Cause | Fix |
|---|---|---|
| `ruff check` fails | Code has style errors | Run `ruff check src/ app/ tests/ --fix` locally, commit |
| `black --check` fails | Code not formatted | Run `black src/ app/ tests/` locally, commit |
| `dvc repro` fails | Import error or missing file | Check `src/config.py` paths, run `dvc repro` locally first |
| `pytest` fails | Test assertion error or import error | Run `pytest -v` locally, fix the failing test |
| Retrain commit fails 403 | Workflow doesn't have write permission | Settings → Actions → General → Read and write permissions |
| Retrain commit fails "nothing to commit" | Metrics didn't change | Normal — means the model output was identical, not an error |

---

## Reading the Actions logs

When a workflow fails, click the failing run → click the failing step name → read the output.

The most useful output comes from:
- **Run DVC pipeline** → shows `[ingest]`, `[preprocess]`, `[train]`, `[evaluate]` logs
- **Run tests** → shows which test passed or failed with full assertion errors
- **Commit updated artifacts** → shows git output or the "nothing to commit" message

A green tick on every step means the workflow passed completely.
