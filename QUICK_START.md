# Quick Start Guide

Get the environment running in about 5 minutes. For full navigation, read [START-HERE.md](START-HERE.md) first — **follow stages, not folder numbers 00→25**.

> **Note:** The week-by-week schedule below is an **accelerated path** for learners who already study full-time. The main [README](README.md) estimates **15–22 months** full-time for the complete curriculum.

## Step 1: Clone the Repository

```bash
git clone https://github.com/NabidAlam/road-to-machine-learning.git
cd road-to-machine-learning
```

If this folder lives inside the [Nabid In Motion study hub](https://github.com/NabidAlam/nabidinmotion) as a submodule, run commands from `road-to-machine-learning/` inside that parent repo.

## Step 2: Set Up Environment

### Option A: Using Anaconda

```bash
conda create -n ml-env python=3.11
conda activate ml-env
pip install -r requirements.txt
```

### Option B: Using Python venv

```bash
python -m venv ml-env
ml-env\Scripts\activate          # Windows
# source ml-env/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

## Step 3: Install Jupyter

```bash
pip install jupyter notebook
```

## Step 4: Start Learning

| Your background | Start here |
|-----------------|------------|
| Complete beginner | [GETTING_STARTED.md](GETTING_STARTED.md) → Iris project |
| Know Python | Module 01 → Module 02 |
| Know ML basics | Pick modules you need → projects in module 16–18 |

## Step 5: First Project

1. Go to `16-projects-beginner/project-02-iris-classification/`
2. Follow the README or run `python iris_classification.py`
3. Move to house price prediction when ready

## Accelerated schedule (optional)

| Weeks | Focus |
|-------|--------|
| 1–2 | Modules 00–01; start **Module 19 SQL** in parallel (Stage 1.5) |
| 3–4 | Modules 02–04 |
| 5–6 | Modules 05–07 |
| 7–8 | 2–3 beginner projects |

Then continue with deep learning (modules 09–12), GenAI (module 25), essential skills (20–21), and deployment (modules 13–14). See [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md).

## Need Help?

- [START-HERE.md](START-HERE.md): pick your path
- [README.md](README.md): full curriculum hub
- [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md): visual stage map

**Ready?** Open [GETTING_STARTED.md](GETTING_STARTED.md) or `00-prerequisites/README.md`.
