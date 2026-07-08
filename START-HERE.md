# Start Here

Not sure where to begin? Pick the path that matches your time and goal.

> **Using the subscriber study hub?** Read [HOW_TO_USE_THE_STUDY_HUB.md](HOW_TO_USE_THE_STUDY_HUB.md) first (10-min setup + learning loop).  
> **Do not follow folder numbers 00→25 in order.** Module folders are for organization; **stages** are the teaching sequence. Read [FOUNDATION_AND_JOB_READINESS.md](FOUNDATION_AND_JOB_READINESS.md) for exit gates, role paths, and job-market timing (especially **SQL early** for analyst/data scientist roles).

## If you have 30 minutes

Do your first ML project today.

1. Read [GETTING_STARTED.md](GETTING_STARTED.md)
2. Run the Iris classification project in `16-projects-beginner/project-02-iris-classification/`

You will load data, train models, and see results without reading the whole curriculum first.

## If you want the full map

See how all 26 modules and 23 projects fit together.

1. Read the [Learning Path Overview](README.md#learning-path-overview) in the main README
2. Open [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) for the visual stage-by-stage path

**Important:** **Module numbers** (00–25) are folder names. **Stages** in the README are the recommended learning order. They are not the same thing. Module 09 is not Stage 9.

## If you want a compressed schedule

You already know Python and want a faster week-by-week plan.

1. Read [QUICK_START.md](QUICK_START.md)
2. Treat it as an **accelerated path**, not the default timeline

The main README estimates **15–22 months** full-time for the complete path. QUICK_START is for learners who want to move quickly through foundations.

## If you are planning a career switch

1. Open [resources/career_roadmap_guide.md](resources/career_roadmap_guide.md)
2. Pick a target role (ML Engineer, Data Scientist, MLOps, and others)
3. Follow the modules listed for that role in the main README career table
4. Read [FOUNDATION_AND_JOB_READINESS.md](FOUNDATION_AND_JOB_READINESS.md) for **exit gates** and portfolio milestones

**Data Analyst / Data Scientist:** Start [Module 19 SQL](19-sql-database-fundamentals/README.md) right after Module 01, not after GenAI.

## Projects run in parallel (not only at the end)

| When | What |
|------|------|
| After Stage 2 (modules 02–05) | Begin [beginner projects](16-projects-beginner/README.md) (code included) |
| After Stages 3–4 | [Intermediate project briefs](17-projects-intermediate/README.md) |
| After deployment skills | [Advanced projects + capstones](18-projects-advanced/README.md) |

Stage 9 in the roadmap is **depth**, not “start projects here.”

## Setup (every path)

```bash
git clone https://github.com/NabidAlam/road-to-machine-learning.git
cd road-to-machine-learning
python -m venv ml-env
ml-env\Scripts\activate          # Windows
# source ml-env/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

Install dependencies from the **repository root**. Individual project folders may not have their own `requirements.txt`.

## Default learning order (short version)

| Stage | Modules | Topic |
|-------|---------|--------|
| Stage 0 | 00 | Python, math, environment |
| Stage 1 | 01 | NumPy, Pandas, visualization, EDA ([essential vs optional](01-python-for-data-science/README.md#essential-path-vs-optional-depth)) |
| **Stage 1.5** | **19** | **SQL & databases** — parallel with Stage 1–2 for analyst, DS, and ML engineer tracks |
| Stage 2 | 02–05 | Classical ML basics — read [ethics primer](resources/ethics_in_ml.md) before Module 04 |
| Stage 3 | 06–07 | Ensembles, feature engineering |
| Stage 4 | 08 | Unsupervised learning |
| Stage 5 | 09–10 | Neural networks, PyTorch/TensorFlow |
| Stage 6 | 11–12 | Computer vision, NLP |
| Stage 7 | 25 | Generative AI and LLMs |
| Stage 7.5 | 20–21 | Imbalanced data, explainability *(SQL is Stage 1.5, not here)* |
| Stage 8 | 13–14 | Deployment, MLOps |
| Stage 9 | 16–18 | Projects (start beginner projects after Stage 2) |
| Stage 10 | 22–24 | RL, graphs, audio (electives) |

**Time series:** Module **15**, intermediate project 6, and advanced project 3 overlap — pick **one** path: [TIME_SERIES_LEARNING_PATH.md](TIME_SERIES_LEARNING_PATH.md).

## Projects: what to expect

| Tier | Delivery | Notes |
|------|----------|--------|
| Beginner (6) | Code included | Runnable scripts and notebooks in most projects |
| Intermediate (8) | Brief + `starter.py` skeleton | README + runnable scaffold in each `project-*/` folder |
| Advanced (9) | Brief + starter guidance | README instructions. Capstone-style builds |

## Where to go next

- Main hub: [README.md](README.md)
- **Study hub users:** [HOW_TO_USE_THE_STUDY_HUB.md](HOW_TO_USE_THE_STUDY_HUB.md)
- First project walkthrough: [GETTING_STARTED.md](GETTING_STARTED.md)
- Visual roadmap: [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md)
- Foundation gates & job market: [FOUNDATION_AND_JOB_READINESS.md](FOUNDATION_AND_JOB_READINESS.md)
- Time series (pick one path): [TIME_SERIES_LEARNING_PATH.md](TIME_SERIES_LEARNING_PATH.md)
- Study Hub (read lessons in browser): [Nabid In Motion](https://github.com/NabidAlam/nabidinmotion)
