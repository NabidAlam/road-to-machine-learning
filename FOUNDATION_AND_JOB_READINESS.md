# Foundation & Job Market Readiness

Expert review guide for learners, instructors, and career switchers. This document defines **what “good enough” looks like** before you advance stages, and how the curriculum maps to **hireable skills** in Germany, the US, and remote-first teams.

**Read this with:** [START-HERE.md](START-HERE.md) · [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) · [Career Roadmap Guide](resources/career_roadmap_guide.md)

---

## Professor’s verdict (short)

| Area | Rating | Notes |
|------|--------|-------|
| **Breadth** | Strong | Tabular ML → DL → GenAI → deploy; 23 projects; system-design side track |
| **Foundation depth** | Strong | Module 00 math + Python is above typical bootcamp level |
| **Hands-on ratio** | Mixed | Beginner projects ship code; intermediate/advanced are brief-led |
| **Job-market alignment** | Good with gaps | SQL and portfolio gates come late in the *default* stage order |
| **Assessment** | Self-paced only | No autograded mastery checks; use exit criteria below |

**Bottom line:** Follow **stages and role paths**, not folder numbers 00→25 in order. Start **projects in parallel** after Stage 2. Treat SQL as **Stage 1.5** if you target analyst or data scientist roles.

---

## Critical rule: stages ≠ folder numbers

| Mistake | Fix |
|---------|-----|
| Opening folders 00, 01, 02… in numeric order only | Use [stage table in START-HERE](START-HERE.md#default-learning-order-short-version) |
| Doing deployment (13–14) before SQL (19) on an analyst track | Follow [Data Analyst path](resources/career_roadmap_guide.md#data-analyst): 00 → 01 → **19** early |
| Waiting until “Stage 9” to build projects | Start **beginner projects after Stage 2** while continuing modules |
| Reading entire 1,900-line guides in one sitting | Use learning objectives + exit criteria; skim advanced files later |

---

## Foundation exit gates (do not skip)

### Gate A, after Module 00 (Stage 0)

You are ready for **Module 01** when you can **without copying from a tutorial**:

- [ ] Write functions, classes, and file I/O (JSON or CSV) in Python
- [ ] Explain Big-O for a simple loop vs nested loop
- [ ] Multiply matrices and explain why linear algebra matters for ML
- [ ] Compute mean, variance, and interpret a normal distribution
- [ ] Explain gradient descent in plain language
- [ ] Run Jupyter and install packages in a virtual environment
- [ ] **Capstone:** Complete the [movie script generator](00-prerequisites/01-python-basics.md) or [NumPy NN tutorial](00-prerequisites/prerequisites-project-tutorial.md)

**Job relevance:** Every technical interview assumes this baseline. Weak Python here causes attrition in Module 01.

---

### Gate B, after Module 01 (Stage 1)

You are ready for **Module 02 (ML)** when you can:

- [ ] Load a CSV with Pandas, handle missing values, and summarize dtypes
- [ ] Build at least **three** plot types (distribution, relationship, categorical)
- [ ] Complete a short EDA narrative: question → data → chart → insight
- [ ] Use NumPy for vectorized operations (avoid Python loops on large arrays)

**Parallel track (analyst / data scientist):** Start [Module 19 SQL](19-sql-database-fundamentals/README.md) **now**, not after GenAI. Most EU and US job postings expect SQL alongside Python.

**Job relevance:** Junior data roles are won on **EDA + SQL + communication**, not on neural networks.

---

### Gate C, after Modules 02–05 (Stage 2)

You are **job-market foundation-ready** for entry-level ML/tabular roles when you can:

- [ ] Build train/validation/test splits and explain **data leakage**
- [ ] Train regression and classification models with scikit-learn
- [ ] Choose metrics (RMSE vs F1 vs PR-AUC) for the business problem
- [ ] Tune hyperparameters with cross-validation
- [ ] Discuss fairness at a basic level ([Module 04](04-supervised-learning-classification/README.md))
- [ ] **Portfolio:** Finish at least **two** [beginner projects](16-projects-beginner/README.md) (e.g. house prices + Titanic)

**Do not wait** for deep learning or GenAI to start GitHub portfolio work.

---

### Gate D: Production-ready (Stages 7–8 + projects)

Target **ML Engineer / MLOps** when you can:

- [ ] Expose a model via FastAPI or Flask with Docker
- [ ] Track experiments (MLflow or W&B) and version a model artifact
- [ ] Explain monitoring and drift at a high level
- [ ] **Portfolio:** One [intermediate](17-projects-intermediate/README.md) + one [advanced](18-projects-advanced/README.md) project with README, tests, and deploy link

**Side track:** Complete [system-design](system-design/README.md) lessons 00–14 before senior ML engineer interviews.

---

## Role-specific “minimum viable” paths

| Role | Minimum modules | SQL timing | First portfolio milestone |
|------|-----------------|------------|---------------------------|
| **Data Analyst** | 00, 01, 19, 20, 21 | After 01 | Streamlit dashboard project |
| **Data Scientist** | 00–08, 15, 19–21 | After 05 or parallel with 06–07 | 3 beginner + 1 intermediate |
| **ML Engineer** | 00–10, 13–14, 19–21 | After 01 | Deployed API + MLflow run |
| **LLM Engineer** | 00–01, 05, 09–10, 12, 25, 13–14, 19 | After 01 | RAG app with eval notes |

Full tables: [career_roadmap_guide.md](resources/career_roadmap_guide.md)

---

## Known curriculum gaps (and workarounds)

| Gap | Why it matters | Workaround in this repo |
|-----|----------------|-------------------------|
| No autograded exams | Hard to prove mastery | Use exit gates above; [time-series exercises](15-time-series-analysis/exercises/README.md) as template |
| SQL late in default stages | Analyst jobs need it early | Stage 1.5 parallel path to Module 19 |
| Intermediate projects lack starter code | Drop-off for guided learners | Pair each brief with a Kaggle dataset + notebook skeleton you create |
| MLOps mostly conceptual | Interviews ask hands-on | [deployment.md](13-model-deployment/deployment.md) + [mlops.md](14-mlops-basics/mlops.md) labs + [Docker tutorial](resources/docker_tutorial.md) |
| Ethics not required | EU AI Act, US enterprise | Read [ethics_in_ml.md](resources/ethics_in_ml.md) before Module 04 fairness section |
| Communication not sequenced | DS hires for storytelling | [stakeholder_communication.md](resources/stakeholder_communication.md) after first EDA project |

---

## Recommended weekly rhythm (working professional)

| % | Activity |
|---|----------|
| **50%** | Code (guides, exercises, projects) |
| **25%** | One module lesson (read + reproduce examples) |
| **15%** | Portfolio / GitHub (README, screenshots, deploy) |
| **10%** | Review (quick-ref sheets, spaced repetition) |

---

## Instructor / self-check questions

Before advancing a stage, ask:

1. Can the learner **explain** the last concept to a non-technical colleague?
2. Can they **reproduce** the core notebook without the guide open?
3. Is there a **GitHub artifact** proving the skill?
4. For EU learners: can they discuss **data minimization** and model limitations?

---

## Next steps

1. New learner → [GETTING_STARTED.md](GETTING_STARTED.md)
2. Career switch → [career_roadmap_guide.md](resources/career_roadmap_guide.md)
3. Compressed timeline → [QUICK_START.md](QUICK_START.md)
4. Portfolio checklist → [career_portfolio.md](resources/career_portfolio.md)
5. Interview prep → [interview_prep.md](resources/interview_prep.md)
