# How to Use the Study Hub for Maximum Learning

This guide is for the **[Nabid In Motion Study Hub](https://nabidinmotion.github.io)**, the subscriber site that syncs this curriculum. It explains **how to learn here**, not what to learn (see [START-HERE.md](START-HERE.md) for that).

**Time to read:** ~8 minutes · **Time to set up:** ~10 minutes once

---

## What this site is (and is not)

| This site is | This site is not |
|--------------|------------------|
| A reader for 265+ synced lessons | A replacement for writing code on your machine |
| Local progress tracking (no account) | An autograded course or certificate program |
| Career-path filtering + project tracker | A linear playlist you finish top to bottom |

**Rule:** Read lessons here. **Run code locally** (clone the [GitHub repo](https://github.com/NabidAlam/road-to-machine-learning)). Mark a lesson complete only after you reproduce the core examples.

---

## 10-minute setup (do once)

1. **Pick a career role** on the homepage (`Career Path` section).  
   The hub filters modules to what matters for that job. You can switch roles anytime.

2. **Set weekly goals** on the homepage progress card:  
   - **Reading goal**, e.g. 3–5 lessons/week (part-time) or 8–12 (full-time)  
   - **Focus goal**, e.g. 150–300 minutes of deep work/week

3. **Open “Today’s practice path”** (homepage) or **Continue Learning**, start the next lesson in your filtered path.

4. **Clone the curriculum repo** on your computer:

   ```bash
   git clone https://github.com/NabidAlam/road-to-machine-learning.git
   cd road-to-machine-learning
   python -m venv ml-env
   ml-env\Scripts\activate          # Windows
   pip install -r requirements.txt
   ```

5. **Bookmark these three guides** (read on-site or on GitHub):  
   - [START-HERE.md](START-HERE.md), stage order (not folder numbers)  
   - [FOUNDATION_AND_JOB_READINESS.md](FOUNDATION_AND_JOB_READINESS.md), exit gates before advancing  
   - [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md), visual milestones

---

## The learning loop (every session)

Use this loop for **maximum retention and portfolio output**:

```
Read → Reproduce → Reflect → Build → Review
```

| Step | On the study hub | On your machine |
|------|------------------|-----------------|
| **Read** | Open one lesson; skim objectives first |, |
| **Reproduce** |, | Re-type examples without the guide open |
| **Reflect** | Mark **Confused** if anything is fuzzy; add a bookmark | One sentence in a notes file: “I learned X” |
| **Build** | Update **Project tracker** (separate from lesson checkboxes) | Run `starter.py` or project scripts in `16–18/` |
| **Review** | Check weekly goals; revisit **Confused** lessons | Spaced repetition: quick-ref sheets |

**Do not** check a lesson “done” after skimming. Checking done means: *I could explain this and run the code.*

---

## How to use each hub feature

### Lessons & modules

- Open from **Modules** (filter by phase) or **Continue Learning**.  
- Each module README is the overview; numbered lessons are the depth.  
- **Module 01:** follow the [essential path](01-python-for-data-science/README.md#essential-path-vs-optional-depth), not all 15 lessons before ML.  
- **SQL (Module 19):** [Stage 1.5](START-HERE.md#default-learning-order-short-version), parallel with Modules 01–02 for most roles.

### Career path filter

- Narrows the sidebar to role-relevant modules.  
- Does **not** change stage order inside those modules.  
- Full role tables: [Career Roadmap Guide](resources/career_roadmap_guide.md).

### Progress checkboxes

- Saved in **localStorage** only (this browser).  
- Use **Export progress** / **Import progress** on the homepage before switching devices.  
- Lesson progress ≠ project completion, track both.

### Project tracker

- **23 projects** in three tiers; status is independent of lesson checkboxes.  
- Start **beginner projects after Stage 2** (modules 02–05), not after finishing all modules.  
- Intermediate projects include `starter.py` scaffolds, run locally after downloading data.

### Bookmarks & Confused

- **Bookmark**, return later (references, cheat sheets).  
- **Confused**, your personal review queue; revisit within 48 hours for best retention.

### Practice path

- Homepage panel with the next 1–3 suggested steps (lesson + optional project).  
- Follow it when overwhelmed; ignore it when you are deep in a project sprint.

### Videos

- YouTube opens in a new tab (no embedded players).  
- Pair videos with the matching module lesson, video for intuition, lesson for code.

---

## Weekly rhythm (working professional)

| Share of time | Activity |
|---------------|----------|
| **50%** | Code (examples, exercises, projects) |
| **25%** | One hub lesson (read + reproduce) |
| **15%** | Portfolio (README, screenshots, GitHub) |
| **10%** | Review (confused lessons, quick-reference) |

Adjust goals on the homepage to match your real schedule. Consistency beats marathon sessions.

---

## Anti-patterns (avoid these)

| Mistake | Fix |
|---------|-----|
| Browsing modules 00 → 25 in folder order | Follow [stages in START-HERE](START-HERE.md#default-learning-order-short-version) |
| Reading 1,900-line guides in one sitting | Objectives + one section + code; skim advanced files later |
| Only using the study hub, never cloning GitHub | Clone repo; projects and `starter.py` files run locally |
| Checking all lessons done, zero GitHub repos | Project tracker + public portfolio |
| Waiting for “advanced modules” before projects | Beginner projects after Stage 2 |
| Doing module 15 + TS project 6 + advanced TS project 3 | [Pick one time-series path](TIME_SERIES_LEARNING_PATH.md) |

---

## When to read here vs GitHub

| Use the study hub | Use GitHub |
|-------------------|------------|
| Reading lessons with progress tracking | Running Python projects and scripts |
| Career filter + practice path | Editing notebooks, committing portfolio work |
| Search across synced content | Issues, PRs, contributing |
| Quick session on phone/tablet | Full dev environment |

---

## Suggested first week

| Day | Hub | Local |
|-----|-----|-------|
| 1 | Pick role; set weekly goals; read [Getting Started](GETTING_STARTED.md) on-site | Clone repo; run Iris project |
| 2–3 | Module 00 overview + one math/Python lesson | Reproduce examples in Jupyter |
| 4–5 | Module 01 essential: NumPy + Pandas lessons | Same, type code yourself |
| 6 | Start Module 19 SQL if analyst/DS track | SQLZoo or lesson exercises |
| 7 | Review **Confused** items; export progress backup | Push a GitHub repo with week-1 notes |

---

## Before you advance a stage

Use [exit gates in FOUNDATION_AND_JOB_READINESS.md](FOUNDATION_AND_JOB_READINESS.md). Short version:

- **After Stage 0:** Python + math baseline without copying tutorials  
- **After Stage 1:** EDA + plots + Pandas fluency  
- **After Stage 2:** Two beginner projects on GitHub  

---

## Need help?

- **What to learn next:** [START-HERE.md](START-HERE.md)  
- **Job-market timing:** [FOUNDATION_AND_JOB_READINESS.md](FOUNDATION_AND_JOB_READINESS.md)  
- **Site issues:** [nabidinmotion GitHub](https://github.com/NabidInMotion/nabidinmotion.github.io)  
- **Curriculum source:** [road-to-machine-learning](https://github.com/NabidAlam/road-to-machine-learning)

---

**Start now:** [Study Hub homepage](https://nabidinmotion.github.io) → pick your role → **Continue Learning**.
