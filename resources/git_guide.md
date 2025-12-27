# Complete Git & GitHub Guide

A comprehensive guide to Git and GitHub with commands, explanations, outputs, logic, practice exercises, and solutions. Perfect for learning, revision, and daily reference.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Basic Git Commands](#basic-git-commands)
- [Working with Branches](#working-with-branches)
- [Remote Repositories & GitHub](#remote-repositories--github)
- [Advanced Git Operations](#advanced-git-operations)
- [Git Workflows](#git-workflows)
- [Practice Exercises](#practice-exercises)
- [Common Scenarios](#common-scenarios)
- [GitHub-Specific Features](#github-specific-features)
- [Resources & Documentation](#resources--documentation)

---

## Introduction

### What is Git?

**Git** is a distributed version control system that tracks changes in your code over time. It allows you to:
- Save snapshots of your work (commits)
- Work on different features simultaneously (branches)
- Collaborate with others without conflicts
- Revert to previous versions if needed

### Why Use Git?

1. **Version Control**: Track every change in your project
2. **Backup**: Your entire project history is saved
3. **Collaboration**: Multiple people can work on the same project
4. **Experimentation**: Try new features without breaking existing code
5. **Professional Standard**: Industry standard for software development

### Git vs GitHub

- **Git**: The version control software (runs on your computer)
- **GitHub**: A cloud-based hosting service for Git repositories (online platform)

---

## Getting Started

### Installation

**Windows:**
```bash
# Download from https://git-scm.com/download/win
# Or use package manager
winget install Git.Git
```

**Mac:**
```bash
# Using Homebrew
brew install git

# Or download from https://git-scm.com/download/mac
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install git

# Fedora
sudo dnf install git
```

### Initial Setup

```bash
# Set your name (used in commits)
git config --global user.name "Your Name"

# Set your email (used in commits)
git config --global user.email "your.email@example.com"

# Set default branch name
git config --global init.defaultBranch main

# Set default editor (optional)
git config --global core.editor "code --wait"  # VS Code
# git config --global core.editor "nano"       # Nano
# git config --global core.editor "vim"        # Vim

# Check your configuration
git config --list
```

**Output:**
```
user.name=Your Name
user.email=your.email@example.com
init.defaultBranch=main
core.editor=code --wait
```

**Why:** These settings identify you as the author of commits. The `--global` flag applies to all repositories on your computer.

### Verify Installation

```bash
git --version
```

**Output:**
```
git version 2.42.0
```

---

## Basic Git Commands

### 1. Initialize a Repository

```bash
# Create a new Git repository in current directory
git init
```

**Output:**
```
Initialized empty Git repository in /path/to/your/project/.git/
```

**What Happens:**
- Creates a hidden `.git` folder that stores all Git data
- Your directory is now a Git repository

**Why:** This tells Git to start tracking changes in this directory.

**Example:**
```bash
mkdir my-project
cd my-project
git init
```

### 2. Check Repository Status

```bash
git status
```

**Output (clean repository):**
```
On branch main
nothing to commit, working tree clean
```

**Output (with changes):**
```
On branch main
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   README.md

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        newfile.py

no changes added to commit (use "git add" to stage changes)
```

**Why:** Shows you what files have changed and what Git is tracking.

### 3. Add Files to Staging Area

```bash
# Add a specific file
git add filename.py

# Add all files in current directory
git add .

# Add all files matching a pattern
git add *.py

# Add all files in a directory
git add src/
```

**Output:** (No output means success)

**What Happens:**
- Files move from "working directory" to "staging area"
- Staged files are ready to be committed

**Why:** Git uses a three-stage system:
1. **Working Directory**: Your files as you edit them
2. **Staging Area**: Files you've marked to be committed
3. **Repository**: Committed snapshots

**Example:**
```bash
echo "# My Project" > README.md
git add README.md
git status
```

**Output:**
```
On branch main
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   README.md
```

### 4. Commit Changes

```bash
# Commit with a message
git commit -m "Add README file"

# Commit with a detailed message (opens editor)
git commit

# Add and commit in one step (only for tracked files)
git commit -am "Update existing files"
```

**Output:**
```
[main (root-commit) a1b2c3d] Add README file
 1 file changed, 1 insertion(+)
 create mode 100644 README.md
```

**What Happens:**
- Creates a snapshot of your staged files
- Assigns a unique commit hash (a1b2c3d)
- Records author, date, and message

**Why:** Commits are like save points. You can return to any commit later.

**Best Practices:**
- Write clear, descriptive commit messages
- Make small, logical commits
- Use present tense: "Add feature" not "Added feature"

**Example:**
```bash
git commit -m "Add user authentication feature"
```

### 5. View Commit History

```bash
# View commit history
git log

# One-line format
git log --oneline

# Graph view
git log --graph --oneline --all

# Last N commits
git log -5

# With file changes
git log --stat

# Search commits
git log --grep="bug fix"
```

**Output (git log --oneline):**
```
a1b2c3d (HEAD -> main) Add README file
d4e5f6a Initial commit
```

**Why:** See what changes were made and when. Useful for debugging and understanding project history.

### 6. View File Differences

```bash
# See changes in working directory
git diff

# See changes in staging area
git diff --staged

# See changes for specific file
git diff filename.py

# See changes between commits
git diff commit1 commit2
```

**Output:**
```
diff --git a/README.md b/README.md
index 1234567..abcdefg 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,2 @@
 # My Project
+This is a great project!
```

**Why:** Understand exactly what changed before committing.

### 7. Discard Changes

```bash
# Discard changes in working directory (unstaged)
git restore filename.py

# Discard all unstaged changes
git restore .

# Unstage a file (keep changes)
git restore --staged filename.py

# Old syntax (still works)
git checkout -- filename.py
```

**Why:** Undo changes you don't want to keep. **Warning:** This permanently discards changes!

**Example:**
```bash
# Accidentally edited a file
git restore README.md  # Reverts to last committed version
```

### 8. Remove Files

```bash
# Remove file from Git and filesystem
git rm filename.py

# Remove file from Git only (keep local file)
git rm --cached filename.py

# Remove directory
git rm -r directory/
```

**Output:**
```
rm 'filename.py'
```

**Why:** Tell Git to stop tracking a file. The file is removed in the next commit.

---

## Working with Branches

### What are Branches?

Branches are parallel versions of your code. They allow you to:
- Work on features without affecting main code
- Experiment safely
- Collaborate without conflicts

### 1. List Branches

```bash
# List local branches
git branch

# List all branches (local + remote)
git branch -a

# List remote branches
git branch -r
```

**Output:**
```
* main
  feature-login
  feature-dashboard
```

**Why:** See what branches exist and which one you're on (marked with *).

### 2. Create a Branch

```bash
# Create a new branch
git branch feature-name

# Create and switch to branch
git checkout -b feature-name

# Modern way (Git 2.23+)
git switch -c feature-name
```

**Output:** (No output means success)

**Why:** Create a separate line of development for a feature or bug fix.

**Example:**
```bash
git checkout -b feature-user-authentication
```

### 3. Switch Branches

```bash
# Switch to existing branch
git checkout branch-name

# Modern way
git switch branch-name

# Switch and create if doesn't exist
git checkout -b new-branch
```

**Why:** Move between different versions of your code.

**Example:**
```bash
git switch main
```

### 4. Merge Branches

```bash
# Merge branch into current branch
git merge feature-name

# Merge with commit message
git merge feature-name -m "Merge feature branch"
```

**Output (fast-forward merge):**
```
Updating a1b2c3d..d4e5f6a
Fast-forward
 README.md | 2 ++
 1 file changed, 2 insertions(+)
```

**Output (merge commit):**
```
Merge made by the 'recursive' strategy.
 README.md | 2 ++
 1 file changed, 2 insertions(+)
```

**Why:** Combine changes from one branch into another.

**Example:**
```bash
git switch main
git merge feature-login
```

### 5. Delete Branches

```bash
# Delete local branch
git branch -d branch-name

# Force delete (even if not merged)
git branch -D branch-name

# Delete remote branch
git push origin --delete branch-name
```

**Why:** Clean up branches you no longer need.

### 6. Rename Branch

```bash
# Rename current branch
git branch -m new-name

# Rename specific branch
git branch -m old-name new-name
```

**Why:** Fix typos or update branch names to match conventions.

---

## Remote Repositories & GitHub

### 1. Add Remote Repository

```bash
# Add remote (usually done once)
git remote add origin https://github.com/username/repo.git

# Add remote with custom name
git remote add upstream https://github.com/original/repo.git

# View remotes
git remote -v
```

**Output:**
```
origin  https://github.com/username/repo.git (fetch)
origin  https://github.com/username/repo.git (push)
```

**Why:** Connect your local repository to GitHub. `origin` is the default name for your main remote.

**Example:**
```bash
git remote add origin https://github.com/NabidAlam/road-to-machine-learning.git
```

### 2. Clone a Repository

```bash
# Clone via HTTPS
git clone https://github.com/username/repo.git

# Clone via SSH
git clone git@github.com:username/repo.git

# Clone to specific directory
git clone https://github.com/username/repo.git my-project

# Clone specific branch
git clone -b branch-name https://github.com/username/repo.git
```

**Output:**
```
Cloning into 'repo'...
remote: Enumerating objects: 100, done.
remote: Counting objects: 100% (100/100), done.
remote: Compressing objects: 100% (80/80), done.
remote: Total 100 (delta 20), reused 100 (delta 20), pack-reused 0
Receiving objects: 100% (100/100), 50.00 KiB | 1.00 MiB/s, done.
Resolving deltas: 100% (20/20), done.
```

**Why:** Download an existing repository from GitHub to your computer.

### 3. Push to Remote

```bash
# Push current branch to origin
git push origin main

# Push and set upstream (first time)
git push -u origin main

# Push all branches
git push --all origin

# Push tags
git push --tags origin
```

**Output:**
```
Enumerating objects: 5, done.
Counting objects: 100% (5/5), done.
Delta compression using up to 8 threads
Compressing objects: 100% (3/3), done.
Writing objects: 100% (3/3), 300 bytes | 300.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
To https://github.com/username/repo.git
   a1b2c3d..d4e5f6a  main -> main
```

**Why:** Upload your local commits to GitHub so others can see them.

**Example:**
```bash
git push -u origin main  # First push
git push                 # Subsequent pushes (Git remembers upstream)
```

### 4. Pull from Remote

```bash
# Pull latest changes
git pull

# Pull from specific remote and branch
git pull origin main

# Fetch and merge separately
git fetch origin
git merge origin/main
```

**Output:**
```
remote: Enumerating objects: 5, done.
remote: Counting objects: 100% (5/5), done.
remote: Compressing objects: 100% (3/3), done.
remote: Total 3 (delta 2), reused 0 (delta 0), pack-reused 0
Unpacking objects: 100% (3/3), done.
From https://github.com/username/repo
   a1b2c3d..d4e5f6a  main       -> origin/main
Updating a1b2c3d..d4e5f6a
Fast-forward
 README.md | 2 ++
 1 file changed, 2 insertions(+)
```

**Why:** Download and merge changes from GitHub into your local repository.

### 5. Fetch from Remote

```bash
# Fetch all remotes
git fetch

# Fetch specific remote
git fetch origin

# Fetch all branches
git fetch --all
```

**Why:** Download changes without merging. Lets you review before merging.

**Example:**
```bash
git fetch origin
git log origin/main  # See what's new
git merge origin/main  # Merge when ready
```

---

## Advanced Git Operations

### 1. Stash Changes

```bash
# Save current changes temporarily
git stash

# Stash with message
git stash save "Work in progress"

# List stashes
git stash list

# Apply most recent stash
git stash apply

# Apply and remove stash
git stash pop

# Apply specific stash
git stash apply stash@{0}

# Delete stash
git stash drop stash@{0}

# Clear all stashes
git stash clear
```

**Output (git stash list):**
```
stash@{0}: WIP on main: a1b2c3d Add feature
stash@{1}: WIP on main: d4e5f6a Fix bug
```

**Why:** Temporarily save uncommitted changes so you can switch branches or pull updates.

**Example:**
```bash
# Working on feature, need to switch branches
git stash
git switch main
# Do something on main
git switch feature
git stash pop  # Resume work
```

### 2. Amend Last Commit

```bash
# Add changes to last commit
git commit --amend

# Change commit message only
git commit --amend -m "New message"

# Add file to last commit
git add forgotten-file.py
git commit --amend --no-edit
```

**Why:** Fix mistakes in your last commit without creating a new commit.

**Warning:** Don't amend commits that have been pushed to shared repositories!

### 3. Reset Commits

```bash
# Soft reset (keep changes staged)
git reset --soft HEAD~1

# Mixed reset (keep changes, unstaged) - DEFAULT
git reset HEAD~1
git reset --mixed HEAD~1

# Hard reset (discard all changes) - DANGEROUS
git reset --hard HEAD~1

# Reset to specific commit
git reset --hard a1b2c3d
```

**Why:** Undo commits. Use carefully!

**Example:**
```bash
# Undo last commit, keep changes
git reset --soft HEAD~1
```

### 4. Revert Commits

```bash
# Create new commit that undoes changes
git revert HEAD

# Revert specific commit
git revert a1b2c3d

# Revert without committing
git revert --no-commit HEAD
```

**Why:** Safely undo changes by creating a new commit. Better than reset for shared branches.

**Example:**
```bash
git revert HEAD  # Undoes last commit
```

### 5. Cherry-pick

```bash
# Apply specific commit to current branch
git cherry-pick a1b2c3d

# Cherry-pick multiple commits
git cherry-pick commit1 commit2

# Cherry-pick without committing
git cherry-pick --no-commit a1b2c3d
```

**Why:** Apply a commit from one branch to another without merging the entire branch.

**Example:**
```bash
git switch main
git cherry-pick feature-branch  # Apply one commit from feature branch
```

### 6. Rebase

```bash
# Rebase current branch onto another
git rebase main

# Interactive rebase (last 3 commits)
git rebase -i HEAD~3

# Continue after resolving conflicts
git rebase --continue

# Abort rebase
git rebase --abort
```

**Why:** Replay your commits on top of another branch, creating a linear history.

**Warning:** Don't rebase commits that have been pushed to shared repositories!

**Example:**
```bash
git switch feature
git rebase main  # Replay feature commits on top of main
```

### 7. Tags

```bash
# Create lightweight tag
git tag v1.0.0

# Create annotated tag
git tag -a v1.0.0 -m "Release version 1.0.0"

# List tags
git tag

# Show tag details
git show v1.0.0

# Delete tag
git tag -d v1.0.0

# Push tags to remote
git push origin v1.0.0
git push --tags
```

**Why:** Mark important points in history (like releases).

**Example:**
```bash
git tag -a v1.0.0 -m "First stable release"
git push origin v1.0.0
```

---

## Git Workflows

### Feature Branch Workflow

```bash
# 1. Create feature branch
git checkout -b feature-new-feature

# 2. Make changes and commit
git add .
git commit -m "Add new feature"

# 3. Push to remote
git push -u origin feature-new-feature

# 4. Create Pull Request on GitHub

# 5. After PR is merged, clean up
git checkout main
git pull
git branch -d feature-new-feature
```

**Why:** Standard workflow for adding features. Keeps main branch stable.

### Fork and Clone Workflow (Contributing)

```bash
# 1. Fork repository on GitHub

# 2. Clone your fork
git clone https://github.com/your-username/repo.git

# 3. Add original repo as upstream
git remote add upstream https://github.com/original-owner/repo.git

# 4. Create feature branch
git checkout -b feature-contribution

# 5. Make changes and commit
git add .
git commit -m "Add contribution"

# 6. Push to your fork
git push -u origin feature-contribution

# 7. Create Pull Request on GitHub

# 8. Keep your fork updated
git fetch upstream
git merge upstream/main
```

**Why:** Contribute to projects you don't own.

---

## Practice Exercises

### Exercise 1: Basic Workflow

**Task:** Create a repository, add files, and make commits.

**Solution:**
```bash
# 1. Create directory and initialize
mkdir practice-repo
cd practice-repo
git init

# 2. Create a file
echo "# Practice Project" > README.md

# 3. Add and commit
git add README.md
git commit -m "Add README"

# 4. Check status
git status

# 5. View history
git log --oneline
```

**Expected Output:**
```
Initialized empty Git repository in practice-repo/.git/
[main (root-commit) a1b2c3d] Add README
 1 file changed, 1 insertion(+)
On branch main
nothing to commit, working tree clean
a1b2c3d (HEAD -> main) Add README
```

### Exercise 2: Branching

**Task:** Create branches, make changes, and merge.

**Solution:**
```bash
# 1. Create feature branch
git checkout -b feature-login

# 2. Create file on branch
echo "def login():" > login.py
git add login.py
git commit -m "Add login function"

# 3. Switch back to main
git checkout main

# 4. Merge feature branch
git merge feature-login

# 5. Delete feature branch
git branch -d feature-login
```

**Expected Output:**
```
Switched to a new branch 'feature-login'
[feature-login d4e5f6a] Add login function
 1 file changed, 1 insertion(+)
Switched to branch 'main'
Updating a1b2c3d..d4e5f6a
Fast-forward
 login.py | 1 +
 1 file changed, 1 insertion(+)
Deleted branch feature-login
```

### Exercise 3: Undoing Changes

**Task:** Make a mistake and fix it.

**Solution:**
```bash
# 1. Create file
echo "Important content" > important.txt
git add important.txt
git commit -m "Add important file"

# 2. Accidentally delete content
echo "" > important.txt

# 3. Restore file
git restore important.txt

# 4. Verify
cat important.txt
```

**Expected Output:**
```
[main a1b2c3d] Add important file
 1 file changed, 1 insertion(+)
Important content
```

### Exercise 4: Stashing

**Task:** Use stash to switch branches with uncommitted changes.

**Solution:**
```bash
# 1. Create branch
git checkout -b feature-dashboard

# 2. Start working
echo "dashboard code" > dashboard.py

# 3. Need to switch branches
git stash save "Dashboard work in progress"

# 4. Switch to main
git checkout main

# 5. Do something on main
echo "main update" >> README.md
git add README.md
git commit -m "Update README"

# 6. Return to feature branch
git checkout feature-dashboard

# 7. Resume work
git stash pop
```

**Expected Output:**
```
Saved working directory and index state On feature-dashboard: Dashboard work in progress
Switched to branch 'main'
[main d4e5f6a] Update README
 1 file changed, 1 insertion(+)
Switched to branch 'feature-dashboard'
On branch feature-dashboard
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   dashboard.py
Dropped refs/stash@{0}
```

### Exercise 5: GitHub Workflow

**Task:** Push repository to GitHub and collaborate.

**Solution:**
```bash
# 1. Create repository on GitHub (do this on GitHub website)

# 2. Add remote
git remote add origin https://github.com/username/practice-repo.git

# 3. Push to GitHub
git push -u origin main

# 4. Make changes locally
echo "New feature" > feature.py
git add feature.py
git commit -m "Add new feature"

# 5. Push changes
git push

# 6. Pull changes (if someone else pushed)
git pull
```

**Expected Output:**
```
Enumerating objects: 3, done.
Counting objects: 100% (3/3), done.
Writing objects: 100% (3/3), 250 bytes | 250.00 KiB/s, done.
To https://github.com/username/practice-repo.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'
[main a1b2c3d] Add new feature
 1 file changed, 1 insertion(+)
Enumerating objects: 5, done.
Writing objects: 100% (3/3), 280 bytes | 280.00 KiB/s, done.
To https://github.com/username/practice-repo.git
   d4e5f6a..a1b2c3d  main -> main
```

---

## Common Scenarios

### Scenario 1: "I committed to wrong branch"

**Solution:**
```bash
# 1. Note the commit hash
git log --oneline
# Output: a1b2c3d Wrong commit

# 2. Switch to correct branch
git checkout correct-branch

# 3. Cherry-pick the commit
git cherry-pick a1b2c3d

# 4. Switch back and remove commit
git checkout wrong-branch
git reset --hard HEAD~1
```

### Scenario 2: "I need to undo last commit but keep changes"

**Solution:**
```bash
git reset --soft HEAD~1
# Changes are now staged, ready to recommit
```

### Scenario 3: "I want to change last commit message"

**Solution:**
```bash
git commit --amend -m "New commit message"
```

### Scenario 4: "I accidentally deleted a file"

**Solution:**
```bash
# Restore from last commit
git restore filename.py

# Or restore from specific commit
git restore --source=HEAD~1 filename.py
```

### Scenario 5: "Merge conflict resolution"

**Solution:**
```bash
# 1. Merge causes conflict
git merge feature-branch
# Output: Auto-merging file.py
#         CONFLICT (content): Merge conflict in file.py

# 2. Open file and resolve conflicts
# Look for markers:
# <<<<<<< HEAD
# Your changes
# =======
# Their changes
# >>>>>>> feature-branch

# 3. Edit file to resolve conflict
# Remove markers and keep desired code

# 4. Stage resolved file
git add file.py

# 5. Complete merge
git commit
```

### Scenario 6: "I want to see what changed in a file"

**Solution:**
```bash
# See changes in working directory
git diff filename.py

# See changes in last commit
git show HEAD:filename.py

# See changes between commits
git diff commit1 commit2 -- filename.py
```

### Scenario 7: "I want to update my fork with upstream changes"

**Solution:**
```bash
# 1. Fetch upstream changes
git fetch upstream

# 2. Merge into your main
git checkout main
git merge upstream/main

# 3. Push to your fork
git push origin main
```

---

## GitHub-Specific Features

### 1. Pull Requests (PR)

**What:** A way to propose changes to a repository.

**Workflow:**
```bash
# 1. Create feature branch
git checkout -b feature-name

# 2. Make changes and commit
git add .
git commit -m "Add feature"

# 3. Push to GitHub
git push -u origin feature-name

# 4. Create PR on GitHub website
# 5. After PR is merged, delete branch
git branch -d feature-name
git push origin --delete feature-name
```

**Why:** Code review process before merging changes.

### 2. Issues

**What:** Track bugs, features, and tasks.

**GitHub Commands:**
- Create issue: Use GitHub website
- Reference in commit: `git commit -m "Fix #123"`
- Close issue: `git commit -m "Closes #123"`

### 3. GitHub Actions (CI/CD)

**What:** Automate workflows.

**Example `.github/workflows/test.yml`:**
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest
```

### 4. GitHub Pages

**What:** Host static websites from repository.

**Setup:**
```bash
# Create gh-pages branch
git checkout -b gh-pages
git push -u origin gh-pages
# Enable in repository settings
```

### 5. Releases

**What:** Package and distribute software versions.

**Workflow:**
```bash
# 1. Create tag
git tag -a v1.0.0 -m "Release version 1.0.0"

# 2. Push tag
git push origin v1.0.0

# 3. Create release on GitHub website
```

---

## Resources & Documentation

### Official Documentation

- **Git Official Documentation**: https://git-scm.com/doc
- **Git Reference Manual**: https://git-scm.com/docs
- **Git Book (Free)**: https://git-scm.com/book
- **GitHub Docs**: https://docs.github.com
- **GitHub Guides**: https://guides.github.com

### Interactive Learning

- **Learn Git Branching**: https://learngitbranching.js.org/ (Interactive visual tutorial)
- **GitHub Learning Lab**: https://lab.github.com/ (Free courses)
- **Atlassian Git Tutorials**: https://www.atlassian.com/git/tutorials

### Cheat Sheets

- **Git Cheat Sheet (GitHub)**: https://education.github.com/git-cheat-sheet-education.pdf
- **Git Commands Reference**: https://git-scm.com/docs

### Video Tutorials

- **Git & GitHub Crash Course (Traversy Media)**: Comprehensive YouTube tutorial
- **Git Tutorial for Beginners (freeCodeCamp)**: Complete beginner guide

### Practice Platforms

- **GitHub**: Create repositories and practice
- **GitKraken**: Visual Git client (free tier available)
- **SourceTree**: Free Git GUI client

### Common Git Commands Quick Reference

```bash
# Setup
git config --global user.name "Name"
git config --global user.email "email"

# Basic
git init
git status
git add .
git commit -m "message"
git log

# Branches
git branch
git checkout -b branch-name
git merge branch-name
git branch -d branch-name

# Remote
git remote add origin url
git push -u origin main
git pull
git fetch

# Undo
git restore file
git restore --staged file
git reset --soft HEAD~1
git revert HEAD

# Advanced
git stash
git stash pop
git cherry-pick commit
git rebase branch
git tag -a v1.0.0 -m "message"
```

---

## Best Practices

### 1. Commit Messages

**Good:**
```
Add user authentication feature
Fix bug in login validation
Update README with installation instructions
```

**Bad:**
```
fix
update
changes
asdf
```

### 2. Branch Naming

**Good:**
```
feature-user-authentication
bugfix-login-error
hotfix-security-patch
```

**Bad:**
```
branch1
test
my-changes
```

### 3. Commit Frequency

- Commit often (small, logical changes)
- One feature per commit
- Test before committing
- Write clear commit messages

### 4. Branch Strategy

- Keep `main` branch stable
- Create feature branches for new work
- Delete merged branches
- Use descriptive branch names

### 5. Collaboration

- Pull before pushing
- Communicate about large changes
- Use Pull Requests for code review
- Keep commits focused and small

---

## Troubleshooting

### Problem: "fatal: not a git repository"

**Solution:**
```bash
# You're not in a Git repository
# Navigate to repository or initialize one
cd /path/to/repo
# or
git init
```

### Problem: "error: failed to push some refs"

**Solution:**
```bash
# Someone else pushed changes
# Pull first, then push
git pull
git push
```

### Problem: "merge conflict"

**Solution:**
```bash
# Resolve conflicts manually
# Edit files with conflict markers
# Then:
git add .
git commit
```

### Problem: "detached HEAD state"

**Solution:**
```bash
# You checked out a commit instead of branch
# Create branch or switch to existing
git checkout -b new-branch
# or
git checkout main
```

### Problem: "permission denied (publickey)"

**Solution:**
```bash
# SSH key not set up
# Use HTTPS instead:
git remote set-url origin https://github.com/username/repo.git
# Or set up SSH keys
```

---

## Summary

### Essential Commands (Daily Use)

1. `git status` - Check what's changed
2. `git add .` - Stage changes
3. `git commit -m "message"` - Save changes
4. `git push` - Upload to GitHub
5. `git pull` - Download from GitHub

### Workflow Commands

1. `git checkout -b feature-name` - Create feature branch
2. `git merge feature-name` - Merge feature
3. `git branch -d feature-name` - Delete branch

### Emergency Commands

1. `git restore file` - Undo changes
2. `git reset --soft HEAD~1` - Undo commit (keep changes)
3. `git stash` - Temporarily save changes

### Remember

- **Commit often** with clear messages
- **Pull before push** to avoid conflicts
- **Use branches** for features
- **Don't force push** to shared branches
- **Read error messages** - they're helpful!

---

**Happy Git-ing!** 🚀

*This guide covers the essentials. Practice regularly and refer back when needed. Git becomes intuitive with experience!*

