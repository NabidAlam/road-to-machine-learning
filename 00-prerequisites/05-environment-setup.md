# Environment Setup for Machine Learning

Complete guide to setting up your development environment for machine learning and data science.

## Table of Contents

- [Installing Python](#installing-python)
- [Virtual Environments](#virtual-environments)
- [Installing Essential Libraries](#installing-essential-libraries)
- [Jupyter Notebook Setup](#jupyter-notebook-setup)
- [IDE Setup](#ide-setup)
- [Git & GitHub Setup](#git--github-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Installing Python

### Windows

**Method 1: Official Installer (Recommended)**

1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer
3. **Important**: Check "Add Python to PATH"
4. Click "Install Now"
5. Verify installation:
```bash
python --version
# Output: Python 3.11.x
```

**Method 2: Using Microsoft Store**
```bash
# Open Microsoft Store
# Search for "Python 3.11"
# Click Install
```

### Mac

**Method 1: Official Installer**
1. Download from [python.org](https://www.python.org/downloads/)
2. Run installer
3. Verify:
```bash
python3 --version
```

**Method 2: Using Homebrew (Recommended)**
```bash
# Install Homebrew first (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python

# Verify
python3 --version
```

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install Python
sudo apt install python3 python3-pip

# Verify
python3 --version
```

### Verify Installation

```bash
# Check Python version
python --version  # or python3 --version

# Check pip (package manager)
pip --version  # or pip3 --version

# Should see something like:
# Python 3.11.5
# pip 23.2.1
```

---

## Virtual Environments

### Why Virtual Environments?

Virtual environments isolate project dependencies, preventing conflicts between different projects.

### Creating Virtual Environment

**Windows:**
```bash
# Create virtual environment
python -m venv ml-env

# Activate
ml-env\Scripts\activate

# You should see (ml-env) in your prompt
```

**Mac/Linux:**
```bash
# Create virtual environment
python3 -m venv ml-env

# Activate
source ml-env/bin/activate

# You should see (ml-env) in your prompt
```

### Using Virtual Environment

```bash
# Activate (do this every time you work on project)
# Windows: ml-env\Scripts\activate
# Mac/Linux: source ml-env/bin/activate

# Install packages (they'll be isolated to this environment)
pip install numpy pandas

# Deactivate when done
deactivate
```

### Best Practices

1. **One environment per project**
2. **Always activate before working**
3. **Create requirements.txt:**
```bash
pip freeze > requirements.txt
```

4. **Share requirements.txt** with your project
5. **Recreate environment from requirements:**
```bash
pip install -r requirements.txt
```

---

## Installing Essential Libraries

### Core Data Science Libraries

```bash
# Activate your virtual environment first!

# NumPy - Numerical computing
pip install numpy

# Pandas - Data manipulation
pip install pandas

# Matplotlib - Plotting
pip install matplotlib

# Seaborn - Statistical visualization
pip install seaborn

# Scikit-learn - Machine learning
pip install scikit-learn
```

### Install All at Once

```bash
# Create requirements.txt with:
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0

# Install all
pip install -r requirements.txt
```

### Verify Installations

```python
# Test in Python
python

>>> import numpy as np
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> import seaborn as sns
>>> from sklearn import datasets

>>> print(np.__version__)  # Should print version number
>>> print(pd.__version__)
>>> # If no errors, everything is installed correctly!
```

---

## Jupyter Notebook Setup

### What is Jupyter Notebook?

Interactive environment for data science. Allows you to write code, see results, and add documentation in one place.

### Installation

```bash
# Install Jupyter
pip install jupyter notebook

# Or install JupyterLab (more features)
pip install jupyterlab
```

### Launching Jupyter

```bash
# Start Jupyter Notebook
jupyter notebook

# Or JupyterLab
jupyter lab

# Browser will open automatically
# If not, go to http://localhost:8888
```

### Creating Your First Notebook

1. Click "New" → "Python 3"
2. Write code in cells
3. Press `Shift + Enter` to run cell
4. Add markdown cells for documentation

### Useful Jupyter Shortcuts

- `Shift + Enter`: Run cell and move to next
- `Ctrl + Enter`: Run cell and stay
- `A`: Insert cell above
- `B`: Insert cell below
- `DD`: Delete cell
- `M`: Convert to markdown
- `Y`: Convert to code

### Installing Jupyter Extensions (Optional)

```bash
# Install extensions
pip install jupyter_contrib_nbextensions

# Enable extensions
jupyter contrib nbextension install --user
```

---

## IDE Setup

### VS Code (Recommended)

**Installation:**
1. Download from [code.visualstudio.com](https://code.visualstudio.com/)
2. Install Python extension
3. Install Jupyter extension

**Setup:**
1. Open VS Code
2. Install extensions:
   - Python (by Microsoft)
   - Jupyter (by Microsoft)
   - Pylance (by Microsoft)
3. Select Python interpreter:
   - `Ctrl + Shift + P` (Windows) or `Cmd + Shift + P` (Mac)
   - Type "Python: Select Interpreter"
   - Choose your virtual environment

**Using Jupyter in VS Code:**
1. Create `.ipynb` file
2. VS Code will recognize it
3. Run cells with play button or `Shift + Enter`

### PyCharm

**Installation:**
1. Download from [jetbrains.com/pycharm](https://www.jetbrains.com/pycharm/)
2. Choose Community Edition (free)

**Setup:**
1. Create new project
2. Set Python interpreter to virtual environment
3. Install packages through PyCharm's package manager

### Google Colab (Cloud Alternative)

**No installation needed!**
1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Sign in with Google account
3. Create new notebook
4. Free GPU access available!

---

## Git & GitHub Setup

### Installing Git

**Windows:**
1. Download from [git-scm.com](https://git-scm.com/download/win)
2. Run installer (use default options)
3. Verify:
```bash
git --version
```

**Mac:**
```bash
# Using Homebrew
brew install git

# Or download from git-scm.com
```

**Linux:**
```bash
sudo apt install git
```

### Initial Git Configuration

```bash
# Set your name
git config --global user.name "Your Name"

# Set your email
git config --global user.email "your.email@example.com"

# Set default branch name
git config --global init.defaultBranch main

# Verify
git config --list
```

### GitHub Setup

1. Create account at [github.com](https://github.com/)
2. Generate SSH key (optional but recommended):
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to GitHub:
# 1. Copy public key: cat ~/.ssh/id_ed25519.pub
# 2. Go to GitHub → Settings → SSH Keys → New SSH Key
# 3. Paste and save
```

### First Repository

```bash
# Create project directory
mkdir my-ml-project
cd my-ml-project

# Initialize Git
git init

# Create .gitignore
echo "ml-env/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore

# Add files
git add .

# First commit
git commit -m "Initial commit"

# Connect to GitHub (create repo on GitHub first)
git remote add origin https://github.com/yourusername/my-ml-project.git
git push -u origin main
```

**See [Complete Git Guide](../resources/git_guide.md) for detailed Git tutorial.**

---

## Verification

### Complete Setup Check

Run this Python script to verify everything:

```python
# verification.py
import sys

print("Python version:", sys.version)
print("\nChecking libraries...")

try:
    import numpy as np
    print("✓ NumPy:", np.__version__)
except ImportError:
    print("✗ NumPy not installed")

try:
    import pandas as pd
    print("✓ Pandas:", pd.__version__)
except ImportError:
    print("✗ Pandas not installed")

try:
    import matplotlib
    print("✓ Matplotlib:", matplotlib.__version__)
except ImportError:
    print("✗ Matplotlib not installed")

try:
    import seaborn as sns
    print("✓ Seaborn:", sns.__version__)
except ImportError:
    print("✗ Seaborn not installed")

try:
    import sklearn
    print("✓ Scikit-learn:", sklearn.__version__)
except ImportError:
    print("✗ Scikit-learn not installed")

print("\nSetup complete! ✓")
```

**Run:**
```bash
python verification.py
```

---

## Troubleshooting

### Problem: "python is not recognized"

**Solution:**
- Python not in PATH
- Use `python3` instead of `python`
- Reinstall Python with "Add to PATH" checked

### Problem: "pip is not recognized"

**Solution:**
```bash
# Use pip3 instead
pip3 install numpy

# Or install pip
python -m ensurepip --upgrade
```

### Problem: "Permission denied" when installing

**Solution:**
- Don't use `sudo` with pip in virtual environment
- Make sure virtual environment is activated
- Use `--user` flag if needed: `pip install --user numpy`

### Problem: Jupyter won't start

**Solution:**
```bash
# Reinstall Jupyter
pip install --upgrade jupyter

# Clear Jupyter cache
jupyter --paths
# Delete cache directories if needed
```

### Problem: Import errors in Jupyter

**Solution:**
- Make sure you installed packages in the same environment
- Check which Python Jupyter is using:
```python
import sys
print(sys.executable)
```

---

## Quick Start Checklist

- [ ] Python installed and verified
- [ ] Virtual environment created and activated
- [ ] Core libraries installed (NumPy, Pandas, etc.)
- [ ] Jupyter Notebook installed and working
- [ ] IDE set up (VS Code or PyCharm)
- [ ] Git installed and configured
- [ ] GitHub account created
- [ ] Verification script runs successfully

---

## Next Steps

1. **Practice**: Create a test notebook and import all libraries
2. **Explore**: Try loading a dataset with Pandas
3. **Move Forward**: Proceed to [01-python-for-data-science](../01-python-for-data-science/README.md)

**You're now ready to start your ML journey!** 🚀

---

## Additional Resources

- [Python Official Docs](https://docs.python.org/3/)
- [Jupyter Documentation](https://jupyter.org/documentation)
- [VS Code Python Guide](https://code.visualstudio.com/docs/python/python-tutorial)
- [Git Guide](../resources/git_guide.md)

**Remember**: If you encounter issues, Google the error message - someone has likely solved it before!

