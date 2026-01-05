# Advanced Introduction to ML Topics

Comprehensive guide to advanced ML workflow patterns, system design, and production considerations.

## Table of Contents

- [Advanced ML Workflow Patterns](#advanced-ml-workflow-patterns)
- [ML System Design Principles](#ml-system-design-principles)
- [Data-Centric vs Model-Centric Approaches](#data-centric-vs-model-centric-approaches)
- [ML Project Management](#ml-project-management)
- [Advanced Problem Framing](#advanced-problem-framing)
- [ML in Production Considerations](#ml-in-production-considerations)

---

## Advanced ML Workflow Patterns

### Iterative ML Development

```python
# Traditional workflow
def traditional_ml_workflow():
    """
    1. Collect data
    2. Train model
    3. Deploy
    4. Monitor
    """
    pass

# Iterative workflow
def iterative_ml_workflow():
    """
    1. Define problem
    2. Collect initial data
    3. Train baseline model
    4. Evaluate and identify issues
    5. Improve (data or model)
    6. Repeat steps 3-5
    7. Deploy
    8. Monitor and iterate
    """
    pass
```

### Experiment Tracking

```python
# Track experiments systematically
experiments = []

def track_experiment(name, params, metrics, model):
    experiment = {
        'name': name,
        'parameters': params,
        'metrics': metrics,
        'model': model,
        'timestamp': datetime.now()
    }
    experiments.append(experiment)
    return experiment

# Example
track_experiment(
    name='baseline_model',
    params={'algorithm': 'logistic_regression', 'C': 1.0},
    metrics={'accuracy': 0.85, 'f1': 0.82},
    model=model
)
```

### Version Control for ML

```python
# Track data versions
import hashlib

def get_data_hash(data):
    """Generate hash for data versioning"""
    return hashlib.md5(str(data).encode()).hexdigest()

# Track model versions
def save_model_version(model, version, metrics):
    """Save model with version info"""
    model_info = {
        'version': version,
        'metrics': metrics,
        'timestamp': datetime.now(),
        'model': model
    }
    # Save to model registry
    return model_info
```

---

## ML System Design Principles

### Scalability

```python
# Design for scale from the start
class ScalableMLSystem:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.cache = {}
    
    def preprocess(self, data):
        """Preprocessing that scales"""
        # Use vectorized operations
        return vectorized_preprocess(data)
    
    def predict(self, data):
        """Batch prediction for efficiency"""
        if len(data) == 1:
            # Single prediction
            return self._predict_single(data[0])
        else:
            # Batch prediction
            return self._predict_batch(data)
    
    def _predict_batch(self, data):
        """Efficient batch processing"""
        preprocessed = self.preprocess(data)
        return self.model.predict(preprocessed)
```

### Modularity

```python
# Modular ML system design
class MLPipeline:
    def __init__(self):
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model = Model()
        self.evaluator = Evaluator()
    
    def run(self, data_source):
        # Load
        data = self.data_loader.load(data_source)
        
        # Preprocess
        data = self.preprocessor.transform(data)
        
        # Feature engineering
        features = self.feature_engineer.transform(data)
        
        # Predict
        predictions = self.model.predict(features)
        
        # Evaluate
        metrics = self.evaluator.evaluate(predictions)
        
        return predictions, metrics
```

### Reproducibility

```python
import random
import numpy as np

# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # For TensorFlow
    # tf.random.set_seed(seed)
    # For PyTorch
    # torch.manual_seed(seed)

# Save configuration
def save_config(config, path):
    """Save experiment configuration"""
    import json
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)

# Load configuration
def load_config(path):
    """Load experiment configuration"""
    import json
    with open(path, 'r') as f:
        return json.load(f)
```

---

## Data-Centric vs Model-Centric Approaches

### Model-Centric Approach

```python
# Focus on improving the model
def model_centric_approach():
    """
    Strategy:
    1. Fix dataset
    2. Improve model architecture
    3. Tune hyperparameters
    4. Try different algorithms
    
    Example:
    - Try more complex models (XGBoost, Neural Networks)
    - Ensemble methods
    - Hyperparameter tuning
    """
    models = [
        LogisticRegression(),
        RandomForestClassifier(),
        XGBClassifier(),
        NeuralNetwork()
    ]
    
    for model in models:
        train_and_evaluate(model)
```

### Data-Centric Approach

```python
# Focus on improving the data
def data_centric_approach():
    """
    Strategy:
    1. Collect more data
    2. Improve data quality
    3. Better labeling
    4. Data augmentation
    5. Feature engineering
    
    Example:
    - Clean noisy labels
    - Collect edge cases
    - Balance dataset
    - Add relevant features
    """
    # Improve data quality
    clean_labels()
    collect_edge_cases()
    balance_dataset()
    engineer_features()
    
    # Then train simple model
    model = LogisticRegression()
    train_and_evaluate(model)
```

### Hybrid Approach

```python
# Best of both worlds
def hybrid_approach():
    """
    1. Start with data-centric improvements
    2. Train baseline model
    3. Identify model limitations
    4. Improve model if needed
    5. Iterate on data based on model errors
    """
    # Step 1: Improve data
    improved_data = improve_data_quality(data)
    
    # Step 2: Baseline model
    baseline = train_baseline(improved_data)
    
    # Step 3: Analyze errors
    errors = analyze_errors(baseline, improved_data)
    
    # Step 4: Improve data based on errors
    better_data = fix_error_cases(improved_data, errors)
    
    # Step 5: Train better model
    final_model = train_final_model(better_data)
    
    return final_model
```

---

## ML Project Management

### Project Structure

```
ml_project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── models/
├── reports/
├── configs/
└── requirements.txt
```

### Agile ML Development

```python
# Sprint planning for ML projects
class MLSprint:
    def __init__(self, duration=2):  # 2 weeks
        self.duration = duration
        self.tasks = []
    
    def add_task(self, task, priority):
        self.tasks.append({
            'task': task,
            'priority': priority,
            'status': 'todo'
        })
    
    def prioritize(self):
        """Prioritize tasks by impact and effort"""
        self.tasks.sort(key=lambda x: (
            x['priority'],  # Higher priority first
            -estimate_effort(x['task'])  # Lower effort first
        ))

# Example sprint
sprint = MLSprint()
sprint.add_task('Collect more training data', priority=1)
sprint.add_task('Improve feature engineering', priority=2)
sprint.add_task('Try new algorithm', priority=3)
sprint.prioritize()
```

### Documentation

```python
# Document ML projects
class MLDocumentation:
    def __init__(self, project_name):
        self.project_name = project_name
        self.sections = {}
    
    def add_section(self, name, content):
        self.sections[name] = content
    
    def generate_report(self):
        """Generate project documentation"""
        report = f"""
# {self.project_name} - ML Project Documentation

## Problem Statement
{self.sections.get('problem', '')}

## Data
{self.sections.get('data', '')}

## Methodology
{self.sections.get('methodology', '')}

## Results
{self.sections.get('results', '')}

## Conclusions
{self.sections.get('conclusions', '')}
"""
        return report
```

---

## Advanced Problem Framing

### Problem Decomposition

```python
# Break down complex problems
def decompose_problem(problem):
    """
    Example: "Predict customer churn"
    
    Sub-problems:
    1. Identify churn indicators
    2. Predict churn probability
    3. Recommend retention actions
    """
    sub_problems = [
        'Identify churn indicators',
        'Predict churn probability',
        'Recommend retention actions'
    ]
    
    return sub_problems

# Solve each sub-problem
def solve_sub_problems(sub_problems):
    solutions = {}
    for problem in sub_problems:
        solution = solve(problem)
        solutions[problem] = solution
    return solutions
```

### Success Metrics Definition

```python
# Define clear success metrics
class SuccessMetrics:
    def __init__(self):
        self.metrics = {}
    
    def add_metric(self, name, target, current=None):
        self.metrics[name] = {
            'target': target,
            'current': current,
            'achieved': current is not None and current >= target
        }
    
    def check_success(self):
        """Check if all metrics are achieved"""
        return all(m['achieved'] for m in self.metrics.values())

# Example
metrics = SuccessMetrics()
metrics.add_metric('accuracy', target=0.90, current=0.92)
metrics.add_metric('latency', target=100, current=80)  # ms
metrics.add_metric('cost', target=0.01, current=0.008)  # per prediction

if metrics.check_success():
    print("Project is successful!")
```

### Risk Assessment

```python
# Assess ML project risks
class MLRiskAssessment:
    def __init__(self):
        self.risks = []
    
    def add_risk(self, risk, probability, impact, mitigation):
        self.risks.append({
            'risk': risk,
            'probability': probability,  # High/Medium/Low
            'impact': impact,  # High/Medium/Low
            'mitigation': mitigation
        })
    
    def get_critical_risks(self):
        """Get high probability, high impact risks"""
        return [r for r in self.risks 
                if r['probability'] == 'High' and r['impact'] == 'High']

# Example
assessment = MLRiskAssessment()
assessment.add_risk(
    risk='Data quality issues',
    probability='High',
    impact='High',
    mitigation='Implement data validation pipeline'
)
assessment.add_risk(
    risk='Model performance degradation',
    probability='Medium',
    impact='High',
    mitigation='Set up monitoring and retraining pipeline'
)
```

---

## ML in Production Considerations

### Model Monitoring

```python
# Monitor model in production
class ModelMonitor:
    def __init__(self, model, baseline_metrics):
        self.model = model
        self.baseline_metrics = baseline_metrics
        self.predictions = []
        self.performance_history = []
    
    def log_prediction(self, input_data, prediction, actual=None):
        """Log prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now(),
            'input': input_data,
            'prediction': prediction,
            'actual': actual
        }
        self.predictions.append(log_entry)
    
    def check_drift(self):
        """Check for data drift"""
        recent_data = self.get_recent_data()
        baseline_dist = self.get_baseline_distribution()
        
        # Statistical test for drift
        drift_detected = statistical_test(recent_data, baseline_dist)
        return drift_detected
    
    def check_performance(self):
        """Check model performance"""
        recent_metrics = self.calculate_recent_metrics()
        
        # Compare with baseline
        performance_degraded = (
            recent_metrics['accuracy'] < self.baseline_metrics['accuracy'] * 0.95
        )
        
        return performance_degraded
```

### A/B Testing Framework

```python
# A/B testing for ML models
class ABTest:
    def __init__(self, model_a, model_b, traffic_split=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.results_a = []
        self.results_b = []
    
    def route_request(self, request):
        """Route request to A or B"""
        if random.random() < self.traffic_split:
            prediction = self.model_a.predict(request)
            self.results_a.append(prediction)
            return prediction, 'A'
        else:
            prediction = self.model_b.predict(request)
            self.results_b.append(prediction)
            return prediction, 'B'
    
    def analyze_results(self):
        """Analyze A/B test results"""
        from scipy import stats
        
        # Statistical test
        statistic, p_value = stats.ttest_ind(
            self.results_a, self.results_b
        )
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_a': np.mean(self.results_a),
            'mean_b': np.mean(self.results_b)
        }
```

### Model Versioning

```python
# Version control for models
class ModelVersioning:
    def __init__(self):
        self.versions = {}
        self.current_version = None
    
    def register_version(self, version, model, metrics, metadata):
        """Register new model version"""
        self.versions[version] = {
            'model': model,
            'metrics': metrics,
            'metadata': metadata,
            'timestamp': datetime.now()
        }
    
    def promote_version(self, version, environment='production'):
        """Promote version to environment"""
        if version in self.versions:
            self.current_version = version
            # Deploy to environment
            self.deploy(version, environment)
    
    def rollback(self, previous_version):
        """Rollback to previous version"""
        if previous_version in self.versions:
            self.promote_version(previous_version)
```

---

## Key Takeaways

1. **Workflow**: Use iterative development with experiment tracking
2. **Design**: Build scalable, modular, and reproducible systems
3. **Approach**: Balance data-centric and model-centric improvements
4. **Management**: Use proper project structure and documentation
5. **Production**: Plan for monitoring, A/B testing, and versioning

---

**Remember**: Advanced ML practices will make your projects more successful and maintainable!

