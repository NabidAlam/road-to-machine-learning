# Comprehensive Data Validation Guide

Complete guide to validating data quality, detecting data drift, and ensuring data integrity for machine learning pipelines.

## Table of Contents

- [Introduction](#introduction)
- [Schema Validation](#schema-validation)
- [Data Quality Checks](#data-quality-checks)
- [Data Drift Detection](#data-drift-detection)
- [Data Integrity Checks](#data-integrity-checks)
- [Automated Validation Pipelines](#automated-validation-pipelines)
- [Tools and Libraries](#tools-and-libraries)
- [Best Practices](#best-practices)

---

## Introduction

### Why Data Validation Matters

**Data quality issues cause:**
- Poor model performance
- Production failures
- Wasted compute resources
- Incorrect business decisions
- Regulatory compliance issues

**Data validation helps:**
- Catch issues early (before training)
- Ensure data consistency
- Detect data drift
- Maintain model performance
- Build trust in ML systems

### When to Validate Data

1. **Before Training**: Ensure training data quality
2. **During Training**: Monitor for issues
3. **Before Deployment**: Validate production data matches training data
4. **In Production**: Continuously monitor for drift

---

## Schema Validation

### What is Schema Validation?

Schema validation ensures data structure matches expected format:
- Column names and types
- Value ranges
- Required vs optional fields
- Data types (int, float, string, datetime)

### Using Pandera

**Pandera** is a Python library for data validation.

```python
import pandas as pd
import pandera as pa
from pandera import Column, Check

# Define schema
schema = pa.DataFrameSchema({
    "age": Column(int, checks=[
        Check.greater_than(0),
        Check.less_than(150)
    ]),
    "income": Column(float, checks=[
        Check.greater_than(0),
        Check.less_than(10000000)
    ]),
    "email": Column(str, checks=[
        Check.str_matches(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    ]),
    "category": Column(str, checks=[
        Check.isin(['A', 'B', 'C'])
    ]),
    "date": Column('datetime64[ns]', nullable=True)
})

# Validate data
try:
    validated_df = schema(df)
    print("Data passes validation")
except pa.errors.SchemaError as e:
    print(f"Validation failed: {e}")
```

### Using Great Expectations

**Great Expectations** is a comprehensive data validation framework.

```python
import great_expectations as ge
from great_expectations.dataset import PandasDataset

# Load data as Great Expectations dataset
df_ge = ge.from_pandas(df)

# Define expectations
df_ge.expect_column_to_exist("age")
df_ge.expect_column_values_to_be_between("age", min_value=0, max_value=150)
df_ge.expect_column_values_to_not_be_null("email")
df_ge.expect_column_values_to_match_regex("email", r'^[^@]+@[^@]+\.[^@]+$')
df_ge.expect_column_values_to_be_in_set("category", ["A", "B", "C"])

# Validate
validation_result = df_ge.validate()
if validation_result["success"]:
    print("All expectations passed")
else:
    print("Some expectations failed:")
    for result in validation_result["results"]:
        if not result["success"]:
            print(f"  - {result['expectation_config']['expectation_type']}")
```

### Custom Schema Validation

```python
def validate_schema(df, expected_schema):
    """
    Validate DataFrame against expected schema
    
    Parameters:
    - df: DataFrame to validate
    - expected_schema: Dict with column names and validation rules
    """
    errors = []
    
    # Check column existence
    expected_cols = set(expected_schema.keys())
    actual_cols = set(df.columns)
    
    missing_cols = expected_cols - actual_cols
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    extra_cols = actual_cols - expected_cols
    if extra_cols:
        errors.append(f"Unexpected columns: {extra_cols}")
    
    # Check each column
    for col, rules in expected_schema.items():
        if col not in df.columns:
            continue
        
        # Check data type
        if 'dtype' in rules:
            if df[col].dtype != rules['dtype']:
                errors.append(f"Column '{col}': expected {rules['dtype']}, got {df[col].dtype}")
        
        # Check value range
        if 'min' in rules:
            if (df[col] < rules['min']).any():
                errors.append(f"Column '{col}': values below minimum {rules['min']}")
        
        if 'max' in rules:
            if (df[col] > rules['max']).any():
                errors.append(f"Column '{col}': values above maximum {rules['max']}")
        
        # Check allowed values
        if 'allowed_values' in rules:
            invalid = ~df[col].isin(rules['allowed_values'])
            if invalid.any():
                errors.append(f"Column '{col}': invalid values found")
        
        # Check nulls
        if 'nullable' in rules and not rules['nullable']:
            if df[col].isna().any():
                errors.append(f"Column '{col}': contains null values")
    
    return errors

# Example usage
schema = {
    'age': {'dtype': 'int64', 'min': 0, 'max': 150, 'nullable': False},
    'income': {'dtype': 'float64', 'min': 0, 'nullable': False},
    'category': {'allowed_values': ['A', 'B', 'C'], 'nullable': False}
}

errors = validate_schema(df, schema)
if errors:
    for error in errors:
        print(f"{error}")
else:
    print("Schema validation passed")
```

---

## Data Quality Checks

### Completeness Checks

```python
def check_completeness(df, threshold=0.95):
    """
    Check if data completeness meets threshold
    
    Parameters:
    - df: DataFrame to check
    - threshold: Minimum completeness ratio (0-1)
    """
    completeness = {}
    issues = []
    
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        total_count = len(df)
        ratio = non_null_count / total_count
        
        completeness[col] = ratio
        
        if ratio < threshold:
            issues.append({
                'column': col,
                'completeness': ratio,
                'missing_count': total_count - non_null_count
            })
    
    return {
        'overall_completeness': df.notna().all(axis=1).sum() / len(df),
        'column_completeness': completeness,
        'issues': issues
    }

# Example
completeness_report = check_completeness(df, threshold=0.95)
print(f"Overall completeness: {completeness_report['overall_completeness']:.2%}")

for issue in completeness_report['issues']:
    print(f"{issue['column']}: {issue['completeness']:.2%} complete "
          f"({issue['missing_count']} missing)")
```

### Uniqueness Checks

```python
def check_uniqueness(df, key_columns):
    """
    Check uniqueness of key columns
    
    Parameters:
    - df: DataFrame to check
    - key_columns: List of columns that should be unique
    """
    issues = []
    
    for col in key_columns:
        if col not in df.columns:
            issues.append(f"Column '{col}' not found")
            continue
        
        duplicates = df[col].duplicated().sum()
        if duplicates > 0:
            issues.append({
                'column': col,
                'duplicate_count': duplicates,
                'duplicate_values': df[df[col].duplicated(keep=False)][col].unique()[:10]
            })
    
    return issues

# Example
key_columns = ['customer_id', 'transaction_id']
uniqueness_issues = check_uniqueness(df, key_columns)

if uniqueness_issues:
    for issue in uniqueness_issues:
        if isinstance(issue, dict):
            print(f"{issue['column']}: {issue['duplicate_count']} duplicates")
        else:
            print(f"{issue}")
else:
    print("All key columns are unique")
```

### Consistency Checks

```python
def check_consistency(df):
    """
    Check data consistency across related columns
    """
    issues = []
    
    # Example: Age and birth_date consistency
    if 'age' in df.columns and 'birth_date' in df.columns:
        df['calculated_age'] = (pd.Timestamp.now() - pd.to_datetime(df['birth_date'])).dt.days / 365.25
        age_diff = abs(df['age'] - df['calculated_age'])
        inconsistent = age_diff > 1  # Allow 1 year difference
        
        if inconsistent.any():
            issues.append({
                'type': 'age_birthdate_mismatch',
                'count': inconsistent.sum(),
                'examples': df[inconsistent][['age', 'birth_date', 'calculated_age']].head()
            })
    
    # Example: Start date before end date
    if 'start_date' in df.columns and 'end_date' in df.columns:
        invalid_dates = pd.to_datetime(df['start_date']) > pd.to_datetime(df['end_date'])
        if invalid_dates.any():
            issues.append({
                'type': 'date_range_invalid',
                'count': invalid_dates.sum()
            })
    
    return issues
```

### Distribution Checks

```python
def check_distribution(train_df, test_df, columns, threshold=0.1):
    """
    Check if test data distribution matches training data
    
    Parameters:
    - train_df: Training DataFrame
    - test_df: Test DataFrame
    - columns: Columns to check
    - threshold: Maximum allowed difference in statistics
    """
    from scipy import stats
    
    issues = []
    
    for col in columns:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            continue
        
        # Compare means
        train_mean = train_df[col].mean()
        test_mean = test_df[col].mean()
        mean_diff = abs(train_mean - test_mean) / (abs(train_mean) + 1e-10)
        
        # Compare stds
        train_std = train_df[col].std()
        test_std = test_df[col].std()
        std_diff = abs(train_std - test_std) / (abs(train_std) + 1e-10)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(train_df[col].dropna(), test_df[col].dropna())
        
        if mean_diff > threshold or std_diff > threshold or ks_pvalue < 0.05:
            issues.append({
                'column': col,
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'ks_pvalue': ks_pvalue,
                'train_mean': train_mean,
                'test_mean': test_mean
            })
    
    return issues

# Example
distribution_issues = check_distribution(X_train, X_test, X_train.columns)

for issue in distribution_issues:
    print(f"{issue['column']}: Distribution mismatch")
    print(f"Mean difference: {issue['mean_diff']:.2%}")
    print(f"KS test p-value: {issue['ks_pvalue']:.4f}")
```

---

## Data Drift Detection

### What is Data Drift?

**Data drift** occurs when the distribution of production data changes compared to training data. This can cause model performance degradation.

**Types of Drift:**
- **Covariate Shift**: Input feature distribution changes
- **Concept Drift**: Relationship between features and target changes
- **Label Drift**: Target distribution changes

### Detecting Covariate Drift

```python
from scipy import stats
import numpy as np

def detect_covariate_drift(train_data, production_data, alpha=0.05):
    """
    Detect covariate drift using statistical tests
    
    Parameters:
    - train_data: Training data (DataFrame)
    - production_data: Production data (DataFrame)
    - alpha: Significance level
    """
    drift_results = {}
    
    for col in train_data.columns:
        if not pd.api.types.is_numeric_dtype(train_data[col]):
            continue
        
        train_values = train_data[col].dropna()
        prod_values = production_data[col].dropna()
        
        if len(train_values) == 0 or len(prod_values) == 0:
            continue
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(train_values, prod_values)
        
        # Mann-Whitney U test (non-parametric)
        mw_stat, mw_pvalue = stats.mannwhitneyu(train_values, prod_values, alternative='two-sided')
        
        # Calculate distribution distance (Wasserstein distance)
        from scipy.stats import wasserstein_distance
        wass_dist = wasserstein_distance(train_values, prod_values)
        
        # Normalize by standard deviation
        wass_dist_norm = wass_dist / (train_values.std() + 1e-10)
        
        drift_detected = ks_pvalue < alpha or mw_pvalue < alpha or wass_dist_norm > 0.5
        
        drift_results[col] = {
            'drift_detected': drift_detected,
            'ks_pvalue': ks_pvalue,
            'mw_pvalue': mw_pvalue,
            'wasserstein_distance': wass_dist_norm,
            'train_mean': train_values.mean(),
            'prod_mean': prod_values.mean(),
            'train_std': train_values.std(),
            'prod_std': prod_values.std()
        }
    
    return drift_results

# Example
drift_results = detect_covariate_drift(X_train, X_production)

for col, result in drift_results.items():
    if result['drift_detected']:
        print(f"DRIFT DETECTED in '{col}':")
        print(f"KS p-value: {result['ks_pvalue']:.4f}")
        print(f"Wasserstein distance: {result['wasserstein_distance']:.4f}")
        print(f"Train mean: {result['train_mean']:.2f}, Prod mean: {result['prod_mean']:.2f}")
```

### Using Evidently AI

**Evidently AI** is a tool for monitoring ML models and detecting drift.

```python
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab

# Create data drift dashboard
data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
data_drift_dashboard.calculate(
    reference_data=train_df,
    current_data=production_df
)

# Save report
data_drift_dashboard.save("data_drift_report.html")
```

---

## Data Integrity Checks

### Referential Integrity

```python
def check_referential_integrity(main_df, reference_df, foreign_key, primary_key):
    """
    Check referential integrity between DataFrames
    
    Parameters:
    - main_df: DataFrame with foreign key
    - reference_df: DataFrame with primary key
    - foreign_key: Column name in main_df
    - primary_key: Column name in reference_df
    """
    main_values = set(main_df[foreign_key].dropna().unique())
    reference_values = set(reference_df[primary_key].unique())
    
    orphaned = main_values - reference_values
    
    if orphaned:
        return {
            'valid': False,
            'orphaned_count': len(orphaned),
            'orphaned_values': list(orphaned)[:10]  # First 10
        }
    else:
        return {'valid': True}

# Example
# Check if customer_id in transactions exists in customers table
integrity_check = check_referential_integrity(
    transactions_df, customers_df,
    'customer_id', 'customer_id'
)

if not integrity_check['valid']:
    print(f"{integrity_check['orphaned_count']} orphaned records found")
```

### Business Rule Validation

```python
def validate_business_rules(df, rules):
    """
    Validate data against business rules
    
    Parameters:
    - df: DataFrame to validate
    - rules: List of rule functions that return (is_valid, message)
    """
    violations = []
    
    for rule_name, rule_func in rules.items():
        is_valid, message = rule_func(df)
        if not is_valid:
            violations.append({
                'rule': rule_name,
                'message': message
            })
    
    return violations

# Define business rules
def rule_total_positive(df):
    """Total amount should be positive"""
    if 'total' in df.columns:
        invalid = df['total'] <= 0
        if invalid.any():
            return False, f"{invalid.sum()} records with non-positive total"
    return True, ""

def rule_discount_range(df):
    """Discount should be between 0 and 100%"""
    if 'discount_pct' in df.columns:
        invalid = (df['discount_pct'] < 0) | (df['discount_pct'] > 100)
        if invalid.any():
            return False, f"{invalid.sum()} records with invalid discount"
    return True, ""

# Apply rules
rules = {
    'total_positive': rule_total_positive,
    'discount_range': rule_discount_range
}

violations = validate_business_rules(df, rules)

for violation in violations:
    print(f"{violation['rule']}: {violation['message']}")
```

---

## Automated Validation Pipelines

### Complete Validation Pipeline

```python
class DataValidator:
    """Comprehensive data validation pipeline"""
    
    def __init__(self, schema, rules=None):
        self.schema = schema
        self.rules = rules or {}
    
    def validate(self, df, reference_df=None):
        """Run all validation checks"""
        results = {
            'schema_validation': self._validate_schema(df),
            'quality_checks': self._quality_checks(df),
            'drift_detection': None,
            'integrity_checks': None
        }
        
        if reference_df is not None:
            results['drift_detection'] = self._detect_drift(df, reference_df)
            results['integrity_checks'] = self._integrity_checks(df, reference_df)
        
        results['is_valid'] = all([
            results['schema_validation']['valid'],
            results['quality_checks']['valid'],
            results['drift_detection'] is None or not results['drift_detection'].get('drift_detected', False)
        ])
        
        return results
    
    def _validate_schema(self, df):
        """Schema validation"""
        errors = validate_schema(df, self.schema)
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _quality_checks(self, df):
        """Data quality checks"""
        completeness = check_completeness(df)
        uniqueness = check_uniqueness(df, self.schema.keys())
        
        return {
            'valid': len(completeness['issues']) == 0 and len(uniqueness) == 0,
            'completeness': completeness,
            'uniqueness': uniqueness
        }
    
    def _detect_drift(self, df, reference_df):
        """Detect data drift"""
        return detect_covariate_drift(reference_df, df)
    
    def _integrity_checks(self, df, reference_df):
        """Referential integrity checks"""
        # Implementation depends on your data structure
        return {'valid': True}

# Usage
validator = DataValidator(schema=expected_schema)
validation_results = validator.validate(new_data, reference_df=train_data)

if validation_results['is_valid']:
    print("All validations passed")
else:
    print("Validation failed:")
    if validation_results['schema_validation']['errors']:
        print("Schema errors:", validation_results['schema_validation']['errors'])
    if validation_results['quality_checks']['issues']:
        print("  Quality issues found")
```

---

## Tools and Libraries

### Popular Tools

1. **Great Expectations**: Comprehensive data validation
2. **Pandera**: Lightweight schema validation
3. **Evidently AI**: ML monitoring and drift detection
4. **Deepchecks**: ML validation library
5. **TensorFlow Data Validation**: For TensorFlow pipelines

### Choosing a Tool

| Tool | Best For | Complexity |
|------|----------|------------|
| **Pandera** | Simple schema validation | Low |
| **Great Expectations** | Comprehensive validation | Medium |
| **Evidently AI** | ML monitoring, drift detection | Medium |
| **Deepchecks** | ML-specific validation | Medium |
| **Custom** | Specific requirements | Variable |

---

## Best Practices

1. **Validate Early**: Check data quality before training
2. **Automate**: Use automated pipelines, don't validate manually
3. **Document**: Document all validation rules and thresholds
4. **Monitor Continuously**: Set up continuous monitoring in production
5. **Set Alerts**: Alert when validation fails
6. **Version Schemas**: Track schema changes over time
7. **Test Validation**: Test your validation logic itself
8. **Balance Strictness**: Too strict = false alarms, too loose = missed issues

---

## Key Takeaways

1. **Schema validation** ensures data structure matches expectations
2. **Quality checks** verify completeness, uniqueness, consistency
3. **Drift detection** identifies when production data changes
4. **Integrity checks** ensure referential and business rule compliance
5. **Automate everything** - manual validation doesn't scale
6. **Monitor continuously** - data quality can degrade over time
7. **Use appropriate tools** - choose based on your needs
8. **Document and version** - track validation rules and changes

---

## Resources

- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [Pandera Documentation](https://pandera.readthedocs.io/)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Data Quality Best Practices](https://www.oreilly.com/library/view/fundamentals-of-data/9781492082099/)

---

**Remember**: Garbage in, garbage out. Always validate your data before it reaches your models!

