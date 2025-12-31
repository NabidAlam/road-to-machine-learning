# Advanced MLOps Topics

Comprehensive guide to advanced MLOps techniques and tools.

## Table of Contents

- [Advanced Experiment Tracking](#advanced-experiment-tracking)
- [Feature Stores](#feature-stores)
- [Model Monitoring](#model-monitoring)
- [Automated Retraining](#automated-retraining)
- [Kubeflow](#kubeflow)
- [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Advanced Experiment Tracking

### Hyperparameter Tuning with MLflow

```python
import mlflow
from sklearn.model_selection import ParameterGrid

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20]
}

best_score = 0
best_params = None

for params in ParameterGrid(param_grid):
    with mlflow.start_run():
        mlflow.log_params(params)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        mlflow.log_metric("accuracy", score)
        
        if score > best_score:
            best_score = score
            best_params = params
            mlflow.sklearn.log_model(model, "model")
            mlflow.set_tag("best_model", "true")
```

---

## Feature Stores

### What is a Feature Store?

Centralized repository for features used in training and serving.

**Benefits:**
- Reuse features across models
- Consistency between training and serving
- Feature versioning
- Real-time feature serving

### Simple Feature Store

```python
class FeatureStore:
    def __init__(self):
        self.features = {}
        self.versions = {}
    
    def register_feature(self, name, feature_fn, version=1):
        """Register a feature computation function"""
        self.features[name] = feature_fn
        self.versions[name] = version
    
    def get_feature(self, name, data):
        """Compute feature"""
        return self.features[name](data)
```

---

## Model Monitoring

### Performance Monitoring

```python
import logging
from datetime import datetime

class ModelMonitor:
    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.timestamps = []
    
    def log_prediction(self, prediction, actual=None):
        """Log prediction for monitoring"""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(datetime.now())
    
    def calculate_drift(self):
        """Calculate prediction drift"""
        if len(self.predictions) < 100:
            return None
        
        recent_preds = self.predictions[-100:]
        older_preds = self.predictions[-200:-100]
        
        # Statistical test for drift
        from scipy import stats
        statistic, p_value = stats.ks_2samp(older_preds, recent_preds)
        return p_value < 0.05  # Significant drift
```

---

## Automated Retraining

### Retraining Pipeline

```python
def should_retrain(monitor):
    """Determine if model should be retrained"""
    # Check data drift
    if monitor.calculate_drift():
        return True
    
    # Check performance degradation
    recent_accuracy = calculate_recent_accuracy(monitor)
    if recent_accuracy < threshold:
        return True
    
    return False

def retrain_pipeline():
    """Automated retraining pipeline"""
    # 1. Check if retraining needed
    if not should_retrain(monitor):
        return
    
    # 2. Load new data
    new_data = load_latest_data()
    
    # 3. Train new model
    with mlflow.start_run():
        model = train_model(new_data)
        mlflow.sklearn.log_model(model, "model")
    
    # 4. Evaluate
    if evaluate_model(model) > current_model_performance:
        # 5. Deploy new model
        deploy_model(model)
```

---

## Kubeflow

### Kubeflow Pipelines

```python
try:
    import kfp
    from kfp import dsl
    
    @dsl.pipeline(
        name='ML Pipeline',
        description='End-to-end ML pipeline'
    )
    def ml_pipeline():
        # Data preparation
        prepare_op = dsl.ContainerOp(
            name='prepare',
            image='prepare-image',
            command=['python', 'prepare.py']
        )
        
        # Training
        train_op = dsl.ContainerOp(
            name='train',
            image='train-image',
            command=['python', 'train.py']
        )
        train_op.after(prepare_op)
        
        # Evaluation
        eval_op = dsl.ContainerOp(
            name='evaluate',
            image='eval-image',
            command=['python', 'evaluate.py']
        )
        eval_op.after(train_op)
    
    # Compile and run
    # kfp.Client().create_run_from_pipeline_func(ml_pipeline, arguments={})
except ImportError:
    print("Install: pip install kfp")
```

---

## Apache Kafka for Data Streaming

### Introduction to Kafka

Apache Kafka is a distributed streaming platform for building real-time data pipelines and streaming applications.

### Key Concepts

- **Topics**: Categories of messages
- **Producers**: Send messages to topics
- **Consumers**: Read messages from topics
- **Brokers**: Kafka servers
- **Partitions**: Topics are divided into partitions

### Installing Kafka

```bash
# Download Kafka
wget https://downloads.apache.org/kafka/2.8.0/kafka_2.13-2.8.0.tgz
tar -xzf kafka_2.13-2.8.0.tgz

# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka
bin/kafka-server-start.sh config/server.properties
```

### Python Kafka Producer

```python
from kafka import KafkaProducer
import json

# Create producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Send messages
for i in range(10):
    message = {'id': i, 'data': f'message_{i}'}
    producer.send('ml-events', message)

producer.flush()
producer.close()
```

### Python Kafka Consumer

```python
from kafka import KafkaConsumer
import json

# Create consumer
consumer = KafkaConsumer(
    'ml-events',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    group_id='ml-group'
)

# Consume messages
for message in consumer:
    print(f"Received: {message.value}")
    # Process message
    process_ml_event(message.value)
```

### Use Cases in MLOps

**Real-time Feature Ingestion:**
```python
# Producer: Send features
producer.send('features', {
    'user_id': 123,
    'features': [1.2, 3.4, 5.6],
    'timestamp': '2024-01-01T12:00:00'
})

# Consumer: Process features for model
consumer = KafkaConsumer('features', ...)
for message in consumer:
    features = message.value['features']
    prediction = model.predict([features])
    # Send prediction back
    producer.send('predictions', {'prediction': prediction})
```

**Model Monitoring:**
```python
# Stream predictions for monitoring
producer.send('predictions', {
    'model_version': 'v1.0',
    'prediction': prediction,
    'features': features,
    'timestamp': datetime.now().isoformat()
})
```

---

## Apache Spark for Big Data Processing

### Introduction to Spark

Apache Spark is a unified analytics engine for large-scale data processing. It's essential for big data ML pipelines.

### Key Features

- **Distributed Computing**: Process data across clusters
- **In-Memory Processing**: Faster than Hadoop MapReduce
- **Multiple APIs**: SQL, Streaming, MLlib, GraphX
- **Fault Tolerance**: Automatic recovery

### Installing Spark

```bash
# Download Spark
wget https://archive.apache.org/dist/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz
tar -xzf spark-3.3.0-bin-hadoop3.tgz

# Install PySpark
pip install pyspark
```

### Basic Spark Operations

```python
from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("MLPipeline") \
    .getOrCreate()

# Read data
df = spark.read.csv('data.csv', header=True, inferSchema=True)

# Transformations
df_filtered = df.filter(df['age'] > 25)
df_grouped = df.groupBy('category').agg({'sales': 'sum'})

# Actions
df_filtered.show()
df_grouped.collect()
```

### Spark MLlib

**Training Model:**
```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler

# Prepare features
assembler = VectorAssembler(
    inputCols=['feature1', 'feature2', 'feature3'],
    outputCol='features'
)

# Create model
rf = RandomForestClassifier(
    labelCol='label',
    featuresCol='features',
    numTrees=100
)

# Create pipeline
pipeline = Pipeline(stages=[assembler, rf])

# Train
model = pipeline.fit(train_df)

# Predict
predictions = model.transform(test_df)
```

### Spark Streaming

```python
from pyspark.streaming import StreamingContext

# Create streaming context
ssc = StreamingContext(sparkContext, 1)  # 1 second batches

# Read from Kafka
lines = ssc.socketTextStream("localhost", 9999)

# Process stream
words = lines.flatMap(lambda line: line.split(" "))
pairs = words.map(lambda word: (word, 1))
word_counts = pairs.reduceByKey(lambda a, b: a + b)
word_counts.pprint()

# Start streaming
ssc.start()
ssc.awaitTermination()
```

### Spark for MLOps

**Distributed Training:**
```python
# Train model on distributed data
model = pipeline.fit(train_df.repartition(100))

# Batch predictions
predictions = model.transform(test_df)
predictions.write.parquet('predictions/')
```

**Feature Engineering:**
```python
from pyspark.ml.feature import StandardScaler, OneHotEncoder

# Scale features
scaler = StandardScaler(
    inputCol='features',
    outputCol='scaled_features'
)

# Encode categorical
encoder = OneHotEncoder(
    inputCols=['category'],
    outputCols=['category_encoded']
)
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Not Versioning Data

**Solution**: Use DVC or similar tool

### Pitfall 2: Not Tracking Experiments

**Solution**: Use MLflow or W&B from the start

### Pitfall 3: No Monitoring

**Solution**: Set up monitoring from day one

---

## Key Takeaways

1. **Advanced Tracking**: Hyperparameter tuning, experiment comparison
2. **Feature Stores**: Centralize feature management
3. **Monitoring**: Track model performance and drift
4. **Automation**: Automate retraining pipelines
5. **Kubeflow**: Orchestrate complex ML pipelines

---

**Remember**: Advanced MLOps requires proper tooling and practices!

