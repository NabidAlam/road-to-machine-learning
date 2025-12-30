# Data Sources and Integration - Complete Guide

Comprehensive guide to working with various data sources: APIs, databases, web scraping, and file formats.

## Table of Contents

- [Introduction](#introduction)
- [Working with APIs](#working-with-apis)
- [Database Integration](#database-integration)
- [Web Scraping](#web-scraping)
- [File Formats](#file-formats)
- [Data Integration Best Practices](#data-integration-best-practices)
- [Practice Exercises](#practice-exercises)

---

## Introduction

### Common Data Sources

In real-world data science, data comes from various sources:
- **APIs**: REST APIs, web services
- **Databases**: SQL databases (PostgreSQL, MySQL), NoSQL (MongoDB)
- **Files**: CSV, Excel, JSON, XML, Parquet
- **Web**: Web scraping, HTML parsing
- **Cloud Storage**: AWS S3, Google Cloud Storage
- **Streaming**: Real-time data streams

### Why This Matters

- **Real Data**: Most projects require fetching data from multiple sources
- **Automation**: APIs and databases enable automated data collection
- **Scale**: Databases handle large datasets efficiently
- **Fresh Data**: APIs provide up-to-date information

---

## Working with APIs

### REST APIs with Requests

```python
import requests
import pandas as pd
import json

# Basic GET request
response = requests.get('https://api.example.com/data')
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")

# With parameters
params = {'key': 'value', 'page': 1}
response = requests.get('https://api.example.com/data', params=params)

# With headers
headers = {
    'Authorization': 'Bearer YOUR_TOKEN',
    'Content-Type': 'application/json'
}
response = requests.get('https://api.example.com/data', headers=headers)

# POST request
data = {'name': 'John', 'age': 30}
response = requests.post('https://api.example.com/data', json=data)

# Convert to DataFrame
if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)
    print(df.head())
```

### Handling API Responses

```python
def fetch_api_data(url, params=None, headers=None, max_retries=3):
    """
    Fetch data from API with error handling and retries
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()  # Raise exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
    return None

# Example: Fetching paginated data
def fetch_all_pages(base_url, params=None, max_pages=100):
    """
    Fetch all pages from a paginated API
    """
    all_data = []
    page = 1
    
    while page <= max_pages:
        if params:
            params['page'] = page
        else:
            params = {'page': page}
        
        data = fetch_api_data(base_url, params=params)
        
        if not data or len(data) == 0:
            break
        
        all_data.extend(data)
        page += 1
    
    return pd.DataFrame(all_data)

# Example usage
df = fetch_all_pages('https://api.example.com/data')
```

### Real-World Example: Weather API

```python
import requests
import pandas as pd
from datetime import datetime

def get_weather_data(city, api_key):
    """
    Fetch weather data from OpenWeatherMap API
    """
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if response.status_code == 200:
        return {
            'city': data['name'],
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'description': data['weather'][0]['description'],
            'timestamp': datetime.now()
        }
    else:
        print(f"Error: {data.get('message', 'Unknown error')}")
        return None

# Example
weather = get_weather_data('London', 'YOUR_API_KEY')
if weather:
    df = pd.DataFrame([weather])
    print(df)
```

---

## Database Integration

### SQL Databases with SQLAlchemy

```python
from sqlalchemy import create_engine, text
import pandas as pd

# Create connection
# PostgreSQL
engine = create_engine('postgresql://user:password@localhost/dbname')

# MySQL
engine = create_engine('mysql+pymysql://user:password@localhost/dbname')

# SQLite (file-based)
engine = create_engine('sqlite:///database.db')

# Read data
query = "SELECT * FROM table_name LIMIT 100"
df = pd.read_sql(query, engine)
print(df.head())

# Write data
df.to_sql('new_table', engine, if_exists='replace', index=False)

# Execute custom queries
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM table_name"))
    count = result.fetchone()[0]
    print(f"Total rows: {count}")
```

### Advanced Database Operations

```python
def read_sql_with_chunks(query, engine, chunk_size=10000):
    """
    Read large datasets in chunks
    """
    chunks = []
    for chunk in pd.read_sql(query, engine, chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

# Example: Complex query
def get_sales_by_category(engine, start_date, end_date):
    """
    Execute complex SQL query
    """
    query = """
    SELECT 
        category,
        SUM(amount) as total_sales,
        COUNT(*) as transaction_count,
        AVG(amount) as avg_amount
    FROM sales
    WHERE date BETWEEN :start_date AND :end_date
    GROUP BY category
    ORDER BY total_sales DESC
    """
    
    df = pd.read_sql(
        query, 
        engine, 
        params={'start_date': start_date, 'end_date': end_date}
    )
    return df

# Example usage
df = get_sales_by_category(engine, '2023-01-01', '2023-12-31')
print(df)
```

### NoSQL: MongoDB

```python
from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['database_name']
collection = db['collection_name']

# Read data
cursor = collection.find({'status': 'active'})
df = pd.DataFrame(list(cursor))
print(df.head())

# Write data
data = df.to_dict('records')
collection.insert_many(data)

# Query examples
# Find documents
results = collection.find({'age': {'$gt': 25}})

# Aggregate
pipeline = [
    {'$match': {'status': 'active'}},
    {'$group': {'_id': '$category', 'count': {'$sum': 1}}}
]
results = collection.aggregate(pipeline)
df = pd.DataFrame(list(results))
```

---

## Web Scraping

### BeautifulSoup for HTML Parsing

```python
from bs4 import BeautifulSoup
import requests
import pandas as pd

def scrape_table(url):
    """
    Scrape HTML table from webpage
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find table
    table = soup.find('table')
    
    # Extract headers
    headers = [th.text.strip() for th in table.find_all('th')]
    
    # Extract rows
    rows = []
    for tr in table.find_all('tr')[1:]:  # Skip header
        row = [td.text.strip() for td in tr.find_all('td')]
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df

# Example
df = scrape_table('https://example.com/table')
print(df.head())
```

### Selenium for Dynamic Content

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

def scrape_dynamic_content(url):
    """
    Scrape content from JavaScript-rendered pages
    """
    # Setup driver (requires ChromeDriver)
    driver = webdriver.Chrome()
    driver.get(url)
    
    # Wait for content to load
    wait = WebDriverWait(driver, 10)
    element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "content")))
    
    # Extract data
    data = []
    items = driver.find_elements(By.CLASS_NAME, "item")
    
    for item in items:
        title = item.find_element(By.CLASS_NAME, "title").text
        price = item.find_element(By.CLASS_NAME, "price").text
        data.append({'title': title, 'price': price})
    
    driver.quit()
    return pd.DataFrame(data)

# Note: Requires ChromeDriver installation
# df = scrape_dynamic_content('https://example.com')
```

### Web Scraping Best Practices

```python
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session_with_retries():
    """
    Create requests session with retry strategy
    """
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def scrape_with_delay(url, delay_range=(1, 3)):
    """
    Scrape with random delay to be respectful
    """
    session = create_session_with_retries()
    response = session.get(url)
    
    # Random delay
    delay = random.uniform(*delay_range)
    time.sleep(delay)
    
    return response

# Always respect robots.txt and terms of service
```

---

## File Formats

### CSV

```python
# Reading CSV
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', sep=';')  # Custom separator
df = pd.read_csv('data.csv', encoding='latin-1')  # Handle encoding
df = pd.read_csv('data.csv', skiprows=2)  # Skip rows
df = pd.read_csv('data.csv', nrows=1000)  # Read first N rows

# Writing CSV
df.to_csv('output.csv', index=False)
df.to_csv('output.csv', index=False, encoding='utf-8-sig')  # Excel-friendly
```

### Excel

```python
# Reading Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_excel('data.xlsx', sheet_name=0)  # First sheet
df = pd.read_excel('data.xlsx', sheet_name=[0, 1])  # Multiple sheets

# Reading multiple sheets
excel_file = pd.ExcelFile('data.xlsx')
all_sheets = {}
for sheet_name in excel_file.sheet_names:
    all_sheets[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)

# Writing Excel
df.to_excel('output.xlsx', sheet_name='Data', index=False)

# Multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1', index=False)
    df2.to_excel(writer, sheet_name='Sheet2', index=False)
```

### JSON

```python
# Reading JSON
df = pd.read_json('data.json')
df = pd.read_json('data.json', orient='records')  # List of records
df = pd.read_json('data.json', lines=True)  # JSONL format

# From API response
response = requests.get('https://api.example.com/data')
df = pd.json_normalize(response.json())  # Flatten nested JSON

# Writing JSON
df.to_json('output.json', orient='records')
df.to_json('output.json', orient='records', indent=2)  # Pretty print
```

### Parquet (Efficient for Large Data)

```python
# Reading Parquet
df = pd.read_parquet('data.parquet')
df = pd.read_parquet('data.parquet', engine='pyarrow')

# Writing Parquet
df.to_parquet('output.parquet')
df.to_parquet('output.parquet', compression='snappy')  # Compressed

# Advantages: Fast, compressed, preserves data types
```

### XML

```python
import xml.etree.ElementTree as ET

def parse_xml_to_df(xml_file):
    """
    Parse XML file to DataFrame
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    data = []
    for item in root.findall('item'):
        row = {}
        for child in item:
            row[child.tag] = child.text
        data.append(row)
    
    return pd.DataFrame(data)

# Example
df = parse_xml_to_df('data.xml')
```

---

## Data Integration Best Practices

### 1. Error Handling

```python
def safe_read_data(source, source_type='csv', **kwargs):
    """
    Safely read data with error handling
    """
    try:
        if source_type == 'csv':
            return pd.read_csv(source, **kwargs)
        elif source_type == 'excel':
            return pd.read_csv(source, **kwargs)
        elif source_type == 'json':
            return pd.read_json(source, **kwargs)
        elif source_type == 'sql':
            return pd.read_sql(source, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    except FileNotFoundError:
        print(f"Error: File {source} not found")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File {source} is empty")
        return None
    except Exception as e:
        print(f"Error reading {source}: {e}")
        return None
```

### 2. Data Validation

```python
def validate_data(df, required_columns=None, data_types=None):
    """
    Validate data after loading
    """
    errors = []
    
    # Check required columns
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")
    
    # Check data types
    if data_types:
        for col, expected_type in data_types.items():
            if col in df.columns:
                if not pd.api.types.is_dtype_equal(df[col].dtype, expected_type):
                    errors.append(f"Column {col} has wrong type: {df[col].dtype} != {expected_type}")
    
    if errors:
        raise ValueError("Data validation failed:\n" + "\n".join(errors))
    
    return True

# Example
validate_data(df, 
              required_columns=['id', 'name', 'age'],
              data_types={'age': 'int64', 'name': 'object'})
```

### 3. Data Pipeline

```python
def create_data_pipeline(config):
    """
    Create automated data pipeline
    """
    all_data = []
    
    # Fetch from multiple sources
    for source in config['sources']:
        if source['type'] == 'api':
            data = fetch_api_data(source['url'], source.get('params'))
        elif source['type'] == 'database':
            data = pd.read_sql(source['query'], source['engine'])
        elif source['type'] == 'file':
            data = pd.read_csv(source['path'])
        else:
            continue
        
        # Transform
        if 'transform' in source:
            data = source['transform'](data)
        
        all_data.append(data)
    
    # Combine
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Final processing
    combined_df = combined_df.drop_duplicates()
    combined_df = combined_df.dropna(subset=config.get('required_columns', []))
    
    return combined_df

# Example config
config = {
    'sources': [
        {'type': 'api', 'url': 'https://api.example.com/data'},
        {'type': 'file', 'path': 'local_data.csv'}
    ],
    'required_columns': ['id', 'name']
}

df = create_data_pipeline(config)
```

---

## Practice Exercises

### Exercise 1: API Integration

1. Find a public API (e.g., JSONPlaceholder, REST Countries)
2. Fetch data from the API
3. Convert to DataFrame
4. Perform basic analysis

### Exercise 2: Database Query

1. Set up a local SQLite database
2. Create a table and insert sample data
3. Query data using pandas
4. Perform aggregations

### Exercise 3: Web Scraping

1. Scrape a simple HTML table
2. Extract specific information
3. Clean and structure the data
4. Save to CSV

---

## Resources

### Libraries

- **requests**: HTTP library for APIs
- **SQLAlchemy**: SQL toolkit
- **pymongo**: MongoDB driver
- **BeautifulSoup**: HTML parsing
- **Selenium**: Browser automation

### APIs for Practice

- [JSONPlaceholder](https://jsonplaceholder.typicode.com/)
- [REST Countries](https://restcountries.com/)
- [OpenWeatherMap](https://openweathermap.org/api)
- [GitHub API](https://docs.github.com/en/rest)

### Documentation

- [Requests Documentation](https://requests.readthedocs.io/)
- [SQLAlchemy Documentation](https://www.sqlalchemy.org/)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

---

## Key Takeaways

1. **APIs are Common**: Most modern data comes from APIs
2. **Databases are Essential**: Learn SQL for data extraction
3. **Web Scraping**: Useful but respect terms of service
4. **Error Handling**: Always handle errors gracefully
5. **Validation**: Validate data after loading
6. **Automation**: Build pipelines for repeated tasks

---

**Remember**: Data integration is a crucial skill. Practice with real APIs and databases to master it!

