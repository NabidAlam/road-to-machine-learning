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

**Note**: For a comprehensive web scraping guide covering Requests, Beautiful Soup, Selenium, Scrapy, and advanced techniques, see [Web Scraping Guide](../resources/web_scraping_guide.md).

This section provides a quick overview. The full guide includes:
- Introduction to web scraping (types, ethics, advantages/disadvantages)
- Primer on web technologies (HTTP, client-server architecture)
- Mastering Requests library (GET, POST, headers, error handling)
- Beautiful Soup for HTML parsing (complete guide)
- Selenium for dynamic content (waits, scrolling, iframes, alerts)
- Scrapy for large-scale scraping (spiders, pipelines, middleware)
- Handling challenges (CAPTCHAs, rate limiting)
- Best practices and real-world projects

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

### Advanced Selenium Web Scraping

Selenium is essential for scraping JavaScript-rendered content that BeautifulSoup cannot handle.

#### Installation and Setup

```bash
# Install Selenium
pip install selenium

# Download ChromeDriver
# https://chromedriver.chromium.org/downloads
# Or use webdriver-manager
pip install webdriver-manager
```

#### Basic Selenium Setup

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# Setup Chrome options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run in background
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--disable-blink-features=AutomationControlled')
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)

# Setup driver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)
```

#### Advanced Selenium Techniques

**1. Handling Dynamic Content:**
```python
def wait_for_element(driver, by, value, timeout=10):
    """Wait for element to be present"""
    wait = WebDriverWait(driver, timeout)
    return wait.until(EC.presence_of_element_located((by, value)))

def wait_for_clickable(driver, by, value, timeout=10):
    """Wait for element to be clickable"""
    wait = WebDriverWait(driver, timeout)
    return wait.until(EC.element_to_be_clickable((by, value)))

# Usage
driver.get(url)
element = wait_for_element(driver, By.CLASS_NAME, "content")
```

**2. Scrolling and Pagination:**
```python
def scroll_to_load_content(driver, scroll_pause_time=2):
    """Scroll page to load dynamic content"""
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while True:
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        # Wait for new content
        time.sleep(scroll_pause_time)
        
        # Calculate new scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        if new_height == last_height:
            break
        last_height = new_height

# Scroll to load all content
driver.get(url)
scroll_to_load_content(driver)
```

**3. Handling Multiple Windows/Tabs:**
```python
# Get current window
main_window = driver.current_window_handle

# Click link that opens new tab
link = driver.find_element(By.LINK_TEXT, "Open New Tab")
link.click()

# Switch to new window
for window_handle in driver.window_handles:
    if window_handle != main_window:
        driver.switch_to.window(window_handle)
        break

# Scrape new tab
data = scrape_data(driver)

# Close tab and switch back
driver.close()
driver.switch_to.window(main_window)
```

**4. Handling Frames:**
```python
# Switch to iframe
iframe = driver.find_element(By.ID, "iframe_id")
driver.switch_to.frame(iframe)

# Scrape content in iframe
content = driver.find_element(By.CLASS_NAME, "content").text

# Switch back to main content
driver.switch_to.default_content()
```

**5. Handling Dropdowns and Select Elements:**
```python
from selenium.webdriver.support.ui import Select

# Find select element
select_element = driver.find_element(By.ID, "dropdown_id")
select = Select(select_element)

# Select by value
select.select_by_value("option_value")

# Select by visible text
select.select_by_visible_text("Option Text")

# Get all options
options = select.options
for option in options:
    print(option.text)
```

**6. Handling Alerts and Popups:**
```python
# Wait for alert
alert = WebDriverWait(driver, 10).until(EC.alert_is_present())
alert_text = alert.text
alert.accept()  # or alert.dismiss()

# Handle popup windows
popup = driver.switch_to.alert
popup.dismiss()
```

#### Smartprix Example (E-commerce Scraping)

```python
def scrape_smartprix_products(search_term, max_pages=5):
    """
    Scrape product data from Smartprix
    """
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    all_products = []
    
    try:
        # Navigate to search page
        search_url = f"https://www.smartprix.com/search?q={search_term}"
        driver.get(search_url)
        
        # Wait for products to load
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "product")))
        
        for page in range(1, max_pages + 1):
            # Scroll to load all products
            scroll_to_load_content(driver)
            
            # Find all product elements
            products = driver.find_elements(By.CLASS_NAME, "product")
            
            for product in products:
                try:
                    # Extract product data
                    name = product.find_element(By.CLASS_NAME, "product-name").text
                    price = product.find_element(By.CLASS_NAME, "price").text
                    rating = product.find_element(By.CLASS_NAME, "rating").text
                    link = product.find_element(By.TAG_NAME, "a").get_attribute("href")
                    
                    all_products.append({
                        'name': name,
                        'price': price,
                        'rating': rating,
                        'link': link
                    })
                except Exception as e:
                    print(f"Error extracting product: {e}")
                    continue
            
            # Go to next page
            if page < max_pages:
                try:
                    next_button = wait_for_clickable(driver, By.CLASS_NAME, "next-page")
                    next_button.click()
                    time.sleep(2)  # Wait for page load
                except:
                    print(f"No more pages after page {page}")
                    break
        
        return pd.DataFrame(all_products)
    
    finally:
        driver.quit()

# Usage
# df = scrape_smartprix_products("laptop", max_pages=3)
```

#### Advanced Selenium Patterns

**1. Retry Logic:**
```python
from functools import wraps
import time

def retry_on_exception(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry_on_exception(max_retries=3)
def scrape_with_retry(driver, url):
    driver.get(url)
    return driver.find_element(By.CLASS_NAME, "content").text
```

**2. Parallel Scraping:**
```python
from concurrent.futures import ThreadPoolExecutor

def scrape_url(url):
    """Scrape single URL"""
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    try:
        driver.get(url)
        data = extract_data(driver)
        return data
    finally:
        driver.quit()

# Scrape multiple URLs in parallel
urls = ["url1", "url2", "url3"]
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(scrape_url, urls))
```

**3. Stealth Mode (Avoid Detection):**
```python
from selenium_stealth import stealth

# Setup driver
driver = webdriver.Chrome()

# Apply stealth
stealth(driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
)

driver.get(url)
```

#### Best Practices

1. **Always use waits**: Don't use `time.sleep()` - use WebDriverWait
2. **Handle exceptions**: Wrap scraping in try-except
3. **Respect robots.txt**: Check before scraping
4. **Add delays**: Be respectful to servers
5. **Use headless mode**: For production
6. **Clean up**: Always quit driver
7. **Handle dynamic content**: Use explicit waits

#### Common Issues and Solutions

**Issue 1: Element not found**
```python
# Solution: Use explicit waits
element = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, "element_id"))
)
```

**Issue 2: Stale element reference**
```python
# Solution: Re-find element
try:
    element.click()
except StaleElementReferenceException:
    element = driver.find_element(By.ID, "element_id")
    element.click()
```

**Issue 3: Timeout errors**
```python
# Solution: Increase timeout or check element existence
try:
    element = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.ID, "element_id"))
    )
except TimeoutException:
    print("Element not found within timeout")
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

## ETL with AWS RDS

### Introduction to ETL

ETL (Extract, Transform, Load) is a process for:
- **Extract**: Get data from source systems
- **Transform**: Clean, validate, and transform data
- **Load**: Load data into target database (AWS RDS)

### AWS RDS Overview

Amazon RDS (Relational Database Service) is a managed database service supporting:
- MySQL
- PostgreSQL
- MariaDB
- Oracle
- SQL Server

### Setting Up AWS RDS

**1. Create RDS Instance:**
```python
import boto3

# Create RDS client
rds_client = boto3.client('rds', region_name='us-east-1')

# Create database instance (example - use AWS Console for actual setup)
# This is typically done via AWS Console or CloudFormation
```

**2. Connect to RDS:**
```python
import pymysql
import pandas as pd

# Connection parameters
host = 'your-rds-endpoint.region.rds.amazonaws.com'
port = 3306
user = 'admin'
password = 'your-password'
database = 'your-database'

# Connect to MySQL RDS
connection = pymysql.connect(
    host=host,
    port=port,
    user=user,
    password=password,
    database=database
)
```

### ETL Pipeline Example

**Extract:**
```python
def extract_from_source():
    """Extract data from source (CSV, API, etc.)"""
    # Example: Extract from CSV
    df = pd.read_csv('source_data.csv')
    return df

# Or extract from API
def extract_from_api():
    import requests
    response = requests.get('https://api.example.com/data')
    data = response.json()
    return pd.DataFrame(data)
```

**Transform:**
```python
def transform_data(df):
    """Clean and transform data"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(0)
    
    # Data type conversions
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Add calculated columns
    df['total'] = df['quantity'] * df['price']
    
    # Filter data
    df = df[df['amount'] > 0]
    
    return df
```

**Load:**
```python
def load_to_rds(df, table_name, connection):
    """Load transformed data to RDS"""
    try:
        # Use pandas to_sql for easy loading
        df.to_sql(
            name=table_name,
            con=connection,
            if_exists='append',  # or 'replace'
            index=False,
            method='multi'  # Batch insert for performance
        )
        print(f"Successfully loaded {len(df)} rows to {table_name}")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Or use SQL directly
def load_with_sql(df, table_name, connection):
    """Load using SQL INSERT statements"""
    cursor = connection.cursor()
    
    for _, row in df.iterrows():
        sql = f"""
        INSERT INTO {table_name} (col1, col2, col3)
        VALUES (%s, %s, %s)
        """
        cursor.execute(sql, (row['col1'], row['col2'], row['col3']))
    
    connection.commit()
    cursor.close()
```

### Complete ETL Pipeline

```python
import pandas as pd
import pymysql
from sqlalchemy import create_engine

def etl_pipeline():
    """Complete ETL pipeline"""
    
    # 1. Extract
    print("Extracting data...")
    source_df = extract_from_source()
    print(f"Extracted {len(source_df)} rows")
    
    # 2. Transform
    print("Transforming data...")
    transformed_df = transform_data(source_df)
    print(f"Transformed to {len(transformed_df)} rows")
    
    # 3. Load
    print("Loading to RDS...")
    # Create SQLAlchemy engine
    engine = create_engine(
        f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'
    )
    
    load_to_rds(transformed_df, 'target_table', engine)
    print("ETL pipeline completed successfully!")

# Run pipeline
if __name__ == '__main__':
    etl_pipeline()
```

### Scheduled ETL with AWS Lambda

```python
import json
import boto3

def lambda_handler(event, context):
    """AWS Lambda function for scheduled ETL"""
    try:
        # Run ETL pipeline
        etl_pipeline()
        
        return {
            'statusCode': 200,
            'body': json.dumps('ETL completed successfully')
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }
```

### Best Practices

1. **Error Handling**: Wrap operations in try-except
2. **Logging**: Log each ETL step
3. **Validation**: Validate data before loading
4. **Incremental Loads**: Load only new/changed data
5. **Monitoring**: Monitor ETL job performance
6. **Backup**: Backup data before transformations

### Advanced: Using AWS Glue

AWS Glue is a serverless ETL service:

```python
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Read from source
datasource = glueContext.create_dynamic_frame.from_catalog(
    database="source_db",
    table_name="source_table"
)

# Transform
transformed = ApplyMapping.apply(
    frame=datasource,
    mappings=[("col1", "string", "new_col1", "string")]
)

# Write to RDS
glueContext.write_dynamic_frame.from_jdbc_conf(
    frame=transformed,
    catalog_connection="rds-connection",
    connection_options={"dbtable": "target_table"}
)
```

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

