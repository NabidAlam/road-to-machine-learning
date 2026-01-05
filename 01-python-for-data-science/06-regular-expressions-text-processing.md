# Regular Expressions and Text Processing - Complete Guide

Comprehensive guide to regular expressions and text processing techniques essential for cleaning and analyzing text data.

## Table of Contents

- [Introduction](#introduction)
- [Regular Expressions Basics](#regular-expressions-basics)
- [Common Regex Patterns](#common-regex-patterns)
- [Text Processing with Pandas](#text-processing-with-pandas)
- [Advanced Text Operations](#advanced-text-operations)
- [Real-World Examples](#real-world-examples)
- [Practice Exercises](#practice-exercises)

---

## Introduction

### Why Text Processing Matters

Text data is everywhere in data science:
- **Customer reviews**: Sentiment analysis, topic extraction
- **Social media**: Tweet analysis, hashtag extraction
- **Web scraping**: HTML parsing, data extraction
- **Data cleaning**: Standardizing formats, removing noise
- **Feature engineering**: Creating text-based features

### Tools We'll Use

- **re module**: Python's built-in regex library
- **Pandas str methods**: Vectorized string operations
- **Regular Expressions**: Pattern matching and extraction

---

## Regular Expressions Basics

### What are Regular Expressions?

**Regular Expressions (regex)** are patterns used to match character combinations in strings. They're powerful for:
- Finding patterns
- Extracting information
- Replacing text
- Validating formats

### Basic Syntax

```python
import re

# Basic pattern matching
text = "Hello, my email is john@example.com"
pattern = r'@\w+\.\w+'  # Match email domain
match = re.search(pattern, text)
if match:
    print(f"Found: {match.group()}")  # Output: @example.com
```

### Common Metacharacters

```python
# . (dot) - matches any character except newline
re.findall(r'.', "abc")  # ['a', 'b', 'c']

# ^ - start of string
re.findall(r'^Hello', "Hello World")  # ['Hello']

# $ - end of string
re.findall(r'World$', "Hello World")  # ['World']

# * - zero or more of preceding character
re.findall(r'ab*', "a ab abb abbb")  # ['a', 'ab', 'abb', 'abbb']

# + - one or more of preceding character
re.findall(r'ab+', "a ab abb abbb")  # ['ab', 'abb', 'abbb']

# ? - zero or one of preceding character
re.findall(r'ab?', "a ab abb")  # ['a', 'ab', 'ab', 'ab']

# {n} - exactly n occurrences
re.findall(r'ab{2}', "ab abb abbb")  # ['abb', 'abb']

# {n,m} - between n and m occurrences
re.findall(r'ab{1,2}', "ab abb abbb")  # ['ab', 'abb', 'abb']
```

### Character Classes

```python
# \d - digit (0-9)
re.findall(r'\d', "Price: $123")  # ['1', '2', '3']

# \w - word character (a-z, A-Z, 0-9, _)
re.findall(r'\w+', "Hello World!")  # ['Hello', 'World']

# \s - whitespace
re.findall(r'\s', "Hello World")  # [' ']

# [abc] - matches a, b, or c
re.findall(r'[aeiou]', "Hello")  # ['e', 'o']

# [a-z] - matches any lowercase letter
re.findall(r'[a-z]+', "Hello World")  # ['ello', 'orld']

# [^abc] - matches anything except a, b, or c
re.findall(r'[^aeiou]', "Hello")  # ['H', 'l', 'l']
```

### Groups and Capturing

```python
# () - capturing group
text = "John Doe, age 30"
pattern = r'(\w+)\s+(\w+),\s+age\s+(\d+)'
match = re.search(pattern, text)
if match:
    print(match.groups())  # ('John', 'Doe', '30')
    print(match.group(1))  # 'John'
    print(match.group(2))  # 'Doe'
    print(match.group(3))  # '30'

# Named groups
pattern = r'(?P<first>\w+)\s+(?P<last>\w+),\s+age\s+(?P<age>\d+)'
match = re.search(pattern, text)
if match:
    print(match.group('first'))  # 'John'
    print(match.group('last'))   # 'Doe'
    print(match.group('age'))    # '30'
```

### Advanced Grouping

**1. Back References:**
Reference previously captured groups in the same pattern.

```python
# Find repeated words
text = "The the cat sat on the mat"
pattern = r'\b(\w+)\s+\1\b'  # \1 refers to first captured group
matches = re.findall(pattern, text, re.IGNORECASE)
print(matches)  # ['the']

# Find repeated patterns
text = "123-123-456"
pattern = r'(\d{3})-\1-(\d{3})'  # First group repeated
match = re.search(pattern, text)
if match:
    print(match.groups())  # ('123', '456')

# Named back references
text = "John John Doe"
pattern = r'(?P<name>\w+)\s+(?P=name)'  # (?P=name) references named group
match = re.search(pattern, text)
if match:
    print(match.group())  # 'John John'
```

**2. Non-Capture Groups:**
Group without capturing (useful for alternation or quantifiers).

```python
# Non-capture group: (?:...)
text = "color colour"
pattern = r'col(?:or|our)'  # Group but don't capture
matches = re.findall(pattern, text)
print(matches)  # ['color', 'colour']

# With quantifiers
text = "abc abcabc abcabcabc"
pattern = r'(?:abc){2,3}'  # Match 'abc' 2-3 times as a group
matches = re.findall(pattern, text)
print(matches)  # ['abcabc', 'abcabcabc']

# Compare with capture group
pattern_capture = r'(abc){2,3}'
matches_capture = re.findall(pattern_capture, text)
print(matches_capture)  # ['abc', 'abc'] - only last repetition captured
```

**3. Lookahead Assertions:**
Match pattern only if followed by another pattern (without consuming it).

```python
# Positive lookahead: (?=...)
text = "cat dog catfish"
pattern = r'cat(?=fish)'  # 'cat' followed by 'fish'
match = re.search(pattern, text)
if match:
    print(match.group())  # 'cat' (from 'catfish')

# Negative lookahead: (?!...)
pattern = r'cat(?!fish)'  # 'cat' NOT followed by 'fish'
matches = re.findall(pattern, text)
print(matches)  # ['cat'] (from 'cat dog', not 'catfish')

# Password validation: at least one digit, one letter
password = "Password123"
pattern = r'^(?=.*\d)(?=.*[a-zA-Z]).{8,}$'  # Multiple lookaheads
is_valid = bool(re.match(pattern, password))
print(f"Valid password: {is_valid}")  # True
```

**4. Lookbehind Assertions:**
Match pattern only if preceded by another pattern (without consuming it).

```python
# Positive lookbehind: (?<=...)
text = "price: $100, cost: $50"
pattern = r'(?<=\$)\d+'  # Digits preceded by '$'
matches = re.findall(pattern, text)
print(matches)  # ['100', '50']

# Negative lookbehind: (?<!...)
text = "price: $100, cost: 50"
pattern = r'(?<!\$)\d+'  # Digits NOT preceded by '$'
matches = re.findall(pattern, text)
print(matches)  # ['50'] (not '100' because it has '$' before)

# Extract words after specific prefix
text = "prefix_word1 suffix_word2 prefix_word3"
pattern = r'(?<=prefix_)\w+'  # Words after 'prefix_'
matches = re.findall(pattern, text)
print(matches)  # ['word1', 'word3']
```

**5. Combined Assertions:**
Use multiple assertions together.

```python
# Extract numbers between $ and ,
text = "price: $100, cost: $50, total: $200"
pattern = r'(?<=\$)(?=\d+)(\d+)(?=,)'  # Between $ and ,
matches = re.findall(pattern, text)
print(matches)  # ['100', '50', '200']

# Word boundaries with lookahead/lookbehind
text = "cat category catfish"
pattern = r'(?<!\w)cat(?!\w)'  # 'cat' as whole word
matches = re.findall(pattern, text)
print(matches)  # ['cat'] (only standalone 'cat')
```

### Regex Flags

Control regex behavior with flags.

```python
# re.IGNORECASE (or re.I): Case-insensitive matching
text = "Hello HELLO hello"
pattern = r'hello'
matches = re.findall(pattern, text, re.IGNORECASE)
print(matches)  # ['Hello', 'HELLO', 'hello']

# re.MULTILINE (or re.M): ^ and $ match start/end of each line
text = "Line 1\nLine 2\nLine 3"
pattern = r'^Line'
matches = re.findall(pattern, text, re.MULTILINE)
print(matches)  # ['Line', 'Line', 'Line']

# re.DOTALL (or re.S): . matches newline too
text = "Start\nEnd"
pattern = r'Start.*End'
match1 = re.search(pattern, text)  # No match (default)
match2 = re.search(pattern, text, re.DOTALL)  # Match
print(f"Without DOTALL: {match1 is None}")  # True
print(f"With DOTALL: {match2 is not None}")  # True

# re.VERBOSE (or re.X): Allow whitespace and comments
pattern = r'''
    \d{3}      # Area code
    -           # Separator
    \d{3}       # Exchange
    -           # Separator
    \d{4}       # Number
'''
text = "123-456-7890"
match = re.search(pattern, text, re.VERBOSE)
print(match.group() if match else None)  # '123-456-7890'

# Multiple flags: Combine with |
text = "Hello\nHELLO\nhello"
pattern = r'^hello$'
matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
print(matches)  # ['Hello', 'HELLO', 'hello']
```

### Advanced Split and Substitution

**1. Advanced Split:**
```python
# Split on multiple delimiters
text = "apple,banana;cherry:grape"
pattern = r'[,;:]'
parts = re.split(pattern, text)
print(parts)  # ['apple', 'banana', 'cherry', 'grape']

# Split with capture groups (include delimiters)
text = "apple,banana;cherry"
pattern = r'([,;])'  # Capture delimiter
parts = re.split(pattern, text)
print(parts)  # ['apple', ',', 'banana', ';', 'cherry']

# Split with maxsplit
text = "one,two,three,four"
parts = re.split(r',', text, maxsplit=2)
print(parts)  # ['one', 'two', 'three,four']
```

**2. Advanced Substitution:**
```python
# Simple substitution
text = "Hello World"
new_text = re.sub(r'World', 'Python', text)
print(new_text)  # 'Hello Python'

# Substitution with function
def replacer(match):
    return match.group().upper()

text = "hello world"
new_text = re.sub(r'\w+', replacer, text)
print(new_text)  # 'HELLO WORLD'

# Substitution with back references
text = "John Doe"
new_text = re.sub(r'(\w+)\s+(\w+)', r'\2, \1', text)  # Swap groups
print(new_text)  # 'Doe, John'

# Named group substitution
text = "2023-12-25"
new_text = re.sub(r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})',
                  r'\g<month>/\g<day>/\g<year>', text)
print(new_text)  # '12/25/2023'

# Count parameter
text = "cat cat cat"
new_text = re.sub(r'cat', 'dog', text, count=2)  # Replace first 2
print(new_text)  # 'dog dog cat'
```

**3. Real-World Examples:**
```python
# Format phone numbers
def format_phone(text):
    pattern = r'(\d{3})(\d{3})(\d{4})'
    return re.sub(pattern, r'(\1) \2-\3', text)

text = "Call 1234567890"
formatted = format_phone(text)
print(formatted)  # 'Call (123) 456-7890'

# Remove duplicate words
def remove_duplicates(text):
    pattern = r'\b(\w+)(\s+\1)+\b'
    return re.sub(pattern, r'\1', text, flags=re.IGNORECASE)

text = "the the cat sat on the the mat"
cleaned = remove_duplicates(text)
print(cleaned)  # 'the cat sat on the mat'

# Extract and format dates
def format_dates(text):
    pattern = r'(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})'
    def replacer(match):
        return f"{match.group('year')}-{match.group('month').zfill(2)}-{match.group('day').zfill(2)}"
    return re.sub(pattern, replacer, text)

text = "Event on 12/25/2023"
formatted = format_dates(text)
print(formatted)  # 'Event on 2023-12-25'
```

---

## Common Regex Patterns

### Email Validation

```python
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

# Test
emails = ["john@example.com", "invalid.email", "test@domain"]
for email in emails:
    print(f"{email}: {validate_email(email)}")
```

### Phone Numbers

```python
def extract_phone_numbers(text):
    # US phone number pattern
    pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    return re.findall(pattern, text)

text = "Call me at (555) 123-4567 or 555.987.6543"
phones = extract_phone_numbers(text)
print(phones)  # ['(555) 123-4567', '555.987.6543']
```

### URLs

```python
def extract_urls(text):
    pattern = r'https?://[^\s]+'
    return re.findall(pattern, text)

text = "Visit https://example.com or http://test.org for more info"
urls = extract_urls(text)
print(urls)  # ['https://example.com', 'http://test.org']
```

### Dates

```python
def extract_dates(text):
    # Various date formats
    pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
    return re.findall(pattern, text)

text = "Dates: 12/25/2023, 01-15-24, 3/5/2024"
dates = extract_dates(text)
print(dates)  # ['12/25/2023', '01-15-24', '3/5/2024']
```

### Credit Cards

```python
def extract_credit_cards(text):
    # Basic pattern (masked for security)
    pattern = r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}'
    return re.findall(pattern, text)

text = "Card: 1234 5678 9012 3456"
cards = extract_credit_cards(text)
print(cards)
```

---

## Text Processing with Pandas

### String Methods

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['John Doe', 'JANE SMITH', '  Bob Wilson  '],
    'email': ['john@example.com', 'JANE@TEST.COM', 'bob@email.org'],
    'phone': ['(555) 123-4567', '555-987-6543', '555.111.2222']
})

# Basic string operations
df['name_upper'] = df['name'].str.upper()
df['name_lower'] = df['name'].str.lower()
df['name_title'] = df['name'].str.title()
df['name_stripped'] = df['name'].str.strip()

# String length
df['name_length'] = df['name'].str.len()

# Replace
df['phone_clean'] = df['phone'].str.replace(r'[^\d]', '', regex=True)

# Split
df[['first_name', 'last_name']] = df['name'].str.split(' ', expand=True, n=1)

print(df)
```

### Pattern Matching

```python
# Check if contains pattern
df['has_digits'] = df['name'].str.contains(r'\d', regex=True)

# Extract patterns
df['email_domain'] = df['email'].str.extract(r'@(\w+\.\w+)')

# Find all matches
df['all_digits'] = df['phone'].str.findall(r'\d')

# Count occurrences
df['digit_count'] = df['phone'].str.count(r'\d')
```

### Advanced Text Operations

```python
# Extract multiple groups
df['name_parts'] = df['name'].str.extract(r'(\w+)\s+(\w+)')
df.columns = ['first', 'last']  # Rename columns

# Replace with function
def format_phone(match):
    digits = match.group(0).replace('-', '').replace('(', '').replace(')', '').replace('.', '').replace(' ', '')
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"

df['phone_formatted'] = df['phone'].str.replace(
    r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
    format_phone,
    regex=True
)
```

---

## Advanced Text Operations

### Text Cleaning Pipeline

```python
def clean_text(text):
    """
    Comprehensive text cleaning function
    """
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

# Apply to DataFrame
df['text_clean'] = df['text'].apply(clean_text)
```

### Extract Information

```python
def extract_entities(text):
    """
    Extract various entities from text
    """
    entities = {
        'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
        'urls': re.findall(r'https?://[^\s]+', text),
        'phones': re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text),
        'dates': re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text),
        'hashtags': re.findall(r'#\w+', text),
        'mentions': re.findall(r'@\w+', text)
    }
    return entities

# Example
text = "Contact john@example.com or visit https://site.com. Call (555) 123-4567. #data #science @user"
entities = extract_entities(text)
print(entities)
```

### Text Normalization

```python
def normalize_text(text):
    """
    Normalize text for analysis
    """
    # Remove URLs
    text = re.sub(r'https?://[^\s]+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

df['text_normalized'] = df['text'].apply(normalize_text)
```

---

## Real-World Examples

### Example 1: Cleaning Customer Data

```python
# Clean customer names
def clean_customer_name(name):
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', str(name))
    # Title case
    name = name.strip().title()
    # Remove special characters except spaces and hyphens
    name = re.sub(r'[^a-zA-Z\s-]', '', name)
    return name

df['customer_name'] = df['customer_name'].apply(clean_customer_name)
```

### Example 2: Extracting Product Codes

```python
# Extract product codes (e.g., PROD-12345)
def extract_product_code(text):
    pattern = r'PROD-?\d{5}'
    match = re.search(pattern, text.upper())
    return match.group(0) if match else None

df['product_code'] = df['description'].apply(extract_product_code)
```

### Example 3: Parsing Log Files

```python
# Parse log entries
log_line = "2024-01-15 10:30:45 INFO User login successful user_id=12345"

pattern = r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+(\w+)\s+(.+?)(?:\s+user_id=(\d+))?$'
match = re.match(pattern, log_line)

if match:
    date, time, level, message, user_id = match.groups()
    print(f"Date: {date}, Time: {time}, Level: {level}")
    print(f"Message: {message}, User ID: {user_id}")
```

### Example 4: Social Media Text Processing

```python
def process_tweet(tweet):
    """
    Process Twitter-like text
    """
    # Extract hashtags
    hashtags = re.findall(r'#\w+', tweet)
    
    # Extract mentions
    mentions = re.findall(r'@\w+', tweet)
    
    # Remove hashtags and mentions for analysis
    clean_tweet = re.sub(r'#\w+|@\w+', '', tweet)
    clean_tweet = re.sub(r'\s+', ' ', clean_tweet).strip()
    
    return {
        'hashtags': hashtags,
        'mentions': mentions,
        'clean_text': clean_tweet,
        'hashtag_count': len(hashtags),
        'mention_count': len(mentions)
    }

tweet = "Great #datascience tutorial by @expert! #python #ml"
result = process_tweet(tweet)
print(result)
```

---

## Practice Exercises

### Exercise 1: Email Validation

Create a function to validate and extract emails from text.

```python
def extract_and_validate_emails(text):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(pattern, text)
    return emails

text = "Contact us at support@company.com or sales@business.org"
emails = extract_and_validate_emails(text)
print(emails)
```

### Exercise 2: Phone Number Standardization

Standardize phone numbers to a consistent format.

```python
def standardize_phone(phone):
    # Extract digits only
    digits = re.sub(r'\D', '', str(phone))
    
    # Format as (XXX) XXX-XXXX
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return phone

df['phone_standardized'] = df['phone'].apply(standardize_phone)
```

### Exercise 3: Text Cleaning

Clean messy text data.

```python
def clean_messy_text(text):
    # Your implementation here
    pass
```

---

## Resources

### Documentation

- [Python re module](https://docs.python.org/3/library/re.html)
- [Pandas String Methods](https://pandas.pydata.org/docs/user_guide/text.html)
- [Regex101](https://regex101.com/) - Test regex patterns online

### Tools

- **regex101.com**: Test and debug regex patterns
- **regexr.com**: Interactive regex learning

### Cheat Sheets

- [Regex Cheat Sheet](https://www.rexegg.com/regex-quickstart.html)

---

## Key Takeaways

1. **Regex is Powerful**: Learn common patterns
2. **Pandas Integration**: Use `.str` methods for vectorized operations
3. **Practice**: Text processing requires hands-on experience
4. **Test Patterns**: Use regex testers to validate patterns
5. **Start Simple**: Build complex patterns from simple ones

---

**Remember**: Regular expressions are essential for text data. Master the basics and practice with real-world examples!

