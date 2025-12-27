# Python Basics for Machine Learning

Complete guide to Python fundamentals needed for machine learning and data science.

## Table of Contents

- [Variables and Data Types](#variables-and-data-types)
- [Control Flow](#control-flow)
- [Functions](#functions)
- [Data Structures](#data-structures)
- [File I/O](#file-io)
- [Error Handling](#error-handling)
- [Object-Oriented Programming](#object-oriented-programming)
- [Practice Exercises](#practice-exercises)

---

## Variables and Data Types

### Variables

Variables store data values. In Python, you don't need to declare variable types.

```python
# Assigning values
name = "Alice"
age = 25
height = 5.6
is_student = True

print(name)    # Output: Alice
print(age)     # Output: 25
print(height)  # Output: 5.6
print(is_student)  # Output: True
```

### Data Types

**1. Numbers**
```python
# Integers
x = 10
y = -5

# Floats (decimals)
pi = 3.14
temperature = -10.5

# Complex numbers
z = 3 + 4j

# Type checking
print(type(x))  # Output: <class 'int'>
print(type(pi))  # Output: <class 'float'>
```

**2. Strings**
```python
# Single or double quotes
name1 = "Alice"
name2 = 'Bob'

# String operations
full_name = name1 + " " + name2  # Concatenation
print(full_name)  # Output: Alice Bob

# String methods
text = "Hello World"
print(text.upper())      # Output: HELLO WORLD
print(text.lower())      # Output: hello world
print(text.replace("World", "Python"))  # Output: Hello Python
print(len(text))         # Output: 11
```

**3. Booleans**
```python
is_true = True
is_false = False

# Boolean operations
result = is_true and is_false  # False
result = is_true or is_false   # True
result = not is_true           # False
```

**4. Type Conversion**
```python
# Convert between types
x = "123"
y = int(x)      # Convert to integer: 123
z = float(x)    # Convert to float: 123.0
w = str(123)    # Convert to string: "123"
```

---

## Control Flow

### If/Else Statements

```python
age = 18

if age >= 18:
    print("You are an adult")
elif age >= 13:
    print("You are a teenager")
else:
    print("You are a child")

# Output: You are an adult
```

**Comparison Operators:**
```python
x = 10
y = 5

print(x > y)   # True
print(x < y)   # False
print(x == y)  # False (equality)
print(x != y)  # True (not equal)
print(x >= y)  # True
print(x <= y)  # False
```

### Loops

**For Loop:**
```python
# Iterate over a list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Output:
# apple
# banana
# cherry

# Using range
for i in range(5):
    print(i)

# Output: 0, 1, 2, 3, 4

# With index
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# Output:
# 0: apple
# 1: banana
# 2: cherry
```

**While Loop:**
```python
count = 0
while count < 5:
    print(count)
    count += 1

# Output: 0, 1, 2, 3, 4
```

**Loop Control:**
```python
# Break: exit loop
for i in range(10):
    if i == 5:
        break
    print(i)
# Output: 0, 1, 2, 3, 4

# Continue: skip iteration
for i in range(5):
    if i == 2:
        continue
    print(i)
# Output: 0, 1, 3, 4
```

---

## Functions

### Defining Functions

```python
def greet(name):
    """This function greets a person"""
    return f"Hello, {name}!"

message = greet("Alice")
print(message)  # Output: Hello, Alice!
```

### Function Parameters

```python
# Default parameters
def power(base, exponent=2):
    return base ** exponent

print(power(3))      # Output: 9 (3^2)
print(power(3, 3))   # Output: 27 (3^3)

# Keyword arguments
def introduce(name, age, city):
    return f"{name} is {age} years old and lives in {city}"

print(introduce(age=25, city="New York", name="Alice"))
# Output: Alice is 25 years old and lives in New York
```

### Lambda Functions

```python
# Anonymous functions
square = lambda x: x ** 2
print(square(5))  # Output: 25

# Common use: with map, filter
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(squared)  # Output: [1, 4, 9, 16, 25]

evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # Output: [2, 4]
```

---

## Data Structures

### Lists

```python
# Creating lists
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]

# Accessing elements
print(fruits[0])      # Output: apple (first element)
print(fruits[-1])     # Output: cherry (last element)

# Slicing
print(numbers[1:3])   # Output: [2, 3] (index 1 to 2)
print(numbers[:3])    # Output: [1, 2, 3] (first 3)
print(numbers[2:])    # Output: [3, 4, 5] (from index 2)

# Modifying lists
fruits.append("orange")      # Add to end
fruits.insert(1, "grape")    # Insert at index
fruits.remove("banana")     # Remove element
fruits.pop()                 # Remove last element

# List comprehension
squares = [x**2 for x in range(5)]
print(squares)  # Output: [0, 1, 4, 9, 16]

# With condition
evens = [x for x in range(10) if x % 2 == 0]
print(evens)  # Output: [0, 2, 4, 6, 8]
```

### Dictionaries

```python
# Creating dictionaries
student = {
    "name": "Alice",
    "age": 20,
    "grades": [85, 90, 88]
}

# Accessing values
print(student["name"])           # Output: Alice
print(student.get("age"))        # Output: 20
print(student.get("city", "N/A"))  # Output: N/A (default if key doesn't exist)

# Modifying dictionaries
student["city"] = "New York"     # Add/update
student["age"] = 21              # Update
del student["grades"]            # Delete key

# Iterating
for key, value in student.items():
    print(f"{key}: {value}")

# Dictionary comprehension
squares_dict = {x: x**2 for x in range(5)}
print(squares_dict)  # Output: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### Tuples

```python
# Tuples are immutable (cannot be changed)
coordinates = (10, 20)
point = (x, y) = (5, 3)

# Accessing
print(coordinates[0])  # Output: 10

# Unpacking
x, y = coordinates
print(x, y)  # Output: 10 20

# Use cases: when you need immutable data
colors = ("red", "green", "blue")
```

### Sets

```python
# Sets store unique elements
unique_numbers = {1, 2, 3, 3, 4}
print(unique_numbers)  # Output: {1, 2, 3, 4}

# Set operations
set1 = {1, 2, 3}
set2 = {3, 4, 5}

print(set1.union(set2))        # Output: {1, 2, 3, 4, 5}
print(set1.intersection(set2)) # Output: {3}
print(set1.difference(set2))   # Output: {1, 2}
```

---

## File I/O

### Reading Files

```python
# Read entire file
with open("data.txt", "r") as file:
    content = file.read()
    print(content)

# Read line by line
with open("data.txt", "r") as file:
    for line in file:
        print(line.strip())  # strip() removes newline

# Read all lines into list
with open("data.txt", "r") as file:
    lines = file.readlines()
```

### Writing Files

```python
# Write to file
with open("output.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("This is a new line")

# Append to file
with open("output.txt", "a") as file:
    file.write("\nAppended text")
```

### CSV Files

```python
import csv

# Reading CSV
with open("data.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# Writing CSV
with open("output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Age", "City"])
    writer.writerow(["Alice", 25, "New York"])
```

---

## Error Handling

### Try/Except

```python
# Basic error handling
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Multiple exceptions
try:
    value = int("not a number")
    result = 10 / value
except ValueError:
    print("Invalid number format")
except ZeroDivisionError:
    print("Cannot divide by zero")
except Exception as e:
    print(f"An error occurred: {e}")

# Finally block (always executes)
try:
    file = open("data.txt", "r")
    content = file.read()
except FileNotFoundError:
    print("File not found")
finally:
    file.close()  # Always close file
```

### Raising Exceptions

```python
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

try:
    result = divide(10, 0)
except ValueError as e:
    print(e)  # Output: Cannot divide by zero
```

---

## Object-Oriented Programming

### Classes and Objects

```python
class Dog:
    # Class attribute
    species = "Canis familiaris"
    
    # Constructor
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age
    
    # Method
    def bark(self):
        return f"{self.name} says Woof!"
    
    def get_info(self):
        return f"{self.name} is {self.age} years old"

# Creating objects
dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)

print(dog1.bark())      # Output: Buddy says Woof!
print(dog2.get_info())  # Output: Max is 5 years old
```

### Inheritance

```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return f"{self.name} makes a sound"

class Cat(Animal):
    def speak(self):  # Override parent method
        return f"{self.name} says Meow"

class Dog(Animal):
    def speak(self):  # Override parent method
        return f"{self.name} says Woof"

cat = Cat("Whiskers")
dog = Dog("Buddy")

print(cat.speak())  # Output: Whiskers says Meow
print(dog.speak())  # Output: Buddy says Woof
```

---

## Practice Exercises

### Exercise 1: Basic Operations

**Task:** Write a function that takes two numbers and returns their sum, difference, product, and quotient.

**Solution:**
```python
def calculate(a, b):
    return {
        "sum": a + b,
        "difference": a - b,
        "product": a * b,
        "quotient": a / b if b != 0 else "Cannot divide by zero"
    }

result = calculate(10, 5)
print(result)
# Output: {'sum': 15, 'difference': 5, 'product': 50, 'quotient': 2.0}
```

### Exercise 2: List Manipulation

**Task:** Write a function that takes a list of numbers and returns a new list with only even numbers squared.

**Solution:**
```python
def even_squares(numbers):
    return [x**2 for x in numbers if x % 2 == 0]

result = even_squares([1, 2, 3, 4, 5, 6])
print(result)  # Output: [4, 16, 36]
```

### Exercise 3: Dictionary Operations

**Task:** Create a function that counts word frequency in a sentence.

**Solution:**
```python
def word_count(sentence):
    words = sentence.lower().split()
    frequency = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1
    return frequency

result = word_count("the quick brown fox jumps over the lazy dog")
print(result)
# Output: {'the': 2, 'quick': 1, 'brown': 1, 'fox': 1, 'jumps': 1, 'over': 1, 'lazy': 1, 'dog': 1}
```

### Exercise 4: File Processing

**Task:** Write a program that reads a file and counts the number of lines and words.

**Solution:**
```python
def count_file_stats(filename):
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
            line_count = len(lines)
            word_count = sum(len(line.split()) for line in lines)
            return {"lines": line_count, "words": word_count}
    except FileNotFoundError:
        return "File not found"

stats = count_file_stats("data.txt")
print(stats)
```

### Exercise 5: Class Implementation

**Task:** Create a `BankAccount` class with deposit, withdraw, and balance methods.

**Solution:**
```python
class BankAccount:
    def __init__(self, account_number, initial_balance=0):
        self.account_number = account_number
        self.balance = initial_balance
    
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            return f"Deposited ${amount}. New balance: ${self.balance}"
        return "Invalid deposit amount"
    
    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            return f"Withdrew ${amount}. New balance: ${self.balance}"
        return "Insufficient funds or invalid amount"
    
    def get_balance(self):
        return f"Account {self.account_number} balance: ${self.balance}"

account = BankAccount("12345", 1000)
print(account.deposit(500))   # Output: Deposited $500. New balance: $1500
print(account.withdraw(200))  # Output: Withdrew $200. New balance: $1300
print(account.get_balance())  # Output: Account 12345 balance: $1300
```

---

## Key Takeaways

1. **Python is dynamically typed** - no need to declare variable types
2. **Indentation matters** - Python uses indentation for code blocks
3. **Lists are versatile** - most commonly used data structure
4. **Dictionaries are powerful** - key-value pairs for structured data
5. **Functions are first-class** - can be passed as arguments
6. **Error handling is important** - use try/except for robust code
7. **OOP helps organize code** - classes and objects for complex programs

---

## Next Steps

- Practice writing Python programs daily
- Work through the exercises above
- Try solving problems on [HackerRank](https://www.hackerrank.com/) or [LeetCode](https://leetcode.com/)
- Move to [02-mathematics-basics.md](02-mathematics-basics.md) when comfortable

**Remember**: Practice is key! Code along with examples and experiment with variations.

