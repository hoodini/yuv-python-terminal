// ==================== Line Numbers ====================
function updateLineNumbers() {
    const codeEditor = document.getElementById('codeEditor');
    const lineNumbers = document.getElementById('lineNumbers');
    const lines = codeEditor.value.split('\n').length;

    let lineNumbersHtml = '';
    for (let i = 1; i <= lines; i++) {
        lineNumbersHtml += i + '\n';
    }
    lineNumbers.textContent = lineNumbersHtml;
}

// ==================== Code Templates ====================
const codeTemplates = {
    hello: `# Hello World - YUV.PYTHON
print("🐍 Welcome to YUV.PYTHON Terminal!")
print("Created by Yuval Avidani - GitHub Star")
print("="*50)

# Greetings
name = "Developer"
print(f"\\nHello, {name}! 👋")
print("Ready to code some Python in the browser?")

# Try some math
print(f"\\n✨ Math Operations:")
print(f"2 + 2 = {2 + 2}")
print(f"10 * 5 = {10 * 5}")
print(f"2 ** 8 = {2 ** 8}")

# Try some strings
print(f"\\n🎨 String Operations:")
message = "Python in WebAssembly!"
print(f"Message: {message}")
print(f"Uppercase: {message.upper()}")
print(f"Length: {len(message)} characters")

print(f"\\n🚀 You're running Python {__import__('sys').version.split()[0]} in your browser!")`,

    fibonacci: `# Fibonacci Sequence Generator
def fibonacci(n):
    """Generate Fibonacci sequence up to n terms"""
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result

# Generate first 15 Fibonacci numbers
fib_numbers = fibonacci(15)
print("Fibonacci Sequence:")
print(fib_numbers)

# Using generator for memory efficiency
def fib_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

print("\\nUsing Generator:")
for num in fib_generator(10):
    print(num, end=' ')`,

    sorting: `# Sorting Algorithms Demo
import random

# Generate random list
numbers = [random.randint(1, 100) for _ in range(10)]
print("Original:", numbers)

# Bubble Sort
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# Quick Sort
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

print("\\nBubble Sort:", bubble_sort(numbers.copy()))
print("Quick Sort:", quick_sort(numbers.copy()))
print("Built-in Sort:", sorted(numbers))`,

    datastructures: `# Data Structures in Python
from collections import deque, Counter

# Stack using list
stack = []
stack.append(1)
stack.append(2)
stack.append(3)
print("Stack:", stack)
print("Pop:", stack.pop())

# Queue using deque
queue = deque([1, 2, 3])
queue.append(4)
print("\\nQueue:", queue)
print("Dequeue:", queue.popleft())

# Dictionary
person = {
    "name": "Yuval",
    "role": "AI Builder",
    "company": "YUV.AI"
}
print("\\nPerson:", person)

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
print("\\nUnion:", set1 | set2)
print("Intersection:", set1 & set2)

# Counter
words = ["python", "ai", "python", "ai", "code"]
print("\\nWord Count:", Counter(words))`,

    decorators: `# Python Decorators
import time

def timer_decorator(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️ {func.__name__} took {end-start:.4f} seconds")
        return result
    return wrapper

def cache_decorator(func):
    """Simple memoization decorator"""
    cache = {}
    def wrapper(*args):
        if args in cache:
            print(f"📦 Cache hit for {args}")
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

@timer_decorator
@cache_decorator
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test it
print("Computing fibonacci(10)...")
print(f"Result: {fibonacci(10)}")
print("\\nComputing again (cached)...")
print(f"Result: {fibonacci(10)}")`,

    comprehensions: `# List, Dict, and Set Comprehensions

# List Comprehension
squares = [x**2 for x in range(10)]
print("Squares:", squares)

# With condition
evens = [x for x in range(20) if x % 2 == 0]
print("\\nEven numbers:", evens)

# Nested comprehension
matrix = [[i*j for j in range(5)] for i in range(5)]
print("\\nMultiplication Table:")
for row in matrix:
    print(row)

# Dictionary Comprehension
word_lengths = {word: len(word) for word in ["python", "code", "ai"]}
print("\\nWord lengths:", word_lengths)

# Set Comprehension
unique_lengths = {len(word) for word in ["hello", "world", "hi", "code"]}
print("Unique lengths:", unique_lengths)

# Generator Expression (memory efficient)
gen = (x**2 for x in range(1000000))
print("\\nFirst 5 from generator:", [next(gen) for _ in range(5)])`,

    classes: `# Object-Oriented Programming in Python

class Person:
    """A class representing a person"""

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"Hi, I'm {self.name}, {self.age} years old"

    def __repr__(self):
        return f"Person(name='{self.name}', age={self.age})"

class Developer(Person):
    """Developer class inheriting from Person"""

    def __init__(self, name, age, language):
        super().__init__(name, age)
        self.language = language

    def introduce(self):
        return f"{super().introduce()} and I code in {self.language}"

    def code(self):
        return f"💻 Coding in {self.language}..."

# Create instances
person = Person("Alice", 25)
dev = Developer("Yuval", 30, "Python")

print(person.introduce())
print(dev.introduce())
print(dev.code())
print(f"\\nDeveloper object: {dev}")`,

    generators: `# Python Generators - Memory Efficient Iteration

def simple_generator():
    """A simple generator function"""
    yield 1
    yield 2
    yield 3

print("Simple generator:")
for value in simple_generator():
    print(value)

def infinite_sequence():
    """Generate infinite sequence"""
    num = 0
    while True:
        yield num
        num += 1

print("\\nFirst 10 from infinite sequence:")
gen = infinite_sequence()
for _ in range(10):
    print(next(gen), end=' ')

def fibonacci_gen():
    """Fibonacci generator"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

print("\\n\\nFibonacci with generator:")
fib = fibonacci_gen()
print([next(fib) for _ in range(10)])

# Generator Expression
squares = (x**2 for x in range(10))
print("\\nSquares generator:", list(squares))

# File reading generator (efficient for large files)
def read_large_file(filename):
    """Memory-efficient file reader"""
    for line in open(filename):
        yield line.strip()

print("\\n✨ Generators are memory efficient!")
print("They generate values on-the-fly instead of storing all in memory")`
};

function loadTemplate(templateName) {
    const template = codeTemplates[templateName];
    if (template) {
        elements.codeEditor.value = template;
        updateLineNumbers();
        elements.codeEditor.scrollTop = 0;
        
        // Update active state
        document.querySelectorAll('.nav-btn').forEach(btn => {
            if (btn.getAttribute('data-template') === templateName) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }
}

// ==================== Python Integration ====================
let pyodide = null;
let isInitialized = false;

const elements = {
    runBtn: document.getElementById('runBtn'),
    clearBtn: document.getElementById('clearBtn'),
    codeEditor: document.getElementById('codeEditor'),
    output: document.getElementById('output'),
    connectionStatus: document.getElementById('connection-status'),
    pythonVersion: document.getElementById('python-version')
};

// Update connection status
function updateConnectionStatus(message) {
    elements.connectionStatus.textContent = message;
    
    // Update dot color based on status
    const dot = document.getElementById('connection-dot');
    if (message === 'SYSTEM READY') {
        dot.style.background = 'var(--accent-success)';
        dot.style.boxShadow = '0 0 10px var(--accent-success)';
    } else if (message === 'INITIALIZING') {
        dot.style.background = 'var(--accent-warning)';
        dot.style.boxShadow = '0 0 10px var(--accent-warning)';
    } else {
        dot.style.background = 'var(--accent-error)';
        dot.style.boxShadow = '0 0 10px var(--accent-error)';
    }
}

// Initialize Pyodide
async function initPyodide() {
    if (isInitialized) return;

    try {
        updateConnectionStatus('INITIALIZING');
        elements.runBtn.disabled = true;
        elements.runBtn.style.opacity = '0.5';

        clearOutput();
        appendOutput('$ ', 'terminal-prompt');
        appendOutput('Initializing Python WebAssembly runtime...\n', 'terminal-text');

        pyodide = await loadPyodide({
            indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/'
        });

        // Redirect Python stdout and stderr to our output
        await pyodide.runPythonAsync(`
import sys
import io

class OutputCatcher(io.StringIO):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def write(self, text):
        if text and text.strip():
            self.callback(text)
        return len(text)

output_buffer = []

def capture_output(text):
    output_buffer.append(text)

sys.stdout = OutputCatcher(capture_output)
sys.stderr = OutputCatcher(capture_output)
        `);

        // Get Python version
        const version = await pyodide.runPythonAsync('sys.version.split()[0]');
        elements.pythonVersion.textContent = `Python ${version}`;

        isInitialized = true;
        updateConnectionStatus('SYSTEM READY');
        elements.runBtn.disabled = false;
        elements.runBtn.style.opacity = '1';

        appendOutput('$ ', 'terminal-prompt');
        appendOutput('Runtime initialized successfully\n', 'success-text');
        appendOutput('$ ', 'terminal-prompt');
        appendOutput('Ready to execute Python code\n\n', 'success-text');

    } catch (error) {
        updateConnectionStatus('SYSTEM ERROR');
        appendOutput('$ ', 'terminal-prompt');
        appendOutput(`Failed to initialize: ${error.message}\n`, 'error-text');
        console.error('Pyodide initialization error:', error);
    }
}

// Append output to the output panel
function appendOutput(text, className = '') {
    const outputElement = elements.output;

    if (className) {
        const span = document.createElement('span');
        span.className = className;
        span.textContent = text;
        outputElement.appendChild(span);
    } else {
        const textNode = document.createTextNode(text);
        outputElement.appendChild(textNode);
    }

    // Auto-scroll to bottom with smooth behavior
    const terminalContent = document.querySelector('.terminal-content');
    terminalContent.scrollTop = terminalContent.scrollHeight;
}

// Clear output
function clearOutput() {
    elements.output.innerHTML = '';
}

// Run Python code with animation
async function runPythonCode() {
    if (!isInitialized) {
        await initPyodide();
        if (!isInitialized) return;
    }

    const code = elements.codeEditor.value.trim();

    if (!code) {
        clearOutput();
        appendOutput('$ ', 'terminal-prompt');
        appendOutput('Error: No code to execute\n', 'error-text');
        return;
    }

    try {
        elements.runBtn.disabled = true;
        elements.runBtn.innerHTML = '<span class="icon">⏳</span> RUNNING...';

        clearOutput();
        appendOutput('$ ', 'terminal-prompt');
        appendOutput('Executing...\n\n', 'success-text');

        // Clear the output buffer
        await pyodide.runPythonAsync('output_buffer.clear()');

        // Run the user's code
        await pyodide.runPythonAsync(code);

        // Get captured output
        const capturedOutput = await pyodide.runPythonAsync(`
''.join(output_buffer)
        `);

        if (capturedOutput) {
            appendOutput(capturedOutput);
            appendOutput('\n');
        } else {
            appendOutput('$ ', 'terminal-prompt');
            appendOutput('Execution complete (no output)\n', 'success-text');
        }

        appendOutput('\n$ ', 'terminal-prompt');
        appendOutput('Done\n', 'success-text');

    } catch (error) {
        appendOutput('\n$ ', 'terminal-prompt');
        appendOutput('Traceback:\n', 'error-text');
        appendOutput(error.message + '\n', 'error-text');
        console.error('Python execution error:', error);

    } finally {
        elements.runBtn.disabled = false;
        elements.runBtn.innerHTML = '<span class="icon">▶</span> RUN';
    }
}

// ==================== Event Listeners ====================

// Template buttons
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const template = btn.getAttribute('data-template');
        loadTemplate(template);
        clearOutput();
        appendOutput('$ ', 'terminal-prompt');
        appendOutput(`Loaded template: ${btn.textContent.trim()}\n`, 'success-text');
    });
});

// Run button
elements.runBtn.addEventListener('click', runPythonCode);

// Clear button
elements.clearBtn.addEventListener('click', () => {
    elements.codeEditor.value = '';
    updateLineNumbers();
    clearOutput();
    appendOutput('$ ', 'terminal-prompt');
    appendOutput('Editor cleared\n', 'terminal-text');
});

// Code editor event listeners
elements.codeEditor.addEventListener('input', updateLineNumbers);

elements.codeEditor.addEventListener('scroll', () => {
    const lineNumbers = document.getElementById('lineNumbers');
    lineNumbers.scrollTop = elements.codeEditor.scrollTop;
});

// Keyboard shortcuts
elements.codeEditor.addEventListener('keydown', (e) => {
    // Ctrl+Enter or Shift+Enter to run
    if ((e.ctrlKey || e.shiftKey) && e.key === 'Enter') {
        e.preventDefault();
        runPythonCode();
    }

    // Tab support (4 spaces)
    if (e.key === 'Tab') {
        e.preventDefault();
        const start = e.target.selectionStart;
        const end = e.target.selectionEnd;
        const value = e.target.value;

        e.target.value = value.substring(0, start) + '    ' + value.substring(end);
        e.target.selectionStart = e.target.selectionEnd = start + 4;
        updateLineNumbers();
    }

    // Ctrl+/ to comment/uncomment
    if (e.ctrlKey && e.key === '/') {
        e.preventDefault();
        const start = e.target.selectionStart;
        const end = e.target.selectionEnd;
        const value = e.target.value;

        const lines = value.split('\n');
        const startLine = value.substring(0, start).split('\n').length - 1;
        const endLine = value.substring(0, end).split('\n').length - 1;

        const isAllCommented = lines.slice(startLine, endLine + 1).every(line => line.trim().startsWith('#'));

        for (let i = startLine; i <= endLine; i++) {
            if (isAllCommented) {
                lines[i] = lines[i].replace(/^\s*#\s?/, '');
            } else {
                lines[i] = '# ' + lines[i];
            }
        }

        e.target.value = lines.join('\n');
        updateLineNumbers();
    }
});

// ==================== Initialize on Load ====================
window.addEventListener('DOMContentLoaded', () => {
    // Initialize line numbers
    updateLineNumbers();

    // Set initial status
    updateConnectionStatus('INITIALIZING');

    // Load default template
    loadTemplate('hello');

    // Set initial output
    clearOutput();
    appendOutput('$ ', 'terminal-prompt');
    appendOutput('YUV.PYTHON Terminal v2.0\n', 'terminal-text');
    appendOutput('$ ', 'terminal-prompt');
    appendOutput('Initializing system...\n', 'terminal-text');
});

// Handle window resize
window.addEventListener('resize', updateLineNumbers);
