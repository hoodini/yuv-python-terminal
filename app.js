// ==================== Particle Background ====================
function initParticles() {
    const canvas = document.getElementById('particles');
    const ctx = canvas.getContext('2d');

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles = [];
    const particleCount = 80;

    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.vx = (Math.random() - 0.5) * 0.5;
            this.vy = (Math.random() - 0.5) * 0.5;
            this.radius = Math.random() * 2 + 1;
            this.opacity = Math.random() * 0.5 + 0.2;
        }

        update() {
            this.x += this.vx;
            this.y += this.vy;

            if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
            if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
        }

        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 255, 249, ${this.opacity})`;
            ctx.fill();
        }
    }

    for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle());
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        particles.forEach(particle => {
            particle.update();
            particle.draw();
        });

        // Draw connections
        particles.forEach((p1, i) => {
            particles.slice(i + 1).forEach(p2 => {
                const dx = p1.x - p2.x;
                const dy = p1.y - p2.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < 150) {
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(0, 255, 249, ${0.15 * (1 - distance / 150)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            });
        });

        requestAnimationFrame(animate);
    }

    animate();

    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
}

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
print("Created by Yuval Avidani")

name = input("What's your name? ")
print(f"Hello, {name}! 👋")

# Try some math
print(f"\\n2 + 2 = {2 + 2}")`,

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
    status: document.getElementById('status'),
    connectionStatus: document.getElementById('connection-status'),
    pythonVersion: document.getElementById('python-version')
};

// Update status indicator with data attribute for CSS styling
function updateStatus(message, status) {
    elements.status.textContent = message;
    elements.status.setAttribute('data-status', status);
}

// Update connection status
function updateConnectionStatus(message) {
    elements.connectionStatus.textContent = message;
}

// Initialize Pyodide
async function initPyodide() {
    if (isInitialized) return;

    try {
        updateStatus('LOADING', 'loading');
        updateConnectionStatus('LOADING RUNTIME');
        elements.runBtn.disabled = true;

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
        updateStatus('READY', 'ready');
        updateConnectionStatus('ONLINE');
        elements.runBtn.disabled = false;

        appendOutput('$ ', 'terminal-prompt');
        appendOutput('Runtime initialized successfully\n', 'success-text');
        appendOutput('$ ', 'terminal-prompt');
        appendOutput('Ready to execute Python code\n\n', 'success-text');

    } catch (error) {
        updateStatus('ERROR', 'error');
        updateConnectionStatus('OFFLINE');
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
    outputElement.scrollTop = outputElement.scrollHeight;
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
        updateStatus('RUNNING', 'running');
        elements.runBtn.disabled = true;

        // Add button animation
        elements.runBtn.style.transform = 'scale(0.95)';
        setTimeout(() => {
            elements.runBtn.style.transform = '';
        }, 100);

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

        updateStatus('READY', 'ready');

    } catch (error) {
        updateStatus('ERROR', 'error');
        appendOutput('\n$ ', 'terminal-prompt');
        appendOutput('Traceback:\n', 'error-text');
        appendOutput(error.message + '\n', 'error-text');
        console.error('Python execution error:', error);

    } finally {
        elements.runBtn.disabled = false;
    }
}

// ==================== Event Listeners ====================

// Template buttons
document.querySelectorAll('.template-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const template = btn.getAttribute('data-template');
        loadTemplate(template);
        clearOutput();
        appendOutput('$ ', 'terminal-prompt');
        appendOutput(`Loaded template: ${btn.querySelector('.template-name').textContent}\n`, 'success-text');
        appendOutput('$ ', 'terminal-prompt');
        appendOutput('Click EXECUTE to run the code\n', 'terminal-text');
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

// ==================== Mobile Toggle ====================
const mobileToggle = document.getElementById('mobileToggle');
const editorPanel = document.getElementById('editorPanel');
const terminalPanel = document.getElementById('terminalPanel');
let showingTerminal = false;

if (mobileToggle) {
    mobileToggle.addEventListener('click', () => {
        showingTerminal = !showingTerminal;

        if (showingTerminal) {
            editorPanel.classList.add('minimized');
            terminalPanel.classList.add('active');
        } else {
            editorPanel.classList.remove('minimized');
            terminalPanel.classList.remove('active');
        }
    });
}

// ==================== Initialize on Load ====================
window.addEventListener('DOMContentLoaded', () => {
    // Initialize particles
    initParticles();

    // Initialize line numbers
    updateLineNumbers();

    // Set initial status
    updateStatus('READY', 'ready');
    updateConnectionStatus('INITIALIZING');

    // Load default template
    loadTemplate('hello');

    // Set initial output
    clearOutput();
    appendOutput('$ ', 'terminal-prompt');
    appendOutput('YUV.PYTHON Terminal v1.0\n', 'terminal-text');
    appendOutput('$ ', 'terminal-prompt');
    appendOutput('Created by Yuval Avidani - GitHub Star\n', 'success-text');
    appendOutput('$ ', 'terminal-prompt');
    appendOutput('Click EXECUTE to initialize Python runtime\n', 'terminal-text');
    appendOutput('$ ', 'terminal-prompt');
    appendOutput('Press Ctrl+Enter to run code\n', 'terminal-text');
    appendOutput('$ ', 'terminal-prompt');
    appendOutput('Try the code templates above!\n\n', 'terminal-text');

    // Update glitch effect data attribute
    const glitchElement = document.querySelector('.glitch');
    if (glitchElement) {
        glitchElement.setAttribute('data-text', 'YUV.PYTHON');
    }

    // Add typing effect to subtitle
    const typingText = document.querySelector('.typing-text');
    if (typingText) {
        const text = typingText.textContent;
        typingText.textContent = '';
        let i = 0;

        function typeWriter() {
            if (i < text.length) {
                typingText.textContent += text.charAt(i);
                i++;
                setTimeout(typeWriter, 100);
            }
        }

        setTimeout(typeWriter, 500);
    }
});

// Handle window resize
window.addEventListener('resize', updateLineNumbers);
