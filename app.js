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

const MICRO_TORCH_LIB = `
# MICRO-TORCH: A lightweight PyTorch shim for the browser (NumPy-based)
import numpy as np
import math

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.grad = None
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"tensor({self.data})"
        
    def backward(self):
        pass # Autograd not fully implemented in shim

    def item(self):
        return self.data.item()
        
    def toList(self):
        return self.data.tolist()

def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad)

def relu(x):
    return Tensor(np.maximum(0, x.data))

def sigmoid(x):
    return Tensor(1 / (1 + np.exp(-x.data)))

class Module:
    def __init__(self):
        self._parameters = []
    
    def parameters(self):
        return self._parameters

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Kaiming initialization approximation
        limit = math.sqrt(6 / (in_features + out_features))
        self.weight = Tensor(np.random.uniform(-limit, limit, (out_features, in_features)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        self._parameters.extend([self.weight, self.bias])
        
    def forward(self, input):
        # input: (batch, in), weight: (out, in) -> output: (batch, out)
        # Input might be a Tensor or numpy
        x = input.data if isinstance(input, Tensor) else np.array(input)
        
        # Simple matrix multiplication: x @ W.T + b
        res = x @ self.weight.data.T + self.bias.data
        return Tensor(res)

# Mock Optimizer
class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
        
    def step(self):
        # Mock update for demo visualization
        for p in self.params:
            if hasattr(p, 'data'):
                # In a real shim we'd need gradients. 
                # For this VISUAL DEMO, we mimic "learning" by nudging weights towards a pattern
                # or just letting the loss function in the user code drive the visuals.
                # Since we don't have real autograd here, we'll implement a 
                # "Gradient Descent Simulation" where we assume gradients exist or just decay loss.
                pass
                
    def zero_grad(self):
        pass

# Structure the 'torch' module
class torch_shim:
    def __init__(self):
        self.Tensor = Tensor
        self.tensor = tensor
        self.relu = relu
        self.sigmoid = sigmoid
        self.float32 = np.float32
        
    def manual_seed(self, seed):
        np.random.seed(seed)
        
    def randn(self, *size):
        return Tensor(np.random.randn(*size))

# Structure 'torch.nn'
class nn_shim:
    def __init__(self):
        self.Module = Module
        self.Linear = Linear
        self.ReLU = lambda: (lambda x: relu(x))
        self.Sigmoid = lambda: (lambda x: sigmoid(x))
        self.MSELoss = lambda: (lambda pred, target: Tensor(np.mean((pred.data - target.data)**2)))

# Structure 'torch.optim'
class optim_shim:
    def __init__(self):
        self.SGD = SGD

# Inject into sys.modules
import sys
from types import ModuleType

# Create torch module
m_torch = ModuleType('torch')
torch_inst = torch_shim()
for attr in dir(torch_inst):
    if not attr.startswith('__'):
        setattr(m_torch, attr, getattr(torch_inst, attr))

# Create torch.nn
m_nn = ModuleType('torch.nn')
nn_inst = nn_shim()
for attr in dir(nn_inst):
    if not attr.startswith('__'):
        setattr(m_nn, attr, getattr(nn_inst, attr))
m_torch.nn = m_nn

# Create torch.optim
m_optim = ModuleType('torch.optim')
optim_inst = optim_shim()
for attr in dir(optim_inst):
    if not attr.startswith('__'):
        setattr(m_optim, attr, getattr(optim_inst, attr))
m_torch.optim = m_optim

sys.modules['torch'] = m_torch
sys.modules['torch.nn'] = m_nn
sys.modules['torch.optim'] = m_optim

print("⚡ Micro-PyTorch (NumPy Shim) loaded for Browser Demo")
`;

// Helper to inject shim if needed
function getAugmentedCode(code) {
    if (code.includes('import torch')) {
        return MICRO_TORCH_LIB + '\n\n' + code;
    }
    return code;
}

const codeTemplates = {
    shim: MICRO_TORCH_LIB,

    hello: `# Hello World - YUV.PY
print("🐍 Welcome to YUV.PY Terminal!")
print("Created by Yuval Avidani - GitHub Star")
print("=" * 50)

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
    print(num, end = ' ')`,

    sorting: `# Sorting Algorithms Demo
import random

# Generate random list
numbers = [random.randint(1, 100) for _ in range(10)]
print("Original:", numbers)

# Bubble Sort
def bubble_sort(arr):
n = len(arr)
for i in range(n):
    for j in range(0, n - i - 1):
        if arr[j] > arr[j + 1]:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]
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
set1 = { 1, 2, 3, 4}
set2 = { 3, 4, 5, 6}
print("\\nUnion:", set1 | set2)
print("Intersection:", set1 & set2)

# Counter
words = ["python", "ai", "python", "ai", "code"]
print("\\nWord Count:", Counter(words))`,

    decorators: `# Python Decorators
import time

def timer_decorator(func):
"""Decorator to measure function execution time"""
    def wrapper(* args, ** kwargs):
start = time.time()
result = func(* args, ** kwargs)
end = time.time()
print(f"⏱️ {func.__name__} took {end-start:.4f} seconds")
return result
return wrapper

def cache_decorator(func):
"""Simple memoization decorator"""
cache = {}
    def wrapper(* args):
if args in cache:
    print(f"📦 Cache hit for {args}")
return cache[args]
result = func(* args)
cache[args] = result
return result
return wrapper

@timer_decorator
@cache_decorator
def fibonacci(n):
if n < 2:
    return n
return fibonacci(n - 1) + fibonacci(n - 2)

# Test it
print("Computing fibonacci(10)...")
print(f"Result: {fibonacci(10)}")
print("\\nComputing again (cached)...")
print(f"Result: {fibonacci(10)}")`,

    comprehensions: `# List, Dict, and Set Comprehensions

# List Comprehension
squares = [x ** 2 for x in range(10)]
print("Squares:", squares)

# With condition
evens = [x for x in range(20) if x % 2 == 0]
print("\\nEven numbers:", evens)

# Nested comprehension
matrix = [[i * j for j in range(5)]for i in range(5)]
print("\\nMultiplication Table:")
for row in matrix:
    print(row)

# Dictionary Comprehension
word_lengths = { word: len(word) for word in ["python", "code", "ai"] }
print("\\nWord lengths:", word_lengths)

# Set Comprehension
unique_lengths = { len(word) for word in ["hello", "world", "hi", "code"] }
print("Unique lengths:", unique_lengths)

# Generator Expression(memory efficient)
gen = (x ** 2 for x in range(1000000))
    print("\\nFirst 5 from generator:", [next(gen) for _ in range(5)])`,

    classes: `# Object - Oriented Programming in Python

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
    print(next(gen), end = ' ')

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
squares = (x ** 2 for x in range(10))
    print("\\nSquares generator:", list(squares))

# File reading generator(efficient for large files)
def read_large_file(filename):
"""Memory-efficient file reader"""
for line in open(filename):
    yield line.strip()

print("\\n✨ Generators are memory efficient!")
print("They generate values on-the-fly instead of storing all in memory")`,

    neuralnet: `# Neural Network Training Simulation
import random
import asyncio

class Neuron:
    def __init__(self):
self.weights = [random.uniform(-1, 1) for _ in range(3)]
self.bias = random.uniform(-1, 1)

    def activate(self, inputs):
        # Only for structure, value unused in this demo sim
return 0

print("🧠 Initializing Neural Network...")
print("Architecture: [Input(3) -> Hidden(4) -> Hidden(4) -> Output(2)]")

# We use an async main function to allow UI updates during sleep
async function main():
    print("\\n🔄 Starting Training Loop...")
epochs = 10

for i in range(epochs):
        # Simulate calculation time
await asyncio.sleep(0.5)
        
        # Calculate dummy metrics
loss = 0.5 * (0.8 ** i) + random.uniform(0.01, 0.05)
acc = 0.5 + (0.45 * (1 - (0.8 ** i))) + random.uniform(-0.02, 0.02)
        
        # Visualization of a progress bar
progress = "█" * (i + 1) + "░" * (epochs - i - 1)

print(f"Epoch {i+1}/{epochs} [{progress}]")
print(f"   Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

print("\\n✨ Training Complete!")
print("Model ready for inference.")

# Check if running in an event loop(Pyodide usually is)
# We can just await it at top level in newer Pyodide / Runners, 
# but safest is simply calling it if we are already in async context or just:
await main()`,
};

// ==================== Neural Network Builder & Viz ====================
const builderState = {
    inputSize: 3,
    outputSize: 2,
    hiddenLayers: [
        { size: 4, activation: 'relu' },
        { size: 4, activation: 'relu' }
    ],
    framework: 'pytorch', // pytorch, tensorflow, pure
    isOpen: false
};

const neuralViz = {
    canvas: null,
    ctx: null,
    isActive: false,
    particles: [],
    connections: [],
    animationId: null,

    init() {
        this.canvas = document.getElementById('neuralCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.resize();
        window.addEventListener('resize', () => this.resize());
    },

    resize() {
        if (!this.canvas) return;
        this.canvas.width = this.canvas.parentElement.offsetWidth;
        this.canvas.height = this.canvas.parentElement.offsetHeight;
        if (this.isActive) this.createNetwork();
    },

    start() {
        if (this.isActive) return;
        this.isActive = true;
        document.getElementById('neural-viz').classList.add('active');
        this.createNetwork();
        this.animate();
    },

    stop() {
        this.isActive = false;
        document.getElementById('neural-viz').classList.remove('active');
        cancelAnimationFrame(this.animationId);
    },

    createNetwork() {
        this.particles = [];
        this.connections = [];

        // Dynamic layers based on builder state
        const layers = [
            builderState.inputSize,
            ...builderState.hiddenLayers.map(l => l.size),
            builderState.outputSize
        ];

        const layerGap = this.canvas.width / (layers.length + 1);

        layers.forEach((nodeCount, layerIndex) => {
            const x = layerGap * (layerIndex + 1);
            // Center nodes vertically
            const totalHeight = this.canvas.height;
            const nodeGap = Math.min(60, totalHeight / (nodeCount + 1)); // Cap gap size
            const totalNodesHeight = nodeGap * (nodeCount - 1);
            const startY = (totalHeight - totalNodesHeight) / 2;

            for (let i = 0; i < nodeCount; i++) {
                this.particles.push({
                    x: x,
                    y: startY + (i * nodeGap),
                    r: layerIndex === 0 || layerIndex === layers.length - 1 ? 6 : 4,
                    layer: layerIndex,
                    active: false,
                    activation: 0 // For smooth glow
                });
            }
        });

        // Create connections
        for (let i = 0; i < this.particles.length; i++) {
            for (let j = 0; j < this.particles.length; j++) {
                if (this.particles[j].layer === this.particles[i].layer + 1) {
                    this.connections.push({
                        from: this.particles[i],
                        to: this.particles[j],
                        weight: Math.random(),
                        signal: 0
                    });
                }
            }
        }
    },

    animate() {
        if (!this.isActive) return;

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Randomly trigger inputs
        if (Math.random() < 0.1) {
            this.particles.filter(p => p.layer === 0).forEach(p => {
                if (Math.random() < 0.5) {
                    p.active = true;
                    p.activation = 1.0;
                }
            });
        }

        // Draw connections
        this.connections.forEach(conn => {
            this.ctx.beginPath();
            this.ctx.moveTo(conn.from.x, conn.from.y);
            this.ctx.lineTo(conn.to.x, conn.to.y);

            // Signal traveling
            if (conn.signal > 0) {
                this.ctx.strokeStyle = `rgba(0, 243, 255, ${conn.signal})`;
                this.ctx.lineWidth = 2;
                conn.signal -= 0.04; // Speed

                // When signal reaches destination (simulated by signal fade for now visuals)
                // Better visual: move a dot? For now, fade is okay, but let's trigger next layer
                if (conn.signal <= 0.1 && conn.to.layer < this.maxLayers) { // logic simplified
                    // handled by from.active logic below
                }
            } else {
                this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
                this.ctx.lineWidth = 1;
            }
            this.ctx.stroke();

            // Propagate signal logic
            if (conn.from.active && conn.signal <= 0) {
                conn.signal = 1;
                // Trigger destination after delay
                setTimeout(() => {
                    conn.to.active = true;
                    conn.to.activation = 1.0;
                }, 100);
            }
        });

        // Draw nodes
        this.particles.forEach(p => {
            if (p.active) {
                p.activation = Math.max(0, p.activation - 0.05);
                if (p.activation <= 0) p.active = false;
            }

            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);

            const alpha = 0.3 + (p.activation * 0.7);
            const color = p.layer === 0 ? '#00f3ff' : // Input (Cyan)
                p.layer === builderState.hiddenLayers.length + 1 ? '#bc13fe' : // Output (Purple)
                    '#ffffff'; // Hidden (White)

            this.ctx.fillStyle = p.active ? color : `rgba(255, 255, 255, 0.2)`;
            this.ctx.fill();

            if (p.active) {
                this.ctx.shadowBlur = 15;
                this.ctx.shadowColor = color;
            } else {
                this.ctx.shadowBlur = 0;
            }
        });
        this.ctx.shadowBlur = 0;

        this.animationId = requestAnimationFrame(() => this.animate());
    }
};

// ==================== Code Generation ====================
function generateCode() {
    const { inputSize, outputSize, hiddenLayers, framework } = builderState;
    let code = '';

    if (framework === 'pytorch') {
        code = `# PyTorch Neural Network(Browser Compatible)
import torch
import torch.nn as nn
import torch.optim as optim
import asyncio

# Define Network
class NeuralNetwork(nn.Module):
    def __init__(self):
super(NeuralNetwork, self).__init__()
self.layers = nn.Module() # Container
        # Architecture: ${inputSize} -> ${hiddenLayers.map(l => l.size).join(' -> ')} -> ${outputSize}
self.fc1 = nn.Linear(${inputSize}, ${hiddenLayers[0]?.size || 4})
        ${hiddenLayers.map((l, i) => i > 0 ? `self.fc${i + 1} = nn.Linear(${hiddenLayers[i - 1].size}, ${l.size})` : '').join('\n        ')}
self.out = nn.Linear(${hiddenLayers[hiddenLayers.length - 1]?.size || 4}, ${outputSize})

    def forward(self, x):

# Initialize Model
model = NeuralNet()
print(model)

# Dummy Data
inputs = torch.randn(10, ${inputSize})
targets = torch.randint(0, 2, (10, ${outputSize})).float()

# Training Loop
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

print("\\nTraining for 5 epochs:")
for epoch in range(5):
    optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
`;
    } else if (framework === 'tensorflow') {
        code = `# TensorFlow / Keras Neural Network
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Build Model
model = models.Sequential()
model.add(layers.Input(shape = (${inputSize},)))
`;
        hiddenLayers.forEach(layer => {
            code += `model.add(layers.Dense(${layer.size}, activation = '${layer.activation}')) \n`;
        });
        code += `model.add(layers.Dense(${outputSize}, activation = 'sigmoid'))

model.summary()

# Dummy Data
X = np.random.random((100, ${inputSize}))
y = np.random.randint(2, size = (100, ${outputSize}))

# Compile & Train
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print("\\nStarting Training...")
history = model.fit(X, y, epochs = 5, batch_size = 10, verbose = 1)
    `;
    } else {
        // Pure Python (Visual Demo support)
        code = `# Pure Python Neural Network(No Libraries)
import random
import math
import asyncio

def sigmoid(x):
return 1 / (1 + math.exp(-x))

class NeuralNetwork:
    def __init__(self):
        # Architecture: ${inputSize} -> ${hiddenLayers.map(l => l.size).join(' -> ')} -> ${outputSize}
self.weights = [] 
        #(Simplified logic for demo)
            
    async def train_simulation(self):
print("🧠 Initializing custom network...")
print("Structure: ${inputSize} inputs, ${outputSize} outputs")
print("\\n🚀 Starting simulated training...")

for i in range(5):
    await asyncio.sleep(0.6)
loss = 1.0 / (i + 1) + random.random() * 0.1
print(f"Step {i+1}: Loss decreasing... Current: {loss:.4f}")

print("\\n✅ Network optimized.")

# Run Simulation
nn = NeuralNetwork()
await nn.train_simulation()
    `;
    }

    return code;
}

function getActivationPyTorch(act) {
    if (!act) return 'ReLU()';
    const map = { 'relu': 'ReLU()', 'sigmoid': 'Sigmoid()', 'tanh': 'Tanh()' };
    return map[act] || 'ReLU()';
}

// ==================== Builder UI Logic ====================
function renderBuilderLayers() {
    const list = document.getElementById('layers-list');
    list.innerHTML = '';

    builderState.hiddenLayers.forEach((layer, index) => {
        const item = document.createElement('div');
        item.className = 'layer-item';
        item.innerHTML = `
    < span > Hidden ${index + 1}</span >
        <div style="display:flex; gap:5px;">
            <input type="number" value="${layer.size}" min="1" max="20" onchange="updateLayer(${index}, 'size', this.value)">
                <select onchange="updateLayer(${index}, 'activation', this.value)" style="background:rgba(255,255,255,0.1); color:white; border:none; border-radius:4px;">
                    <option value="relu" ${layer.activation === 'relu' ? 'selected' : ''}>ReLU</option>
                    <option value="sigmoid" ${layer.activation === 'sigmoid' ? 'selected' : ''}>Sigmoid</option>
                    <option value="tanh" ${layer.activation === 'tanh' ? 'selected' : ''}>Tanh</option>
                </select>
                <button onclick="removeLayer(${index})" style="background:none; border:none; color:#ff3860; cursor:pointer;">✕</button>
        </div>
`;
        list.appendChild(item);
    });
}

function updateLayer(index, field, value) {
    if (field === 'size') value = parseInt(value);
    builderState.hiddenLayers[index][field] = value;
    neuralViz.createNetwork(); // Live update viz
}

function removeLayer(index) {
    builderState.hiddenLayers.splice(index, 1);
    renderBuilderLayers();
    neuralViz.createNetwork();
}

function addLayer() {
    builderState.hiddenLayers.push({ size: 4, activation: 'relu' });
    renderBuilderLayers();
    neuralViz.createNetwork();
}

function toggleBuilder() {
    const panel = document.getElementById('builder-panel');
    builderState.isOpen = !builderState.isOpen;
    if (builderState.isOpen) {
        panel.classList.add('active');
        renderBuilderLayers();
    } else {
        panel.classList.remove('active');
    }
}

// Global exposure for inline events
window.updateLayer = updateLayer;
window.removeLayer = removeLayer;
window.toggleBuilder = toggleBuilder; // expose global for easy debugging if needed, though we use event listeners mostly for buttons we can reach


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

        const MICRO_TORCH_LIB = `
# MICRO - TORCH: A lightweight PyTorch shim for the browser(NumPy - based)
import numpy as np
import math

class Tensor:
    def __init__(self, data, requires_grad = False):
self.data = np.array(data, dtype = np.float32)
self.grad = None
self.requires_grad = requires_grad

    def __repr__(self):
return f"tensor({self.data})"
        
    def backward(self):
        pass # Autograd not fully implemented in shim

    def item(self):
return self.data.item()
        
    def toList(self):
return self.data.tolist()

def tensor(data, requires_grad = False):
return Tensor(data, requires_grad)

def relu(x):
return Tensor(np.maximum(0, x.data))

def sigmoid(x):
return Tensor(1 / (1 + np.exp(-x.data)))

class Module:
    def __init__(self):
self._parameters = []
    
    def parameters(self):
return self._parameters

    def __call__(self, * args, ** kwargs):
return self.forward(* args, ** kwargs)

class Linear(Module):
    def __init__(self, in_features, out_features):
super().__init__()
        # Kaiming initialization approximation
limit = math.sqrt(6 / (in_features + out_features))
self.weight = Tensor(np.random.uniform(-limit, limit, (out_features, in_features)), requires_grad = True)
self.bias = Tensor(np.zeros(out_features), requires_grad = True)
self._parameters.extend([self.weight, self.bias])
        
    def forward(self, input):
        # input: (batch, in), weight: (out, in) -> output: (batch, out)
        # Input might be a Tensor or numpy
x = input.data if isinstance(input, Tensor) else np.array(input)
        
        # Simple matrix multiplication: x @W.T + b
res = x @self.weight.data.T + self.bias.data
return Tensor(res)

# Mock Optimizer
class SGD:
    def __init__(self, params, lr = 0.01):
self.params = params
self.lr = lr
        
    def step(self):
        # Mock update for demo visualization
        for p in self.params:
        if hasattr(p, 'data'):
                # In a real shim we'd need gradients. 
                # For this VISUAL DEMO, we mimic "learning" by nudging weights towards a pattern
                # or just letting the loss function in the user code drive the visuals.
                # Since we don't have real autograd here, we'll implement a 
                # "Gradient Descent Simulation" where we assume gradients exist or just decay loss.
    pass
                
    def zero_grad(self):
pass

# Structure the 'torch' module
class torch_shim:
    def __init__(self):
self.Tensor = Tensor
self.tensor = tensor
self.relu = relu
self.sigmoid = sigmoid
self.float32 = np.float32
        
    def manual_seed(self, seed):
np.random.seed(seed)
        
    def randn(self, * size):
return Tensor(np.random.randn(* size))

# Structure 'torch.nn'
class nn_shim:
    def __init__(self):
self.Module = Module
self.Linear = Linear
self.ReLU = lambda: (lambda x: relu(x))
self.Sigmoid = lambda: (lambda x: sigmoid(x))
self.MSELoss = lambda: (lambda pred, target: Tensor(np.mean((pred.data - target.data) ** 2)))

# Structure 'torch.optim'
class optim_shim:
    def __init__(self):
self.SGD = SGD

# Inject into sys.modules
import sys
    from types import ModuleType

# Create torch module
m_torch = ModuleType('torch')
torch_inst = torch_shim()
for attr in dir(torch_inst):
    if not attr.startswith('__'):
setattr(m_torch, attr, getattr(torch_inst, attr))

# Create torch.nn
m_nn = ModuleType('torch.nn')
nn_inst = nn_shim()
for attr in dir(nn_inst):
    if not attr.startswith('__'):
setattr(m_nn, attr, getattr(nn_inst, attr))
m_torch.nn = m_nn

# Create torch.optim
m_optim = ModuleType('torch.optim')
optim_inst = optim_shim()
for attr in dir(optim_inst):
    if not attr.startswith('__'):
setattr(m_optim, attr, getattr(optim_inst, attr))
m_torch.optim = m_optim

sys.modules['torch'] = m_torch
sys.modules['torch.nn'] = m_nn
sys.modules['torch.optim'] = m_optim

print("⚡ Micro-PyTorch (NumPy Shim) loaded for Browser Demo")
    `;

        // Helper to inject shim if needed
        function getAugmentedCode(code) {
            if (code.includes('import torch')) {
                return MICRO_TORCH_LIB + '\n\n' + code;
            }
            return code;
        }

        // ... (in toggleBuilder)
        // Trigger visualization if neural net
        if (templateName === 'neuralnet') {
            document.getElementById('neural-viz').style.display = 'block'; // Ensure visible
            // Force resize next frame to ensure layout is collected
            requestAnimationFrame(() => {
                neuralViz.resize();
                neuralViz.start();
            });

            // Auto open builder for visibility
            if (!builderState.isOpen) {
                toggleBuilder();
            }
        } else {
            neuralViz.stop();
            document.getElementById('neural-viz').style.display = 'none'; // Hide cleanly
            if (builderState.isOpen) {
                toggleBuilder(); // Close if switching away
            }
        }
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

        // Define a JS function for Python to call
        // We attach it to the window (global) or direct exposure if possible, 
        // but pyodide can access JS scope. 
        // Actually, we can register it.
        self.py_std_out = (text) => {
            appendOutput(text);
        };

        // Redirect Python stdout and stderr to our JS function
        await pyodide.runPythonAsync(`
import sys
import io
import js

class OutputCatcher(io.StringIO):
    def write(self, text):
if text:
    js.py_std_out(text)
return len(text)

sys.stdout = OutputCatcher()
sys.stderr = OutputCatcher()
    `);

        // Get Python version
        const version = await pyodide.runPythonAsync('sys.version.split()[0]');
        elements.pythonVersion.textContent = `Python ${version} `;

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
        appendOutput(`Failed to initialize: ${error.message} \n`, 'error-text');
        console.error('Pyodide initialization error:', error);
    }
}

// Append output to the output panel
function appendOutput(text, className = '') {
    const outputElement = elements.output;

    // Handle ANSI escape codes if any (basic filtering) or newlines
    // For now, simple text appending

    if (className) {
        const span = document.createElement('span');
        span.className = className;
        span.textContent = text;
        outputElement.appendChild(span);
    } else {
        const textNode = document.createTextNode(text);
        outputElement.appendChild(textNode);
    }

    // Auto-scroll to bottom
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

        // We wrap user code in an async execution if it contains async keywords, 
        // or just standard run. 
        // To support top-level await if user writes async code:
        // Inject Shim if needed
        const finalCode = getAugmentedCode(code);
        await pyodide.runPythonAsync(finalCode);

        // Output is handled by streaming now, so we don't need to fetch a buffer.

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
        appendOutput(`Loaded template: ${btn.textContent.trim()} \n`, 'success-text');
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
    // Initialize neural viz
    neuralViz.init();

    // Initialize line numbers
    updateLineNumbers();

    // Builder UI Events
    const openBuilderBtn = document.getElementById('open-builder-btn');
    const closeBuilderBtn = document.getElementById('close-builder');
    const addLayerBtn = document.getElementById('add-layer-btn');
    const generateCodeBtn = document.getElementById('generate-code-btn');
    const inputSizeInput = document.getElementById('input-size');
    const outputSizeInput = document.getElementById('output-size');

    if (openBuilderBtn) openBuilderBtn.addEventListener('click', toggleBuilder);
    if (closeBuilderBtn) closeBuilderBtn.addEventListener('click', toggleBuilder);
    if (addLayerBtn) addLayerBtn.addEventListener('click', addLayer);

    if (inputSizeInput) inputSizeInput.addEventListener('change', (e) => {
        builderState.inputSize = parseInt(e.target.value);
        neuralViz.createNetwork();
    });

    if (outputSizeInput) outputSizeInput.addEventListener('change', (e) => {
        builderState.outputSize = parseInt(e.target.value);
        neuralViz.createNetwork();
    });

    // Framework selection
    document.querySelectorAll('input[name="framework"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            builderState.framework = e.target.value;
        });
    });

    if (generateCodeBtn) {
        generateCodeBtn.addEventListener('click', () => {
            const code = generateCode();
            elements.codeEditor.value = code;
            updateLineNumbers();

            // Auto run if simple python
            if (builderState.framework === 'pure') {
                runPythonCode();
                if (window.innerWidth <= 768) {
                    toggleBuilder(); // Close on mobile after run
                }
            } else {
                // Just show code
                clearOutput();
                appendOutput('$ ', 'terminal-prompt');
                appendOutput(`Generated ${builderState.framework} code.\n`, 'success-text');
                appendOutput('Note: PyTorch/TensorFlow libraries are not yet fully supported in this WASM runtime.\n', 'text-secondary');
                appendOutput('You can copy this code to your local machine, or switch to "Pure Python" to run a simulation here.\n', 'text-secondary');
                // Close builder to see code
                toggleBuilder();
            }
        });
    }

    // Mobile Menu Toggle
    const mobileToggle = document.getElementById('mobile-menu-toggle');
    const sidebar = document.querySelector('.sidebar');

    if (mobileToggle) {
        mobileToggle.addEventListener('click', () => {
            sidebar.classList.toggle('active');
        });
    }

    // Close mobile menu when a template is selected
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            // Simply check if we are in mobile mode by checking computed style or width
            if (window.innerWidth <= 768) {
                sidebar.classList.remove('active');
            }
        });
    });

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
