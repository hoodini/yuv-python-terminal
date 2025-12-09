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
    // Ctrl+Enter to run
    if (e.ctrlKey && e.key === 'Enter') {
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

    // Set initial output
    clearOutput();
    appendOutput('$ ', 'terminal-prompt');
    appendOutput('PyWASM Terminal v1.0\n', 'terminal-text');
    appendOutput('$ ', 'terminal-prompt');
    appendOutput('Click EXECUTE to initialize Python runtime\n', 'terminal-text');
    appendOutput('$ ', 'terminal-prompt');
    appendOutput('Press Ctrl+Enter to run code\n\n', 'terminal-text');

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
