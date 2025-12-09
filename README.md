# YUV.PY // Neural Nexus

![YUV.PY Banner](screenshots/banner.png)

> **✨ [Live Demo](https://hoodini.github.io/yuv-python-terminal/)** - Try it in your browser!

Welcome to **YUV.PY**! This is a futuristic, web-based Python Playground that lets you write and run Python code directly in your browser. No complex installations, no heavy backend—just you and the code, powered by WebAssembly.

## 🌟 What is this?
YUV.PY is a "Terminal of the Future". It combines a beautiful, sci-fi interface with the power of Python.
- **Run Python Instantly**: Type code and press Run. It happens right in your browser.
- **Learn & Experiment**: Use our built-in templates to learn sorting, data structures, and more.
- **Build AI Visually**: Use the **Neural Network Builder** to design AI models without writing boilerplate code.

## 🚀 Quick Start (How to Run)

### Method 1: The Easiest Way (VS Code)
If you are using **Visual Studio Code**:
1. Install the "Live Server" extension.
2. Right-click on `index.html` and choose **"Open with Live Server"**.
3. That's it! The app will open in your browser.

### Method 2: The Developer Way (Node.js)
If you prefer using the terminal (command line):

1. **Install Node.js**: Make sure you have [Node.js](https://nodejs.org/) installed.
2. **Open Terminal**: Open your command prompt/terminal in this project folder.
3. **Start the Server**:
   Run this command to start a local web server:
   ```bash
   npx serve .
   ```
4. **Open in Browser**: The terminal will show a link (usually `http://localhost:3000`). Click it to open the app!

> **Note**: Opening `index.html` directly (double-clicking it) might not work in some browsers due to security restrictions on WebAssembly. Please use a local server as described above.

## 🎮 How to Use

### 1. The Terminal
The main screen is your Python playground.
- **Write Code**: Type Python in the editor.
- **Run**: Click the **▶ RUN** button (or press `Ctrl+Enter`).
- **Clear**: Click the refresh icon to clear the output.

![Main Interface](screenshots/main.png)

### 2. Using Templates
Stuck? Need inspiration?
- Click the **Menu** icon (☰) on mobile, or check the sidebar on desktop.
- Select a template like **Fibonacci**, **Sorting**, or **Neural Networks**.
- The code will instantly load into the editor. Run it and see what happens!

### 3. Neural Network Builder (NEW!)
Want to build AI?
1. Select the **Neural Networks** template 🧠.
2. You will see a glowing network visualization.
3. Click the **⚙️ BUILDER** button in the overlay.
4. **Customize**:
   - Add/Remove layers.
   - Change layer sizes (how many "neurons").
   - Pick activation functions (ReLU, Sigmoid, etc.).
5. **Generate**: Click **Generate & Run**.
   - **Pure Python**: Runs a simulation instantly in the browser.
   - **PyTorch/TensorFlow**: Generates professional code you can copy and use in real AI projects!

![Neural Builder](screenshots/builder.png)

### 4. Mobile Experience
YUV.PY is fully responsive. Code on your phone with a dedicated mobile view!

![Mobile View](screenshots/mobile.png)

## 📦 Features Summary
- **Zero Install**: Powered by Pyodide (WebAssembly).
- **Responsive Design**: Works on Desktop, Tablet, and Mobile.
- **Real Python**: Supports standard library, math, random, and more.
- **Visual AI**: Interactive particle system visualizing neural connections.

## 🤝 Contributing
Feel free to fork this repository and submit Pull Requests. We love open source!

---
*Created with ❤️ by Yuval Avidani - YUV.AI*
