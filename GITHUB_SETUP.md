# 📦 GitHub Repository Setup Guide

This guide will help you publish your PyWASM Terminal to GitHub.

## 🚀 Quick Steps to Create GitHub Repository

### Step 1: Create Repository on GitHub

1. Go to [GitHub.com](https://github.com) and log in
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name**: `pywasm-terminal` (or your preferred name)
   - **Description**: "🐍 Execute Python code in your browser using WebAssembly - Cyberpunk themed terminal"
   - **Visibility**: ✅ **Public** (so others can see it!)
   - **DO NOT** initialize with README (we already have one)
5. Click **"Create repository"**

### Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these in your terminal:

```bash
# Add the GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/pywasm-terminal.git

# Push your code to GitHub
git branch -M main
git push -u origin main
```

**Replace `YOUR_USERNAME`** with your actual GitHub username!

### Step 3: Add Screenshots (Optional but Recommended!)

To make your README look amazing with screenshots:

1. Open your application at `http://localhost:8000`
2. Take screenshots:
   - Full page view (hero shot)
   - Code editor with sample code
   - Terminal output showing results
   - Mobile view (resize browser or use browser dev tools)

3. Save screenshots in the `screenshots` folder with these names:
   - `hero.png` - Main banner image
   - `main-interface.png` - Full interface
   - `execution.png` - Code running with output
   - `mobile.png` - Mobile responsive view
   - `dark-theme.png` - Close-up of theme details

4. Commit and push the screenshots:
```bash
git add screenshots/
git commit -m "Add screenshots to README"
git push
```

### Step 4: Enable GitHub Pages (Free Hosting!)

Host your app for FREE on GitHub Pages:

1. Go to your repository on GitHub
2. Click **"Settings"** tab
3. Scroll to **"Pages"** in the left sidebar
4. Under **"Source"**, select:
   - Branch: `main`
   - Folder: `/ (root)`
5. Click **"Save"**
6. Wait 1-2 minutes
7. Your site will be live at: `https://YOUR_USERNAME.github.io/pywasm-terminal/`

## 🎯 Quick Command Reference

```bash
# Check status
git status

# Add new files
git add .

# Commit changes
git commit -m "Your commit message"

# Push to GitHub
git push

# View remote URL
git remote -v
```

## 🏷️ Add Topics to Your Repository

Make your repo discoverable! Add these topics on GitHub:

1. Go to your repository
2. Click the gear icon ⚙️ next to "About"
3. Add topics: `python`, `webassembly`, `pyodide`, `terminal`, `browser`, `cyberpunk`, `javascript`, `html5`, `css3`, `web-app`
4. Add website URL if using GitHub Pages
5. Save changes

## 📱 Share Your Project

After publishing, share it on:

- Twitter/X: "Just built a cyberpunk Python terminal that runs in the browser! 🐍✨ #Python #WebAssembly"
- LinkedIn: Write a post about your project
- Reddit: r/Python, r/webdev, r/programming
- Dev.to: Write an article about how you built it
- Hacker News: Share when it's ready

## 🎉 You're Done!

Your PyWASM Terminal is now:
- ✅ Published on GitHub
- ✅ Version controlled
- ✅ Shareable with anyone
- ✅ (Optional) Hosted online for free

## 🐛 Troubleshooting

### "Permission denied" error
- Make sure you're logged into GitHub
- Use HTTPS URL or set up SSH keys
- Try: `gh auth login` if using GitHub CLI

### "Repository not found"
- Double-check the repository URL
- Ensure the repository exists on GitHub
- Verify you have write access

### Screenshots not showing
- Make sure images are in the `screenshots/` folder
- Check image names match the README
- Commit and push the images
- Wait a few minutes for GitHub to update

## 📞 Need Help?

If you encounter issues:
1. Check GitHub's documentation
2. Search for the error message online
3. Ask on GitHub Community Discussions
4. Reach out to the creator (Yuval Avidani) via social links in README

---

**Happy coding! 🚀**

Created with ❤️ by Yuval Avidani | [yuv.ai](https://yuv.ai)
