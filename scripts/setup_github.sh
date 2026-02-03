#!/bin/bash

# Event Discovery Framework - GitHub Setup Script
# This script sets up the repository and pushes to GitHub

set -e  # Exit on error

echo "=========================================="
echo "Event Discovery Framework - GitHub Setup"
echo "=========================================="
echo ""

# Configuration
REPO_URL="https://github.com/meshal-alawein/event-discovery-framework.git"
REPO_NAME="event-discovery-framework"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "✓ Git initialized"
else
    echo "✓ Git already initialized"
fi

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p data
mkdir -p results/figures
mkdir -p results/tables
mkdir -p results/videos
mkdir -p paper/figures
echo "✓ Directories created"

# Create empty __init__.py files for Python packages
echo ""
echo "Setting up Python package structure..."
touch src/__init__.py
touch src/methods/__init__.py
touch src/core/__init__.py
touch src/utils/__init__.py
echo "✓ Python packages configured"

# Create example annotations file
echo ""
echo "Creating example annotations..."
cat > data/annotations.json << 'EOF'
{
  "events": [
    {
      "start_time": 10.0,
      "end_time": 12.5,
      "label": "lane_change",
      "description": "Vehicle performs unsafe lane change without signaling"
    },
    {
      "start_time": 45.2,
      "end_time": 47.8,
      "label": "near_miss",
      "description": "Near collision with pedestrian crossing street"
    },
    {
      "start_time": 120.5,
      "end_time": 123.0,
      "label": "traffic_violation",
      "description": "Running red light at intersection"
    }
  ],
  "metadata": {
    "video_duration": 600.0,
    "annotator": "example",
    "date": "2026-02-02"
  }
}
EOF
echo "✓ Example annotations created"

# Create data download script
echo ""
echo "Creating data download script..."
cat > scripts/download_data.sh << 'EOF'
#!/bin/bash

# Download example data (placeholder)
# Replace with actual dataset URLs

echo "Downloading example video..."
# wget -O data/example_video.mp4 URL_TO_VIDEO

echo "For now, please manually add video files to data/"
echo "Supported formats: .mp4, .avi, .mov"
EOF
chmod +x scripts/download_data.sh
echo "✓ Download script created"

# Create LICENSE
echo ""
echo "Creating MIT License..."
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2026 Meshal Alshammari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
echo "✓ License created"

# Git configuration
echo ""
echo "Configuring git..."
git add .
git status
echo ""

# Commit
read -p "Do you want to commit these changes? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git commit -m "Initial commit: Physics-inspired event discovery framework

- Complete LaTeX paper with mathematical framework
- Implementation of 6 methods (Hierarchical Energy, Geometric Outlier, etc.)
- Colab-ready demo notebook
- Comprehensive documentation
- Evaluation scripts and figure generation
- Ready for publication and deployment"
    echo "✓ Changes committed"
fi

# Add remote if not exists
echo ""
if ! git remote | grep -q origin; then
    echo "Adding remote repository..."
    git remote add origin $REPO_URL
    echo "✓ Remote added"
else
    echo "✓ Remote already configured"
fi

# Push to GitHub
echo ""
read -p "Do you want to push to GitHub now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pushing to GitHub..."
    git branch -M main
    git push -u origin main
    echo "✓ Pushed to GitHub"
    echo ""
    echo "Repository URL: $REPO_URL"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Add example video to data/example_video.mp4"
echo "2. Run demo: jupyter notebook notebooks/01_demo_quick.ipynb"
echo "3. Compile paper: cd paper && make pdf"
echo "4. Share Colab link: https://colab.research.google.com/github/meshal-alawein/$REPO_NAME/blob/main/notebooks/01_demo_quick.ipynb"
echo ""
echo "Good luck! 🚀"
