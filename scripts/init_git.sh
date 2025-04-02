#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Initializing Git repository for MST Algorithms project...${NC}"

# Initialize the Git repository
git init

# Add the .gitignore file
if [ -f ".gitignore" ]; then
    echo -e "${GREEN}Adding .gitignore file...${NC}"
    git add .gitignore
    git commit -m "Initial commit: Add .gitignore"
else
    echo "Error: .gitignore file not found. Please create it first."
    exit 1
fi

# Add the README.md
if [ -f "README.md" ]; then
    echo -e "${GREEN}Adding README.md...${NC}"
    git add README.md
    git commit -m "Add README.md with project documentation"
else
    echo "Warning: README.md not found."
fi

# Add source code files
echo -e "${GREEN}Adding source code files...${NC}"
git add src/*.py
git commit -m "Add source code implementation"

# Add requirements.txt
if [ -f "requirements.txt" ]; then
    echo -e "${GREEN}Adding requirements.txt...${NC}"
    git add requirements.txt
    git commit -m "Add project dependencies"
else
    echo "Warning: requirements.txt not found."
fi

# Add run.sh script
if [ -f "run.sh" ]; then
    echo -e "${GREEN}Adding run.sh script...${NC}"
    git add run.sh
    git commit -m "Add execution script"
else
    echo "Warning: run.sh not found."
fi

# Add documentation files
echo -e "${GREEN}Adding documentation files...${NC}"
git add docs/*
git commit -m "Add project documentation"

echo -e "${BLUE}Git repository initialization complete!${NC}"
echo -e "${BLUE}Next steps:${NC}"
echo -e "1. Create a remote repository on GitHub or similar service"
echo -e "2. Add the remote: ${GREEN}git remote add origin <repository-url>${NC}"
echo -e "3. Push your code: ${GREEN}git push -u origin main${NC}" 