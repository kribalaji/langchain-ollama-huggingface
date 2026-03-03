@echo off
REM Git commands for pushing to GitHub

echo Adding all files...
git add .

echo Committing...
git commit -m "Add Ollama and HuggingFace modules with RAG and chatbot examples"

echo Adding remote...
git remote add origin https://github.com/kribalaji/langchain-ollama-huggingface.git

echo Pushing to GitHub...
git push -u origin main

echo Done!
pause
