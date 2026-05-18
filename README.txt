LectureLens
AI-Powered Lecture Capture System

Created by Aran Randhawa

Overview
LectureLens is a Python-based lecture capture tool designed to help students and teachers create AI-generated notes from classroom whiteboards and screen content.

The program can:
- Capture whiteboard photos from a webcam
- Take screenshots when needed
- Send captures to Claude for AI-generated notes
- Save lecture images and notes into organized session folders
- Generate HTML study notes with math rendering
- Recover notes if the program closes early

This project is mainly designed for math, physics, chemistry, engineering, and other courses where normal audio transcription is not enough because the important content is written visually on the board.

Main Features
- Whiteboard camera capture
- Screenshot capture
- AI note generation using Claude
- Local duplicate checking to avoid repeated captures
- Session folders for each lecture
- Backup text files for recovery
- HTML output files with MathJax support
- Recovery tool for rebuilding notes from backups

Important Security Note
Do NOT upload your API key to GitHub.

Keep these files private:
- config.txt
- .env
- any file containing an Anthropic API key

The program expects the Anthropic API key to be provided through config.txt or an environment variable.

Example config.txt:
ANTHROPIC_API_KEY=your_key_here

Recommended GitHub Folder Structure
LectureLens/
  src/
    lecturelens.py
    recover.py

  docs/
    LectureLens_Documentation.pdf

  README.txt
  requirements.txt
  .gitignore

Do Not Upload
Do not upload these files or folders to GitHub:
- config.txt
- dist/
- build/
- sessions/
- fonts/
- *.exe
- *.pkg
- *.spec
- __pycache__/
- *.pyc

Requirements
The main Python packages used by LectureLens are:
- opencv-python
- anthropic
- matplotlib
- pillow
- fpdf
- pynput
- numpy

Install requirements with:
pip install -r requirements.txt

Example requirements.txt
opencv-python
anthropic
matplotlib
pillow
fpdf
pynput
numpy

How to Run from Source
1. Install Python.
2. Install the required packages:
   pip install -r requirements.txt

3. Create a config.txt file beside the Python file or exe:
   ANTHROPIC_API_KEY=your_key_here

4. Run:
   python src/lecturelens.py

How to Build the EXE
Use PyInstaller:

pyinstaller --onefile src/lecturelens.py --name LectureLens

The final EXE will appear in:
dist/

Copy config.txt beside the EXE before running:
dist/config.txt

How to Use
- SPACE or remote button: capture whiteboard photo
- S: take screenshot
- N: add text note or question
- O: add photo note with board context
- TAB: switch camera if multiple cameras are connected
- ESC: stop session and generate HTML files

Output Files
Each lecture session is saved inside the sessions/ folder.

Typical output:
- 01_photos.html
- 02_ai_notes.html
- 03_summary.html
- images/
- backups/

Recovery
If the program closes early, run recover.py or Recover.exe.

The recovery tool rebuilds the HTML notes using files saved in the backups/ folder.

Cost Notes
Claude API cost depends on:
- number of captures
- number of screenshots
- number of image crops sent
- output length
- whether Smart Lookback is enabled

To reduce cost:
- Use camera capture as the main workflow
- Use screenshots only when needed
- Keep local duplicate checking enabled
- Avoid uploading repeated captures
- Keep Smart Lookback disabled unless needed

Recommended .gitignore
config.txt
.env
*.key

build/
dist/
*.exe
*.pkg
*.spec

sessions/
images/
backups/
fonts/
*.ttf
*.otf
*.woff
*.woff2

__pycache__/
*.pyc

Thumbs.db
.DS_Store
.vscode/

License
Add your license here if you choose to release this publicly.

Notes
This project is experimental and should be tested carefully before being used in real classroom settings.
Always protect API keys and do not commit private lecture data to GitHub.
