# CLAUDE.md - Liberty AI Delivery Manager Case

## Project Context
This project is an end-to-end Machine Learning case for Liberty Latin America (CellMov). The goal is to build a contextual recommendation model (Contextual Bandit) to optimize offer personalization in a Customer Value Management (CVM) campaign, maximizing total net margin.

*   **Primary Instructions Source:** Always refer to **`docs/case_instructions.md`** as the ground truth for business requirements, technical specifications, and specific evaluation criteria.

## Environment & Run Commands
- **Environment:** Local PC / VSCode using **Python 3.13.5** (Conda/Virtual environment is named `Dev`).
- **Data Files Location:** `./data_and_supporting_code/`
- **Main Notebook:** `./CVM Simulation - Case.ipynb`
- **Supporting Code:** `./data_and_supporting_code/base.py`, `./data_and_supporting_code/experiments.py`, `./data_and_supporting_code/sim.py`

### Execution Commands
- Run notebook locally: `jupyter notebook` or execute cells directly in VSCode using the `Dev` kernel.
- Export notebook to HTML: 
  `jupyter nbconvert --to html --output-dir='./deliverables' './CVM Simulation - Case.ipynb'`

## Core Development Guidelines
- **Language Constraint:** **ALL code, docstrings, variable names, comments, commit messages, and documentation within scripts or notebooks MUST be written in English.**
- **Collaborative AI Constraint:** DO NOT solve the case or run simulations autonomously. Work step-by-step alongside the user.
- **Coding Style:** Clean, highly readable, production-grade Python. 
- **Explainability:** Code must be simple enough to be easily modified live during an executive presentation. Prioritize robust, well-explained linear classifiers or simple tree models over black-box complexity.
- **No Reverse Engineering:** The simulation engine (`sim()`) must be treated as a black-box environment. Do not hardcode rules based on its internals.
- **Bandit Policy:** Use a Contextual Bandit approach (e.g., Logistic Regression / LightGBM with epsilon-greedy exploration) to balance exploration and exploitation across the 10 sequential training datasets.

## Key Deliverables Checklist (Target Directory: `./deliverables/`)
1. Completed Jupyter Notebook: `./CVM Simulation - Case.ipynb`
2. Exported HTML: `./deliverables/CVM Simulation - Case.html`
3. Prompt Log: `./deliverables/prompt_log.txt` (Documentation of AI prompts used)
