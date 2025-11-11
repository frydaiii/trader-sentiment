# Suggested Commands
- `ls`, `tree`: inspect the repository layout (currently only `.serena/`).
- `python3 -m venv .venv && source .venv/bin/activate`: create/activate a virtual environment once dependencies are added.
- `pip install -r requirements.txt` (or `pip install -e .` if packaging is used) after the dependency file is created.
- `pytest` / `python -m pytest`: placeholder test command to adopt when tests arrive.
- `ruff check` / `black .` (or other linters/formatters) can be introduced later; document concrete tooling when chosen.
- `git status` / `git diff`: review pending changes before committing.