# Shared Python environment

All stage folders use D:\Projects\EmoB\code\.venv.

From PowerShell:

```powershell
cd D:\Projects\EmoB\code
.\.venv\Scripts\Activate.ps1
python .\explore\emob_geometry_investigation.py --help
```

Activation is optional; call the interpreter directly from any folder:

```powershell
& 'D:\Projects\EmoB\code\.venv\Scripts\python.exe' 'D:\Projects\EmoB\code\explore\emob_geometry_investigation.py' --help
```

To recreate the environment with the captured package versions:

```powershell
cd D:\Projects\EmoB\code
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

requirements.txt lists direct dependencies; requirements-lock.txt captures installed versions at migration time. The original explore\.venv is retained as a fallback. VS Code settings select the shared interpreter when opening code or explore; if VS Code remembers a previous selection, use Python: Select Interpreter and select code\.venv\Scripts\python.exe.
