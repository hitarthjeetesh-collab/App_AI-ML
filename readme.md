Project1
========

Short project description
-------------------------
This small repository contains a minimal PyTorch benchmark script (`main.py`) that multiplies square matrices of increasing size and reports runtime. The script was created as part of a learning exercise.

Repository layout
-----------------
- `main.py` - example script that creates tensors and measures the runtime of matrix multiplication.
- `.gitignore` - files and folders that should be ignored by Git (project-specific rules are provided).

Requirements
------------
- Python 3.8+ (recommended)
- PyTorch (see https://pytorch.org for installation instructions)

Quick start
-----------
1. Create or activate a Python environment (venv or conda).
2. Install dependencies:

   pip install -r requirements.txt

3. Run the script:

   python main.py

Notes
-----
- `main.py` currently selects torch device `xpu` and calls `torch.xpu.synchronize()`. That device name is not available on most machines. To run the script on CPU, edit `main.py` and change the device line:

```python
device = torch.device("cpu")
```

Or, to run on CUDA (GPU), set:

```python
device = torch.device("cuda")
```

- If you plan to run on a GPU, make sure to install the appropriate CUDA-enabled PyTorch build following the instructions on the PyTorch website.

License
-------
MIT License (replace with your preferred license)


