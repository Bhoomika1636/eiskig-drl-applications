# EISKIG DRL Applications

This repository is a fork of the repo [experiments_hr](https://git.ptw.maschinenbau.tu-darmstadt.de/eta-fabrik/projekte/experiments_hr), where major developments of the framework took place.. The **EISKIG DRL Applications** Repo is used to bundle the use cases of the **EISKIG project**.

Welcome! This repository contains experiments built on the [eta-utility functions](https://eta-utility.readthedocs.io/en/master/).
It provides a unified file and folder structure for connecting algorithms (DRL or conventional) to environments (simulated or real) as well as a growning number of experiments.

**The code of this repository**
* works on Windows or Linux systems,
* can make use of CUDA ready graphics card, if available,
* is maintained by Heiko Ranzau, please feel free to contact me with questions.

> **IMPORTANT**
> * Do not publish or upload the files contained in this repository anywhere else! Some files are subject to confidentiality agreements.
> * Make sure to only work in your branch - your branch will be merged into development when your feature is complete and your code is checked.
> * Never upload large files (the results folder is added to the .gitignore file for this reason).


## Recommended Software
* [Visual Studio Code](https://code.visualstudio.com/download)
* [GitHub Desktop](https://desktop.github.com/)
* [Python 3.11.0 (64 Bit)](https://www.python.org/downloads/release/python-3110/)

## Installation Guide
1. Install Python, make sure you add it to PATH during the installation.
2. Install GitHub Desktop and clone this repository (File > Clone Repository > URL).
3. In GitHub Desktop, create your own branch (Branch > New Branch > Your Name in format: Ranzau_H_Title) from the "main" branch and choose a suitable local path with write permissions and sufficient capacity (> 20 GB, in case you do a lot of experiments).
4. Install Visual Studio Code and the extensions "Python" and "GitLens".
5. In VS Code, open the folder that has been synchronized with GitHub Desktop to the local path (File > Open Folder). If the correct path is chosen the setup.py should be visible in the VS Code explorer.
6. Install all necessary packages in a virtual environment:
    - Open a new terminal (Terminal > New Terminal).
    - Select CMD as your default terminal provile (click on the down-facing arrow on the top right corner of the terminal sub-window > Select Default Profile > Command Prompt).
    - In the terminal, type: ```where python ``` and copy the link to the correct version.
    - Type ```"Link to python" -m venv .venv```. You should be able to see the .venv folder in your VS Code explorer. This creates a virtual environment, where all necessary packages will be installed together with the chosen python version. VS Code might ask you if it should automaticly activate the virtual environment at next start-up > accept.
    - Type ```.venv\scripts\activate```. This activates your virtual environment for the first time.
    - Type ```pip install -e.```. This installs all packages.
7. After the installation is done, you can run the following experiment: experiments_hr > supplysystems_a > experiments_rule_based.py by clicking the right facing arrow in the top right corner or via Run > Run Without Debugging. You should see some output in the terminal and inside the supplysystems_a folder, there should be a new folder called "results" now, where you find some graphs of the experiment.
8. If you want to create your own experiment, just use the template folder or copy the supplysystem_a folder and change it to your needs.

9. If you want to use the GPU CUDA acceleration, please make sure to follow the instructions provided by NVIDIA for your own hardware setup. If you are using the ETA-Analytics 3 Server, you can enable CUDA support by additionally installing the neccessary packages via```pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113```

## Common Errors
* Sometimes it is necessary to restart VS Code after the first installation of all packages for it to properly detect them.
* The installation may take quite long > 20 minutes, when pip is scanning for package interoperability. Usually it should take less than 5 minutes.
* There currently is an error when using CUDA with advanced extractors.

