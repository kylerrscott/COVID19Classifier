# COVID-19 Classifier

---

## Quick Start

Howdy everyone! Welcome to the *COVID-19 Classifier* repository!

Here is a walkthrough of all of the tools that you will need and how to install all of them.

**IF YOU HAVE ANY QUESTIONS, PLEASE REACH OUT TO KYLE**

##### Windows Only:
1. WSL
2. Ubuntu for WSL
3. Windows Terminal (optional, but a very good terminal)

##### All:
1. Visual Studio Code or PyCharm
2. git
3. This repo 

### Windows Subsystem for Linux (WSL)
This is a low overhead virtual machine that runs a Linux distrubtion on a Windows computer. We will be using this to standardize all of our terminals/command prompts in a Unix-like shell.
1. Run Windows PowerShell as Administrator
    1. Press windows key and type PowerShell, click `Run as Administrator` on the right side of the search window
2. Run the command `dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart`
3. Restart your computer
4. WSL should be activated!

### Ubuntu for WSL
1. Open the `Microsoft Store`, press windows key and type Microsoft Store.
2. Search for `Ubuntu` and install.
3. You should now be able to open the `Ubuntu` application on your computer and have a fully functional Unix-like shell

### Windows Terminal
1. Open the `Microsoft Store`
2. Search for `Windows Terminal` and install
3. You should now have a terminal that will allow you to run PowerShell and Ubuntu
4. Press the drop down arrow at the top to open an `Ubuntu` terminal.

### Visual Studio Code
1. Download from `https://code.visualstudio.com/`
2. Open Visual Studio Code
3. Go to Extensions (or press Ctrl+Shift+X) and search and install for the extension `Remote - WSL`
4. You can close this for now

### Python
1. Download the latest version of Python from `https://www.python.org/`
2. Install

### git 
1. Check if already installed:
    1. Open Ubuntu in Windows Terminal or Terminal on Mac
    2. run `git --version` if it returns a version number, then you're done!
2. If you do not already have git installed, continue
3. Run `sudo apt-get install git` to install git
4. Check that it installed correctly `git --version`
#### git config: Set up Git config files for easier interfacing with github
1. run `git config --global user.name "YourGitHubName"`
2. run `git config --global user.email "youremail@domain.com"` with the email associated with your GitHub account

### This Repository! Finally!!!
1. Now that we have everything else downloaded, we are ready to get the project files on your computer.
2. Open `Ubuntu` in Windows Terminal or open `Terminal` on Mac
3. Navigate to wherever you want to save this repository on your filesystem, you do not need to make a folder for the project, it will come in a folder labeled `COVID19Classifier`.
4. Run `git clone https://github.com/AMyscich/COVID19Classifier.git COVID19Classifier`
5. Navigate into the project, `cd COVID19Classifier`

**CONGRATS YOU'RE DONE!**

---

## Typical work flow (Coding/Git guide)
1. Navigate to the `/COVID19Classifier` file on your computer in Terminal (Ubuntu in Windows Terminal or Terminal on Mac)
2. Run `code .` to open this folder in Visual Studio code using a Unix-style terminal.
3. Open the terminal in VS Code (toggle using <code>Ctrl+`</code> on Windows or <code>Cmd+`</code> on Mac)
4. Create a new **branch** in git by using `git checkout -b new-branch-name`. See below for more information on git.
5. Make changes to files, complete tasks.
6. Run `git status` to see what files you have made changes to.
7. Queue changes up to save to git using `git add filename` or add all using `git add .`
8. Make a commit to your local git repository using `git commit -m "Your commit message"`
9. Push your local commit to the GitHub repostiory using `git push`
10. When all the features that you are working on are done on your branch, create a Pull Request on GitHub to prepare your changes for the `master` branch. 
11. Your code will then be reviewed by other team members and if approved, your changes will present in the `master` branch. 
12. If there are some problems, just go back to that branch, make the changes, commit, and push to GitHub for code to be rereviewed.

---

## git helpful tips
- Prior to making a new branch, run `git fetch` to know about all changes from other team members from the GitHub repository to your local git repository. `git fetch` will not change any files in your local repository. In order to actually get your branches to have the changes, nagivate to that branch, `git checkout branch-name`, then run `git pull` to update your local files.
- When in doubt, `git status` out! `git status` is your best friend and will help guide you to do what you want to do.
- git can be very intimidating, but you will typically just use a few commands. If you need help, please ask!
- **Merge conflicts**: This can happen when you are trying to do a Pull Request, but there are changes that someone else has made that would conflict with your changes.


## Requirements

- `Python3+`
