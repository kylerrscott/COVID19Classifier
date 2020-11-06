# COVID-19 Classifier

Quick Start
Howdy everyone! Welcome to the COVID19Classifier repository!

Here is a walkthrough of all of the tools that you will need and how to install all of them.

IF YOU HAVE ANY QUESTIONS, PLEASE REACH OUT TO KYLE

Windows Only:
WSL
Ubuntu for WSL
Windows Terminal (optional, but a very good terminal)
All:
Visual Studio Code or PyCharm
git

This repo
Windows Subsystem for Linux (WSL)
This is a low overhead virtual machine that runs a Linux distrubtion on a Windows computer. We will be using this to standardize all of our terminals/command prompts in a Unix-like shell.

Run Windows PowerShell as Administrator
Press windows key and type PowerShell, click Run as Administrator on the right side of the search window
Run the command dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
Restart your computer
WSL should be activated!
Ubuntu for WSL
Open the Microsoft Store, press windows key and type Microsoft Store.
Search for Ubuntu and install.
You should now be able to open the Ubuntu application on your computer and have a fully functional Unix-like shell
Windows Terminal
Open the Microsoft Store
Search for Windows Terminal and install
You should now have a terminal that will allow you to run PowerShell and Ubuntu
Press the drop down arrow at the top to open an Ubuntu terminal.
Visual Studio Code
Download from https://code.visualstudio.com/
Open Visual Studio Code
Go to Extensions (or press Ctrl+Shift+X) and search and install for the extension Remote - WSL
You can close this for now

git
Check if already installed:
Open Ubuntu in Windows Terminal or Terminal on Mac
run git --version if it returns a version number, then you're done!
If you do not already have git installed, continue
Run sudo apt-get install git to install git
Check that it installed correctly git --version
git config: Set up Git config files for easier interfacing with github
run git config --global user.name "YourGitHubName"
run git config --global user.email "youremail@domain.com" with the email associated with your GitHub account
This Repository! Finally!!!
Now that we have everything else downloaded, we are ready to get the project files on your computer.
Open Ubuntu in Windows Terminal or open Terminal on Mac
Navigate to wherever you want to save this repository on your filesystem, you do not need to make a folder for the project, it will come in a folder labeled COVID19Classifier.
Run git clone https://github.com/AMyscich/COVID19Classifier.git COVID19Classifier
Navigate into the project, cd COVID19Classifier

CONGRATS YOU'RE DONE!

Feel free to poke around the files. A lot of the files are automatically generated boiler plate. We will write code in the /src/ directory.

Typical work flow (Coding/Git guide)
Navigate to the /COVID19Classifier file on your computer in Terminal (Ubuntu in Windows Terminal or Terminal on Mac)
Run code . to open this folder in Visual Studio code using a Unix-style terminal.
Open the terminal in VS Code (toggle using Ctrl+</code> on Windows or <code>Cmd+ on Mac)
Create a new branch in git by using git checkout -b new-branch-name. See below for more information on git.
Make changes to files, complete tasks.
Run git status to see what files you have made changes to.
Queue changes up to save to git using git add filename or add all using git add .
Make a commit to your local git repository using git commit -m "Your commit message"
Push your local commit to the GitHub repostiory using git push
When all the features that you are working on are done on your branch, create a Pull Request on GitHub to prepare your changes for the master branch.
Your code will then be reviewed by other team members and if approved, your changes will present in the master branch.
If there are some problems, just go back to that branch, make the changes, commit, and push to GitHub for code to be rereviewed.
git helpful tips
Prior to making a new branch, run git fetch to know about all changes from other team members from the GitHub repository to your local git repository. git fetch will not change any files in your local repository. In order to actually get your branches to have the changes, nagivate to that branch, git checkout branch-name, then run git pull to update your local files.
When in doubt, git status out! git status is your best friend and will help guide you to do what you want to do.
git can be very intimidating, but you will typically just use a few commands. If you need help, please ask!
Merge conflicts: This can happen when you are trying to do a Pull Request, but there are changes that someone else has made that would conflict with your changes.
Why
Simple to jump into, Fast because it is simple.
Separate tsconfig.json for client and server.
Client and server can share code (And types). For example: IUserDTO.d.ts
The client is bundled using Webpack because it goes to the browser.
The server is emitted by TypeScript because node 6 supports es6.


Requirements
Python3
