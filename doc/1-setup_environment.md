1. Clone the repository

```
git clone https://github.com/eugeniaring/topic-modeling-reviews.git
````

2. Create a virtual environment in Python

Windows commands

```
py -m venv venv
echo venv >> .gitignore
venv/Scripts/activate 
````

Linux/Ubuntu commands
```
python3 -m venv venv
source venv/bin/activate
```

3. DVC

Install DVC and fastds
```
pip install dvc
pip install fastds
```

If DVC gives errors:
```
pip install --upgrade pip setuptools wheel
```

create data and model folders
```
mkdir -p data model
```

initialize both Git and DVC with one command
```
fds init
```

Add these folders into gitignore
```
fds add data model
dvc add model
git add model.dvc . gitignore
```

Git remote
```
dvc remote add origin https://dagshub.com/eugenia.anello/topic-modeling-reviews.dvc
dvc remote modify origin --local auth basic 
dvc remote modify origin --local user <username>
dvc remote modify origin --local password <password> 
```
