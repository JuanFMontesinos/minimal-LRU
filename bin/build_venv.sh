PYTHON_PATH=~/.pyenv/versions/3.10.6/bin/python3.10
VENV_DIR=./venv
mkdir $VENV_DIR

# Create a virtual environment
virtualenv -p $PYTHON_PATH $VENV_DIR