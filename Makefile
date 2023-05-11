python_cmd := python
pip_cmd := $(python_cmd) -m pip

clean:
	$(pip_cmd) uninstall -y rcal
	rm -rf dist
	rm -rf build
	rm -rf rcal.egg-info

install:
	$(pip_cmd) install --upgrade pip
	$(pip_cmd) install -e .

dev_install:
	$(pip_cmd) install --upgrade pip
	$(pip_cmd) install -e .
	$(pip_cmd) install -r requirements-dev.txt

test:
# 	$(python_cmd) -m pydocstyle convention=numpy rcal
# 	$(python_cmd) -m pytest --codestyle --cov=./ --cov-report=xml
	$(python_cmd) -m pytest
	$(python_cmd) -m build
	$(python_cmd) -m twine check dist/*
