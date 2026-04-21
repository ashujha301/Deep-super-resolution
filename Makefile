# ---- Environment setup
setup:
	conda env create -f environment.yml || true
	pip install -r requirements.txt

# ---- Run tests
test:
	pytest tests/

test-unit:
	pytest tests/unit/

# ---- Training & evaluation
train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

# ---- Benchmarking
benchmark:
	pytest tests/performance/

# ---- Run API server
serve:
	uvicorn deployment.api.main:app --reload

# ---- Build C++ extensions
build-cpp:
	mkdir -p cpp/build
	cd cpp/build && cmake .. && make

# ---- Clean project
clean:
	rm -rf __pycache__
	rm -rf *.pyc
	rm -rf cpp/build
	rm -rf build dist *.egg-info

# ---- Linting (optional for now)
lint:
	echo "Add flake8/black later"

# ---- Docs placeholder
docs:
	echo "Docs generation later"