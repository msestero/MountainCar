# Makefile for managing Python virtual environment

# Variables
VENV_DIR = venv
PYTHON = python3
REQUIREMENTS = requirements.txt

# Default target
all: $(VENV_DIR)

# Create the virtual environment
$(VENV_DIR): $(REQUIREMENTS)
	@echo "Creating virtual environment in $(VENV_DIR)..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Activating virtual environment and installing requirements..."
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r $(REQUIREMENTS)
	@echo "Virtual environment setup complete."

# Clean up the virtual environment
clean:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Clean-up complete."

# Recreate the virtual environment
rebuild: clean all
	@echo "Virtual environment rebuilt."

# Help command
help:
	@echo "Makefile for Python Virtual Environment Management"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Create the virtual environment and install dependencies"
	@echo "  clean     - Remove the virtual environment"
	@echo "  rebuild   - Recreate the virtual environment"
	@echo "  help      - Display this help message"
