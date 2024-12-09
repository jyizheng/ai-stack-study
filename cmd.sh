#!/bin/bash

pip install service_identity

python -c "import service_identity; print(service_identity.__version__)"

pytest --markers

pytest -s
