pip freeze | xargs pip uninstall -y
pip install --no-cache-dir -r requirements.txt
