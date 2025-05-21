import subprocess
import sys

# Yüklenmesi gereken paketler
packages = [
    # "flask",
    # "flask-cors",
    # "matplotlib",
    # "seaborn",
    # "scikit-learn",
    # "pandas"
    "tensorflow",
]

# Her paketi tek tek yükle
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
