import sys
from pathlib import Path
import pandas, numpy, requests, sklearn, streamlit, plotly
print("Python:", sys.version.split()[0])
print("Project folder:", Path(__file__).resolve().parents[1])
print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("streamlit:", streamlit.__version__)
print("plotly:", plotly.__version__)
print("Environment check passed.")
