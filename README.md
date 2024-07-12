# Incident Prediction and Police Allocation Model

## Overview
This project predicts incident counts using a RandomForest model based on the month, district, and neighborhood. The model utilizes features such as time (encoded month or hour), normalized incident counts, and cluster information derived from K-means clustering, along with categorical data like police districts and neighborhoods.

## Getting Started
You can find our app running at https://policeallocation.streamlit.app/

For running this repository locally
### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- pandas
- numpy
- scikit-learn
- streamlit
- joblib

You can install the necessary libraries using `pip`:
```bash 
pip install pandas numpy scikit-learn streamlit
```
```Then proceed to clone the repository 
git clone https://github.com/your-username/your-repository.git
```

```cd your-repository```

## Usage
To run the Streamlit application:
```streamlit run app.py```

Navigate to http://localhost:8501 in your web browser to interact with the app.





