import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    
    # Drop rows with missing key values
    df = df.dropna(subset=['Age', 'Gender', 'Location', 'Education', 'Diploma',
                           'Training', 'InternetSpending', 'Satisfaction', 'Job_Search'])
    
    # Label encode categorical features
    le = LabelEncoder()
    for col in ['Gender', 'Location', 'Education', 'Diploma', 'Training', 'Satisfaction', 'Job_Search']:
        df[col] = le.fit_transform(df[col])
    
    # Standardize numerical features
    scaler = StandardScaler()
    df[['Age', 'InternetSpending']] = scaler.fit_transform(df[['Age', 'InternetSpending']])
    
    return df
