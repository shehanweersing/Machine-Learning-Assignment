import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessor(X):
    """
    Constructs a ColumnTransformer to handle numerical scaling 
    and categorical one-hot encoding safely.
    """
    # Identify column types dynamically
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Define preprocessing steps for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

if __name__ == '__main__':
    # Test pipeline loading
    df = pd.read_csv('../data/student-mat.csv', sep=';')
    X = df.drop(columns=['G3'])
    preprocessor = get_preprocessor(X)
    print("Preprocessing pipeline successfully configured.")