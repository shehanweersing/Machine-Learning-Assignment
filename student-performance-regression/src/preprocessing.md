 # Preprocessing Strategy & Justification 
 
 ## Missing Value & Duplicate Audit: 
 Confirmed that the dataset contains zero missing values and no duplicate row records, rendering imputation or deletion routines unnecessary.  
 
 ## Categorical Feature Encoding: 
 Applied One-Hot Encoding to all nominal categorical features (e.g., Mjob, Fjob, reason, guardian) using OneHotEncoder(handle_unknown='ignore'). This prevents linear models and tree algorithms from assuming false ordinal or mathematical hierarchies among categorical variables.  
 
 ## Numerical Feature Scaling: 
 Standardised all continuous numerical attributes (such as age, absences) to a zero mean and unit variance using StandardScaler. This ensures that features with larger numeric ranges do not disproportionately bias distance-based or regularised models.  
 
 ## Data Leakage and Progression Control: 
 Explicitly managed intermediate term grades (G1 and G2). Retaining them establishes an academic progression-tracking pipeline, while isolating the target variable (G3) prevents target leakage during feature transformation splits