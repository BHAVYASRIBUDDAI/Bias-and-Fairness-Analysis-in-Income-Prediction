import numpy as np
from folktables import ACSDataSource, ACSIncome

def load_data(state="CA"):
    # Just grabbing the 2018 census data
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=[state], download=True)
    
    # Standard Income task: X is features, y is labels (>50k income)
    X, y, _ = ACSIncome.df_to_numpy(acs_data)
    
    # We need the Sex column for the fairness check
    # In ACSIncome, sex is usually one of the features
    sex_idx = ACSIncome.features.index("SEX")
    sex_labels = X[:, sex_idx]
    
    return X, y, sex_labels