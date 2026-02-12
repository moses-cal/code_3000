# packages
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# set seed
seed = 314

def train_model(X, y, seed=seed):
    """
    Build a GBM on given data
    """
    model = GradientBoostingClassifier(
        learning_rate=0.05,  # slower learning
        n_estimators=400,   # for more trees
        max_depth=3,        # shallow trees
        subsample=0.8,      # some randomness added
        min_samples_leaf=10, # prevent tiny leaves
        random_state=seed
    )
    model.fit(X, y)
    return model