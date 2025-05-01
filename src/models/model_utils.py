import joblib



# Small model utility functions to save and load models:
def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)



# Small utility functions to save and load best parameters:
def save_best_params(params_dict, path):
    joblib.dump(params_dict, path)

def load_best_params(path):
    return joblib.load(path)
