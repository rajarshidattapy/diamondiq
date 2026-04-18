########################################################
# score.py (Save this in the same directory as the main script)
########################################################

import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('diamond_price_predictions')  # The name used when registering the model
    model = joblib.load(model_path)

def run(raw_data):
    try:
        # Parse input data
        data = json.loads(raw_data)["data"]
        data = np.array(data).reshape(1, -1)

        # Make prediction
        result = model.predict(data)

        # Return prediction as JSON
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
