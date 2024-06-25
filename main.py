
import json
import ray
import requests
from ray import serve
from ray import tune
from ray.train import ScalingConfig, RunConfig
from ray.train.xgboost import XGBoostTrainer
from ray.tune import Tuner, TuneConfig
from starlette.requests import Request

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Load and split dataset
dataset = ray.data.read_parquet("s3://anonymous@anyscale-training-data/intro-to-ray-air/nyc_taxi_2021.parquet").repartition(16)
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)

# Setup and train the XGBoost model
trainer = XGBoostTrainer(
    label_column="is_big_tip",
    scaling_config=ScalingConfig(num_workers=4, use_gpu=False),
    params={"objective": "binary:logistic"},
    datasets={"train": train_dataset, "valid": valid_dataset},
    run_config=RunConfig(storage_path='/mnt/cluster_storage/')
)
result = trainer.fit()

# Tune the model
tuner = Tuner(trainer, 
    param_space={'params': {'max_depth': tune.randint(2, 12)}},
    tune_config=TuneConfig(num_samples=3, metric='train-logloss', mode='min'),
    run_config=RunConfig(storage_path='/mnt/cluster_storage/')
)
checkpoint = tuner.fit().get_best_result().checkpoint

# Offline Predictor Class
class OfflinePredictor:
    def __init__(self):
        import xgboost
        self._model = xgboost.Booster()
        self._model.load_model(checkpoint.path + '/model.json')
    def __call__(self, batch):
        import xgboost
        import pandas as pd
        dmatrix = xgboost.DMatrix(pd.DataFrame(batch))    
        outputs = self._model.predict(dmatrix)
        return {"prediction": outputs}

predicted_probabilities = trainer.predict()

# Online Predictor Deployment
@serve.deployment
class OnlinePredictor:
    def __init__(self, checkpoint):
        import xgboost
        self._model = xgboost.Booster()
        self._model.load_model(checkpoint.path + '/model.json')        
        
    async def __call__(self, request: Request) -> dict:
        data = await request.json()
        data = json.loads(data)
        return {"prediction": self.get_response(data)}
    
    def get_response(self, data):
        import xgboost
        import pandas as pd
        dmatrix = xgboost.DMatrix(pd.DataFrame(data, index=[0])) 
        return self._model.predict(dmatrix)

handle = serve.run(OnlinePredictor.bind(checkpoint=checkpoint))

# Example input
sample_input = valid_dataset.take(1)[0]
del(sample_input['is_big_tip'])
del(sample_input['__index_level_0__'])
response = requests.post("http://localhost:8000/", json=json.dumps(sample_input)).json()
print(response)

# Shutdown the serve
serve.shutdown()

# Chat Deployment
@serve.deployment
class Chat:
    def __init__(self, msg: str):
        self._msg = msg # initial state
    async def __call__(self, request: Request) -> dict:
        data = await request.json()
        data = json.loads(data)
        return {"result": self.get_response(data['input'])}
    
    def get_response(self, message: str) -> str:
        return self._msg + message

handle = serve.run(Chat.bind(msg="Yes... "), name='hello_world')

# Test the Chat deployment
sample_json = '{ "input" : "hello" }'
response = requests.post("http://localhost:8000/", json=sample_json).json()
print(response)

# Ray Serve management
serve.status()
serve.delete('hello_world')
