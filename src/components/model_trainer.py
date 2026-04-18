import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from functools import partial
import optuna
import os
import pickle
import logging
import joblib
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def objective(self, trial, model_name, X_train, y_train):
        if model_name == 'Lasso':
            alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
            model = Lasso(alpha=alpha)
        elif model_name == 'Ridge':
            alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
            model = Ridge(alpha=alpha)
        elif model_name == 'ElasticNet':
            alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        elif model_name == 'DecisionTree':
            max_depth = trial.suggest_int("max_depth", 2, 32)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
        elif model_name == 'XGBoost':
            max_depth = trial.suggest_int("max_depth", 2, 32)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, objective='reg:squarederror')
        else:
            model = LinearRegression()

        score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()
        return score

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = ['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet', 'DecisionTree', 'XGBoost']
            best_models = {}

            for model_name in models:
                logging.info(f'Starting Optuna study for {model_name}')
                study = optuna.create_study(direction='maximize')
                study.optimize(partial(self.objective, model_name=model_name, X_train=X_train, y_train=y_train), n_trials=50)

                logging.info(f"Best trial for {model_name}: {study.best_trial.params} with R2 score: {study.best_value}")
                best_models[model_name] = study.best_value

            logging.info(f"Model report: {best_models}")

            best_model_name = max(best_models, key=best_models.get)
            best_model_score = best_models[best_model_name]

            # Rebuild the best model with its best parameters
            best_trial_params = study.best_trial.params
            if best_model_name == 'Lasso':
                best_model = Lasso(**best_trial_params)
            elif best_model_name == 'Ridge':
                best_model = Ridge(**best_trial_params)
            elif best_model_name == 'ElasticNet':
                best_model = ElasticNet(**best_trial_params)
            elif best_model_name == 'DecisionTree':
                best_model = DecisionTreeRegressor(**best_trial_params)
            elif best_model_name == 'XGBoost':
                best_model = XGBRegressor(**best_trial_params)
            else:
                best_model = LinearRegression()

            logging.info(f'Best Model Found, Model name: {best_model_name}, R2 score: {best_model_score}')
            print(f'Best Model Found, Model name: {best_model_name}, R2 score: {best_model_score}')

            # Save the best model
            joblib.dump(best_model, self.model_trainer_config.trained_model_file_path)
            logging.info(f'Best model saved to {self.model_trainer_config.trained_model_file_path}')

            return best_model

        except Exception as e:
            logging.error("Exception occurred at Model Training")
            raise e

# Example usage (make sure to provide train_array and test_array as numpy arrays)
# train_array = np.array(...)  # Load your training data
# test_array = np.array(...)    # Load your test data
# model_trainer = ModelTrainer()
# model_trainer.initiate_model_training(train_array, test_array)
