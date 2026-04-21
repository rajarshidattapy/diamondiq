import os, numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

train_df = pd.read_csv('artifacts/train.csv')
test_df  = pd.read_csv('artifacts/test.csv')

target = 'price'
drop   = [target, 'id']

X_train = train_df.drop(columns=drop)
y_train = train_df[target]
X_test  = test_df.drop(columns=drop)
y_test  = test_df[target]

num_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
cat_cols = ['cut', 'color', 'clarity']

num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[
        ['Fair','Good','Very Good','Premium','Ideal'],
        ['D','E','F','G','H','I','J'],
        ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
    ])),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

X_train_t = preprocessor.fit_transform(X_train)
X_test_t  = preprocessor.transform(X_test)

model = XGBRegressor(max_depth=9, learning_rate=0.16, n_estimators=300,
                     objective='reg:squarederror', n_jobs=-1)
model.fit(X_train_t, y_train)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, model.predict(X_test_t))
print(f"R2 score: {r2:.4f}")

joblib.dump(preprocessor, 'artifacts/preprocessor.pkl')
joblib.dump(model,        'artifacts/model.pkl')
print("Saved preprocessor.pkl and model.pkl")
