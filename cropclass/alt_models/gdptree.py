import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from datetime import datetime


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# x and y data
x_ds = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/training_data/x_y/x_6-dates')
x_ds = x_ds.drop([0])
x_ds = x_ds.apply(pd.to_numeric)

# Normalize data TODO: this should happen after train/test split - normalize test data with train mu/sd
x_norm = (x_ds - x_ds.mean()) / x_ds.std()

y_ds = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/training_data/x_y/y_6-dates', header=None)
y = y_ds.values.ravel()

# Train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.20, random_state=42)

# fit model on training data
model = xgb.XGBClassifier()
model.fit(x_train, y_train)

# make predictions for test data
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

plot_importance(model)


# ---------- Fit XGBoost model with a random parameter grid search ----------- #

model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1000, objective='multi:softmax', silent=True)

# Parameter grid for XGBoost
params = {'learning_rate': [0.01, 0.1, 0.2],
          'min_child_weight': [1, 5, 10],
          'gamma': [0.5, 1, 1.5, 2, 5],
          'subsample': [0.6, 0.8, 1.0],
          'colsample_bytree': [0.6, 0.8, 1.0],
          'max_depth': [3, 4, 5, 7],
          'n_estimators': [50, 100, 250, 500, 1000, 2500]}

# K-fold cross-validation
folds = 5
param_comb = 5

#
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

# Set-up grid-search
random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb, scoring='accuracy',
                                   n_jobs=4, cv=skf.split(x_norm, y), verbose=3, random_state=1001)

# Perform and time grid-search
start_time = timer(None)
random_search.fit(x_norm, y)
timer(start_time)

print(random_search.best_estimator_)
print(random_search.best_score_ * 100)
print(random_search.best_params_)


# Best model parameters
final_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=2500, objective='multi:softmax', silent=True,
                                subsample=0.8, min_child_weight=1, max_depth=4, gamma=0.5, colsample_bytree=0.6)

x_train, x_test, y_train, y_test = train_test_split(x_ds, y, test_size=0.20, random_state=42)

mu = x_train.mean()
sd = x_train.std()

x_train_norm = (x_train - mu) / sd
x_test_norm = (x_test - mu) / sd

final_model.fit(x_train_norm, y_train)

# make predictions for test data
y_pred = final_model.predict(x_test_norm)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Variable importance
plot_importance(final_model)