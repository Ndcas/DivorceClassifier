import os
import joblib
import pandas
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")
K_FOLDS = int(os.getenv("K_FOLDS"))
SELECTOR_C = float(os.getenv("SELECTOR_C"))

dataFrame = pandas.read_csv(DATA_PATH)

features = dataFrame.drop("divorced", axis=1)

labels = dataFrame["divorced"]

numericalFeatures = [
    "age_at_marriage",
    "marriage_duration_years",
    "num_children",
    "combined_income",
    "communication_score",
    "conflict_frequency",
    "financial_stress_level",
    "social_support",
    "shared_hobbies_count",
    "trust_score"
]

categoricalCols = [
    "education_level",
    "employment_status",
    "religious_compatibility",
    "conflict_resolution_style",
    "marriage_type"
]

preprocessor = ColumnTransformer(transformers=[
    ("numerical", StandardScaler(), numericalFeatures),
    ("categorical", OneHotEncoder(), categoricalCols)
], remainder="passthrough")


def getKNeighborsClassifier():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("feature_selector", SelectFromModel(LinearSVC(C=SELECTOR_C, penalty="l1", random_state=1))),
        ("classifier", KNeighborsClassifier())
    ])
    param_grid = {
        "classifier__n_neighbors": [2 * i + 1 for i in range(10)],
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "nan_euclidean"]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=K_FOLDS, scoring="f1", n_jobs=-1)
    grid.fit(features, labels)
    n_neighbors = grid.best_params_["classifier__n_neighbors"]
    weights = grid.best_params_["classifier__weights"]
    metric = grid.best_params_["classifier__metric"]
    print(f"n_neighbors: {n_neighbors}, weights: {weights}, metric: {metric}, F1: {grid.best_score_}")
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("feature_selector", SelectFromModel(LinearSVC(C=SELECTOR_C, penalty="l1", random_state=1))),
        ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric))
    ])
    model.fit(features, labels)
    return model


def getDecisionTreeClassifier():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("feature_selector", SelectFromModel(LinearSVC(C=SELECTOR_C, penalty="l1", random_state=1))),
        ("classifier", DecisionTreeClassifier(random_state=1))
    ])
    param_grid = {
        "classifier__criterion": ["gini", "entropy", "log_loss"],
        "classifier__splitter": ["random", "best"],
        "classifier__max_depth": list(range(1, 31))
    }
    grid = GridSearchCV(pipeline, param_grid, cv=K_FOLDS, scoring="f1", n_jobs=-1)
    grid.fit(features, labels)
    criterion = grid.best_params_["classifier__criterion"]
    splitter = grid.best_params_["classifier__splitter"]
    max_depth = grid.best_params_["classifier__max_depth"]
    print(f"criterion: {criterion}, splitter: {splitter}, max_depth: {max_depth}, F1: {grid.best_score_}")
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("feature_selector", SelectFromModel(LinearSVC(C=SELECTOR_C, penalty="l1", random_state=1))),
        ("classifier",
         DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, random_state=1))
    ])
    model.fit(features, labels)
    return model


def getMLPCLassifier():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("feature_selector", SelectFromModel(LinearSVC(C=SELECTOR_C, penalty="l1", random_state=1))),
        ("classifier", MLPClassifier(random_state=1, max_iter=300, early_stopping=True))
    ])
    param_grid = {
        "classifier__hidden_layer_sizes": [(32, 16, 8), (16, 8, 4), (32, 16), (16, 8), (32), (16)],
        "classifier__activation": ["identity", "logistic", "tanh", "relu"],
        "classifier__solver": ["sgd", "adam"],
        "classifier__learning_rate": ["constant", "invscaling", "adaptive"]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=K_FOLDS, scoring="f1", n_jobs=-1)
    grid.fit(features, labels)
    hidden_layer_sizes = grid.best_params_["classifier__hidden_layer_sizes"]
    activation = grid.best_params_["classifier__activation"]
    solver = grid.best_params_["classifier__solver"]
    learning_rate = grid.best_params_["classifier__learning_rate"]
    print(
        f"hidden_layer_sizes: {hidden_layer_sizes}, activation: {activation}, solver: {solver}, learning_rate: {learning_rate}, F1: {grid.best_score_}")
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("feature_selector", SelectFromModel(LinearSVC(C=SELECTOR_C, penalty="l1", random_state=1))),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            learning_rate=learning_rate,
            random_state=1,
            max_iter=300,
            early_stopping=True
        ))
    ])
    model.fit(features, labels)
    return model


def trainModel(classifier, fileName):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model_dir = os.path.dirname(MODEL_PATH)
    joblib.dump(classifier, os.path.join(model_dir, fileName))


def train():
    print("\nĐang huấn luyện KNN")
    trainModel(getKNeighborsClassifier(), "knn.pkl")
    print("\nĐang huấn luyện Decision Tree")
    trainModel(getDecisionTreeClassifier(), "dt.pkl")
    print("\nĐang huấn luyện MLP")
    trainModel(getMLPCLassifier(), "mlp.pkl")
    print("\nĐã huấn luyện và lưu tất cả các mô hình!")


if __name__ == "__main__":
    train()