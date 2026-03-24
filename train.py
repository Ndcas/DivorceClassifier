import os
import statistics
import joblib
import pandas
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")
K_FOLDS = int(os.getenv("K_FOLDS"))

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
        ("classifier", KNeighborsClassifier())
    ])
    param_grid = {'classifier__n_neighbors': list(range(1, 59, 2))}
    grid = GridSearchCV(pipeline, param_grid, cv=K_FOLDS, scoring="f1", n_jobs=-1)
    grid.fit(features, labels)
    best_k = grid.best_params_['classifier__n_neighbors']
    return KNeighborsClassifier(n_neighbors=best_k)


def getDecisionTreeClassifier():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=1))
    ])
    param_grid = {'classifier__max_depth': list(range(1, 31))}
    grid = GridSearchCV(pipeline, param_grid, cv=K_FOLDS, scoring="f1", n_jobs=-1)
    grid.fit(features, labels)
    best_depth = grid.best_params_['classifier__max_depth']
    return DecisionTreeClassifier(max_depth=best_depth, random_state=1)


def getSVC():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", SVC())
    ])
    param_grid = {'classifier__C': [0.001 * pow(2, i) for i in range(30)]}
    grid = GridSearchCV(pipeline, param_grid, cv=K_FOLDS, scoring="f1", n_jobs=-1)
    grid.fit(features, labels)
    best_c = grid.best_params_['classifier__C']
    return SVC(C=best_c)


def getMLPCLassifier():
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        random_state=1,
        max_iter=300,
        early_stopping=True
    )


def train():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model_dir = os.path.dirname(MODEL_PATH)
    print("Đang lấy mô hình KNN")
    knn_est = getKNeighborsClassifier()
    knn_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", knn_est)])
    print("Đang huấn luyện KNN")
    knn_pipeline.fit(features, labels)
    joblib.dump(knn_pipeline, os.path.join(model_dir, "knn.pkl"))
    print("Đang lấy mô hình Decision Tree")
    dt_est = getDecisionTreeClassifier()
    dt_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", dt_est)])
    print("Đang huấn luyện Decision Tree")
    dt_pipeline.fit(features, labels)
    joblib.dump(dt_pipeline, os.path.join(model_dir, "dt.pkl"))
    print("Đang lấy mô hình SVC")
    sv_est = getSVC()
    sv_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", sv_est)])
    print("Đang huấn luyện SVC")
    sv_pipeline.fit(features, labels)
    joblib.dump(sv_pipeline, os.path.join(model_dir, "svc.pkl"))
    print("Đang lấy mô hình MLP")
    mlp_est = getMLPCLassifier()
    mlp_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", mlp_est)])
    print("Đang huấn luyện MLP")
    mlp_pipeline.fit(features, labels)
    joblib.dump(mlp_pipeline, os.path.join(model_dir, "mlp.pkl"))
    print("Đã huấn luyện và lưu tất cả các mô hình!")


if __name__ == "__main__":
    train()
