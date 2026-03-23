import os
import statistics
import joblib
import pandas
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
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
    scores = []
    while len(scores) < 300:
        k = 1 + len(scores) * 2
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier(n_neighbors=k))
        ])
        score = statistics.mean(cross_val_score(model, features, labels, cv=K_FOLDS, scoring="f1"))
        scores.append(score)
        if len(scores) >= 5 and scores[-1] <= scores[-2] <= scores[-3] <= scores[-4] <= scores[-5]:
            break
    bestK = scores.index(max(scores)) * 2 + 1
    return KNeighborsClassifier(n_neighbors=bestK)


def getDecisionTreeClassifier():
    scores = []
    while len(scores) < 300:
        depth = len(scores) + 1
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(max_depth=depth, random_state=1))
        ])
        score = statistics.mean(cross_val_score(model, features, labels, cv=K_FOLDS, scoring="f1"))
        scores.append(score)
        if len(scores) >= 5 and scores[-1] <= scores[-2] <= scores[-3] <= scores[-4] <= scores[-5]:
            break
    bestDepth = scores.index(max(scores)) + 1
    return DecisionTreeClassifier(max_depth=bestDepth, random_state=1)


def getSVC():
    scores = []
    while len(scores) < 300:
        c = pow(2, len(scores))
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", SVC(C=c))
        ])
        score = statistics.mean(cross_val_score(model, features, labels, cv=K_FOLDS, scoring="f1"))
        scores.append(score)
        if len(scores) >= 5 and scores[-1] <= scores[-2] <= scores[-3] <= scores[-4] <= scores[-5]:
            break
    bestC = pow(2, scores.index(max(scores)))
    return SVC(C=bestC)


def getMLPCLassifier():
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        random_state=1,
        max_iter=300,
        early_stopping=True
    )


def train():
    print("Đang lấy mô hình KNN")
    kNeighbor = getKNeighborsClassifier()
    print("Đang lấy mô hình Decision Tree")
    decisionTree = getDecisionTreeClassifier()
    print("Đang lấy mô hình SVC")
    sv = getSVC()
    print("Đang lấy mô hình MLP")
    mlp = getMLPCLassifier()
    voting = VotingClassifier(
        estimators=[("kNeighbor", kNeighbor), ("decisionTree", decisionTree), ("sv", sv), ("mlp", mlp)],
        voting="soft"
    )
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", voting)
    ])
    print("Đang huấn luyện mô hình")
    model.fit(features, labels)
    print("Đang lưu mô hình")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Đã huấn luyện mô hình, lưu tại {MODEL_PATH}")

if __name__ == "__main__":
    train()
