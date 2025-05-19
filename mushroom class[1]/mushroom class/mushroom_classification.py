import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv(r"Data\Original Data\mushrooms.csv")
df.isnull().sum()

label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)

# Based on the feature importance analysis, we selected the top 8 attributes
# that showed the highest contribution to the model's predictive performance
# for use in the training process.

top_features = ['odor', 'spore-print-color', 'gill-color', 'gill-size', 'stalk-root', 'ring-type', 'population', 'stalk-surface-below-ring']
df = df[top_features + ['class']]

X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "NaiveBayes": GaussianNB(),
    "ANN": MLPClassifier(max_iter=1000, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

accuracies = {}

for name, model in models.items():
    print(f"\n** Training {name} **")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    pd.DataFrame({"Prediction": y_pred}).to_csv(f"prediction_{name}.csv", index=False)

    print(classification_report(y_test, y_pred))

    # Accuracy for bar chart
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

    # Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoders['class'].classes_,
                yticklabels=label_encoders['class'].classes_)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name}.png')
    plt.close()

# Accuracy comparison bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.show()

scores = cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=5)

print("Cross-validation accuracy:", scores.mean())
# After applying cross-validation on the selected features, the average accuracy (~90%) confirms that
# the models are not overfitting. Despite high performance on the train/test split, cross-validation
# shows consistent generalization performance across different data folds.
