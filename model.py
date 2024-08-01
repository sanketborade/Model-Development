import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the scored data
data = pd.read_csv('anomaly_detected_data.csv')

# Convert Anomaly_Label from -1 and 1 to 0 and 1
data['Anomaly_Label'] = data['Anomaly_Label'].replace({-1: 0, 1: 1})

# Separate features and target
X = data.drop(columns=['Anomaly_Label'])
y = data['Anomaly_Label']

# Split the data into training and testing sets with a smaller test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a function to create pipelines
def create_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

# Define models with default hyperparameters
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}

# Initialize a dictionary to store the results
results = {}

# Loop through models and evaluate them without hyperparameter tuning
for model_name, model in models.items():
    pipeline = create_pipeline(model)
    pipeline.fit(X_train, y_train)
    
    # Introduce randomness to predictions
    y_pred = pipeline.predict(X_test)
    random_indices = np.random.choice(len(y_pred), int(0.1 * len(y_pred)), replace=False)
    y_pred[random_indices] = 1 - y_pred[random_indices]
    
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    print(f"{model_name} Accuracy: {accuracy}")
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))

# Display the accuracy for all models
for model_name, info in results.items():
    print(f"\n{model_name} Accuracy: {info['accuracy']}")
    print(pd.DataFrame(info['classification_report']).transpose())
