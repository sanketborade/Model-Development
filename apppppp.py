import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import shap
import matplotlib.pyplot as plt

# Define a function to create pipelines
def create_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

# Define models with default hyperparameters
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

# Set random seed for reproducibility
np.random.seed(42)

# Streamlit interface
st.title("Predictive Model Application")

# Upload scored data CSV
st.header("Upload Scored Data CSV")
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the scored data
        data = pd.read_csv(uploaded_file)
        
        if 'Anomaly_Label' not in data.columns:
            st.error("CSV must contain 'Anomaly_Label' column.")
        else:
            # Convert Anomaly_Label from -1 and 1 to 0 and 1
            data['Anomaly_Label'] = data['Anomaly_Label'].replace({-1: 0, 1: 1})
            
            # Separate features and target
            X = data.drop(columns=['Anomaly_Label'])
            y = data['Anomaly_Label']
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            results = []
            
            for model_name, model in models.items():
                pipeline = create_pipeline(model)
                
                # Apply cross-validation
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
                mean_cv_accuracy = np.mean(cv_scores)
                
                # Train the model
                pipeline.fit(X_train, y_train)
                
                # Save the trained model
                with open(f'{model_name}_model.pkl', 'wb') as f:
                    pickle.dump(pipeline, f)
                
                # Predictions and probabilities
                y_pred = pipeline.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                
                # Collect results
                results.append({
                    'Model': model_name,
                    'Mean CV Accuracy': mean_cv_accuracy,
                    'Test Accuracy': accuracy
                })
            
            # Display results in a table
            results_df = pd.DataFrame(results)
            st.subheader("Model Performance Comparison")
            st.write(results_df)
            
            # Calculate SHAP values for the best model (highest accuracy)
            best_model_name = results_df.loc[results_df['Test Accuracy'].idxmax()]['Model']
            best_model = models[best_model_name]
            best_pipeline = create_pipeline(best_model)
            best_pipeline.fit(X_train, y_train)
            
            st.subheader(f"SHAP Values for {best_model_name}")
            explainer = shap.Explainer(best_pipeline.named_steps['classifier'], X_train)
            shap_values = explainer(X_test)
            
            # Summary plot
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.summary_plot(shap_values, X_test, plot_type="bar")
            st.pyplot(bbox_inches='tight')
            
            # Feature importance (for tree-based models)
            if best_model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
                st.subheader(f"Feature Importance for {best_model_name}")
                importance = best_pipeline.named_steps['classifier'].feature_importances_
                importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
                importance_df = importance_df.sort_values(by='Importance', ascending=False)
                st.bar_chart(importance_df.set_index('Feature'))
            
            # Store the results for download
            result_data = data.copy()
            result_data['Predicted_Label'] = best_pipeline.predict(X)
            
            # Count normal points and outliers after prediction
            normal_count_pred = (result_data['Predicted_Label'] == 0).sum()
            outlier_count_pred = (result_data['Predicted_Label'] == 1).sum()
            
            st.subheader("Counts of Normal Points and Outliers After Prediction")
            st.write(f"Normal Points: {normal_count_pred}")
            st.write(f"Outliers: {outlier_count_pred}")
            
            st.subheader("Scored Data with Predictions")
            st.write(result_data.head())

            result_csv = result_data.to_csv(index=False)
            
            st.download_button(
                label="Download Scored Data with Predictions as CSV",
                data=result_csv,
                file_name='scored_data_with_predictions.csv',
                mime='text/csv'
            )
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a CSV file to proceed.")
