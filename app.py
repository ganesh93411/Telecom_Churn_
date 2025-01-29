import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

# Add custom CSS to style the top border and text
st.markdown(
    """
    <style>
    .top-border {
        background-color: purple;   /* Border color */
        color: white;  /* Text color */
        font-size: 30px;  /* Text size - increase for larger text */
        font-family: 'Arial', sans-serif;  /* Font style - change to your preferred font */
        font-weight: bold;  /* Text weight */
        padding: 20px 0;  /* Increase padding to make the border taller */
        text-align: center;  /* Center the text horizontally */
        width: 100%;  /* Full width of the page */
    }
    </style>
    """, unsafe_allow_html=True
)
# Add the top border with text
st.markdown('<div class="top-border"> Telecom Churn Prediction App </div>', unsafe_allow_html=True)

# Title of the Streamlit app
st.title("Churn Prediction Model")

if "model" not in st.session_state:
    st.session_state.model = None

# Display the image at the top with a fixed height and width (rectangular shape)
st.image("https://miro.medium.com/v2/resize:fit:795/0*8Iu_eymr6eR-YuQw", width=800)

st.write("This application explores the Telecom Churn Dataset And Uses Machine Learning Models for Prediction.")


# Upload Dataset
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Read the uploaded file
    if uploaded_file.name.endswith('xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    st.write("Dataset Preview")
    st.write(df.head())

    # Show basic information and describe the dataset
    if st.checkbox("Show Dataset Description"):
        st.write(df.describe())
    
    if st.checkbox("Show Visualizations"):
            st.markdown('<h4 style="color: navy;">Numerical Feature Distribution: </h4>', unsafe_allow_html=True)
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            for column in numeric_columns:
                fig, ax = plt.subplots()
                sns.histplot(df[column], kde=True, ax=ax)
                st.write(f"Distribution of {column}")
                st.pyplot(fig)

            st.markdown('<h4 style="color: navy;">Categorical Feature Count: </h4>', unsafe_allow_html=True)
            categorical_columns = df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                fig, ax = plt.subplots()
                sns.countplot(data=df, x=column, palette="turbo")
                st.write(f"Count Plot for {column}")
                st.pyplot(fig)

            st.markdown('<h4 style="color: navy;">Correlation Heatmap: </h4>', unsafe_allow_html=True)
            if len(numeric_columns) > 1:
                fig, ax = plt.subplots()
                sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

    # Data Preprocessing
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df['day.charge'] = df['day.charge'].astype(float)
    df['eve.mins'] = df['eve.mins'].astype(float)

    df['day.charge'] = df['day.charge'].fillna(df['day.charge'].mean())
    df['eve.mins'] = df['eve.mins'].fillna(df['eve.mins'].mean())

    target_columns = ['churn', 'voice.plan', 'intl.plan']
    for column in target_columns:
        if column in df.columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])

    # One Hot Encoding
    ohe = pd.get_dummies(df[['voice.plan', 'intl.plan', 'area.code']], drop_first=True)
    df = pd.concat([df, ohe], axis=1)
    df.drop(['voice.plan', 'intl.plan', 'area.code', 'state'], axis=1, inplace=True)

    # Train-test split
    X = df.drop(columns='churn')
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=104)

    # Scaling the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Select model
    model_name = st.sidebar.selectbox("Select a Model", 
                                      ["Logistic Regression", "Random Forest", "Naive Bayes", "Decision Tree"])

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=104)
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=104)

    # Train the selected model
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    st.write(f"Accuracy: {accuracy*100:.2f}")
    st.write(f"ROC-AUC Score: {roc_auc*100:.2f}")

    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_title(f"ROC Curve for {model_name}")
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc='lower right')
    st.pyplot(fig_roc)

    # Hyperparameter Tuning
    if st.checkbox("Hyperparameter Tuning"):
        param_grid = None
        if model_name == "Logistic Regression":
            param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
        elif model_name == "Random Forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif model_name == "Naive Bayes":
            param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}
        elif model_name == "Decision Tree":
            param_grid = {
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        st.write("Best Parameters:", grid_search.best_params_)

        best_model.fit(X_train, y_train)
        y_pred_best = best_model.predict(X_test)
        accuracy_best = accuracy_score(y_test, y_pred_best)
        st.write(f"Accuracy after Hyperparameter Tuning: {accuracy_best*100:.2f}")

    # Cross-Validation
    if st.checkbox("Cross-Validation"):
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        st.write(f"Cross-validation Accuracy: {cv_scores.mean()*100:.2f}")
    
    if 'model_performance' not in st.session_state:
        st.session_state['model_performance'] = {}

# Save metrics after evaluating a model
    st.session_state['model_performance'][model_name] = {
        'accuracy': accuracy,
        'precision': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision'],
        'recall': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall'],
        'f1_score': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score'],
        'roc_auc': roc_auc
        }
    
    # Check if model and features are set in session state
    # Save the trained model and features to session state
    st.session_state['model'] = model
    st.session_state['features'] = X.columns.tolist()

# Sidebar for prediction
    if 'model' in st.session_state and 'features' in st.session_state:
        st.sidebar.subheader("Make a Prediction")
        user_input = []
        for feature in st.session_state['features']:
            value = st.sidebar.number_input(f"Enter value for {feature}", value=0.0, step=0.1)
            user_input.append(value)

        user_input = np.array(user_input).reshape(1, -1)

    if st.sidebar.button("Predict"):
        prediction = st.session_state['model'].predict(user_input)
        prediction_proba = st.session_state['model'].predict_proba(user_input)[0]
        st.sidebar.write("Prediction Result:", "Loyal" if prediction[0] == 1 else "Churn")
        st.sidebar.write("Prediction Probability:", f"{prediction_proba[prediction[0]]*100:.2f}")
    else:
        st.sidebar.warning("Model not trained yet or features not available!")

    if st.checkbox("Compare Models"):
        if st.session_state['model_performance']:
               # Create a performance summary without confusion_matrix and roc_curve for comparison
               comparison_metrics = {
                    model: {
                         key: (value if not isinstance(value, (np.ndarray, tuple)) else "N/A")
                         for key, value in metrics.items()
                    }
                    for model, metrics in st.session_state['model_performance'].items()
               }
               # Convert metrics dictionary to DataFrame
               performance_df = pd.DataFrame(comparison_metrics).T
               performance_df.reset_index(inplace=True)
               performance_df.rename(columns={'index': 'Model'}, inplace=True)

               st.subheader("Model Comparison Table")
               st.write(performance_df)

               st.subheader("Comparison Bar Chart")
               metric_to_compare = st.selectbox("Select Metric for Comparison", ['accuracy', 'precision', 'recall', 'f1_score'])

               fig, ax = plt.subplots()
               sns.barplot(x='Model', y=metric_to_compare, data=performance_df, ax=ax, palette="viridis")
               ax.set_ylabel(metric_to_compare.capitalize())
               ax.set_title(f"Model Comparison: {metric_to_compare.capitalize()}")
               st.pyplot(fig)

               # Identify the best model based on the selected metric
               best_model_row = performance_df.loc[performance_df[metric_to_compare].idxmax()]
               best_model_name = best_model_row['Model']
               best_model_value = best_model_row[metric_to_compare]

               st.success(f"The best-performing model based on **{metric_to_compare}** is **{best_model_name}** with a score of **{best_model_value*100:.2f}**.")

               # Display ROC AUC scores if available
               roc_scores = {model: metrics.get('roc_auc', "N/A") for model, metrics in st.session_state['model_performance'].items()}
               st.subheader("ROC AUC Scores")
               for model, score in roc_scores.items():
                    st.write(f"{model}: {score}")


