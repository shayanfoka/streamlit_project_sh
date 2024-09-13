import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import plotly.express as px
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures ,LabelEncoder
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import emoji
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report)
from sklearn.linear_model import LogisticRegression, LinearRegression , Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor , GradientBoostingClassifier , GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import nltk
from nltk.corpus import wordnet , stopwords
from nltk.tokenize import word_tokenize 
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA



def delete_columns(df):
    """
    Function to allow users to remove specific columns from the dataset.
    Users can select multiple columns for removal. This step happens before other preprocessing tasks.
    """
    st.sidebar.subheader("Step 1: Delete Columns")
    columns_to_delete = st.sidebar.multiselect("Select columns to remove", df.columns)
    
    if columns_to_delete:
        df = df.drop(columns=columns_to_delete)
        st.write(f"Removed columns: {columns_to_delete}")
        st.dataframe(df)
    
    return df

def handle_missing_values(df):
    """
    Function to handle missing values in the dataset.
    Users can choose to drop rows, fill with mean, median, mode, or apply forward/backward fill.
    """
    st.sidebar.subheader("Missing Value Handling")
    missing_option = st.sidebar.selectbox(
        "Choose how to handle missing values",
        ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode", "Forward Fill", "Backward Fill"]
    )

    if missing_option == "Drop Rows":
        df = df.dropna()
    elif missing_option == "Fill with Mean":
        df = df.fillna(df.mean())
    elif missing_option == "Fill with Median":
        df = df.fillna(df.median())
    elif missing_option == "Fill with Mode":
        df = df.fillna(df.mode().iloc[0])
    elif missing_option == "Forward Fill":
        df = df.fillna(method="ffill")
    elif missing_option == "Backward Fill":
        df = df.fillna(method="bfill")

    return df


def apply_feature_scaling(df):
    """
    Function to scale numerical features in the dataset.
    Users can choose between StandardScaler, MinMaxScaler, and RobustScaler.
    """
    st.sidebar.subheader("Feature Scaling")
    scaling_method = st.sidebar.selectbox(
        "Choose a scaling method", 
        ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
    )

    if scaling_method != "None":
        scaler = None
        if scaling_method == "StandardScaler":
            scaler = StandardScaler()
        elif scaling_method == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scaling_method == "RobustScaler":
            scaler = RobustScaler()

        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[num_cols] = scaler.fit_transform(df[num_cols])
        st.write("Data after Scaling:")
        st.dataframe(df)

    return df

def feature_engineering(df):
    """
    Function to allow users to create polynomial features in the dataset.
    Users can select polynomial feature creation.

    **Polynomial features**: They can be useful when the relationship 
    between the features and the target variable is non-linear.
    """
    st.sidebar.subheader("Feature Engineering")
    polynomial = st.sidebar.checkbox("Generate Polynomial Features")

    if  polynomial:
        poly = PolynomialFeatures(degree=2,)
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df_poly = poly.fit_transform(df[num_cols])
        df_poly = pd.DataFrame(df_poly, columns=poly.get_feature_names_out(num_cols))
        st.write("Data with new interaction/polynomial features:")
        st.dataframe(df_poly)
    
    return df



def handle_categorical_data(df):
    """
    Function to handle categorical data .
    Users can select the handling method for all categorical columns at once.
    Then, based on the method, users select specific columns to apply transformations.
    """
    st.sidebar.subheader("Step 2: Categorical Data Handling")
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if cat_columns:
        # Allow users to select which columns to encode and which method to use
        encoding_method = st.sidebar.selectbox(
            "Choose encoding methods",
            ["None", "One-Hot Encoding", "Label Encoding", "Both"]
        )

        if encoding_method == "Both":
            one_hot_cols = st.sidebar.multiselect("Select columns for One-Hot Encoding", cat_columns)
            label_encode_cols = st.sidebar.multiselect("Select columns for Label Encoding", cat_columns)

            if one_hot_cols:
                drop_first = st.sidebar.checkbox("Drop first dummy variable?", value=True)
                df = pd.get_dummies(df, columns=one_hot_cols, drop_first=drop_first)
                st.write(f"Applied One-Hot Encoding to columns: {one_hot_cols}")

            if label_encode_cols:
                le = LabelEncoder()
                for col in label_encode_cols:
                    df[col] = le.fit_transform(df[col])
                st.write(f"Applied Label Encoding to columns: {label_encode_cols}")

            st.dataframe(df)

        elif encoding_method == "One-Hot Encoding":
            # Option to choose whether or not to drop the first dummy variable
            drop_first = st.sidebar.checkbox("Drop first dummy variable?", value=True)
            # Ask the user which categorical columns they want to encode
            columns_to_encode = st.sidebar.multiselect("Select categorical columns for One-Hot Encoding", cat_columns)
            if columns_to_encode:
                df = pd.get_dummies(df, columns=columns_to_encode, drop_first=drop_first)
                st.write(f"Applied One-Hot Encoding to columns: {columns_to_encode}")
                st.dataframe(df)

        elif encoding_method == "Label Encoding":
            # Ask the user which categorical columns they want to encode
            columns_to_encode = st.sidebar.multiselect("Select categorical columns for Label Encoding", cat_columns)
            if columns_to_encode:
                le = LabelEncoder()
                for col in columns_to_encode:
                    df[col] = le.fit_transform(df[col])
                st.write(f"Applied Label Encoding to columns: {columns_to_encode}")
                st.dataframe(df)

    else:
        st.write("No categorical columns found for encoding.")
    
    return df



def create_visualizations(df):
    """
    Function to generate different types of visualizations using Plotly and Seaborn.
    
    Parameters:
    df: DataFrame
        Input DataFrame for generating visualizations.
    """

    if len(df.columns) < 2:
        st.warning("The dataset must have at least two numeric columns for visualizations.")
        return

    # Column selection for scatter plot
    st.write("Scatter Plot (Seaborn)")
    x_axis = st.selectbox("Select X-axis column for Scatter Plot", df.columns)
    y_axis = st.selectbox("Select Y-axis column for Scatter Plot", df.columns)
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
    st.pyplot(fig)

    # Column selection for bar plot
    st.write("Bar Plot (Plotly)")
    x_axis_bar = st.selectbox("Select X-axis column for Bar Plot", df.columns, key="bar_x")
    y_axis_bar = st.selectbox("Select Y-axis column for Bar Plot", df.columns, key="bar_y")
    
    fig_bar = px.bar(df, x=x_axis_bar, y=y_axis_bar)
    st.plotly_chart(fig_bar)

    # Column selection for 3D scatter plot
    st.write("3D Scatter Plot (Plotly)")
    if len(df.columns) >= 3:  # Ensure there are enough numeric columns for a 3D scatter plot
        x_axis_3d = st.selectbox("Select X-axis column for 3D Scatter Plot", df.columns, key="3d_x")
        y_axis_3d = st.selectbox("Select Y-axis column for 3D Scatter Plot", df.columns, key="3d_y")
        z_axis_3d = st.selectbox("Select Z-axis column for 3D Scatter Plot", df.columns, key="3d_z")
        target = st.selectbox("Please select the target label", df.columns, key="color")
    
        
            # Create the 3D scatter plot
        fig_3d = px.scatter_3d(df, x=x_axis_3d, y=y_axis_3d, z=z_axis_3d, color=target)
    
    # Display the plot
        st.plotly_chart(fig_3d)

def return_df(file):
    """
    Helper function to load tabular data based on file type.
    """
    name = file.name
    extension = name.split(".")[-1]

    if extension == "csv":
        df = pd.read_csv(file)
    elif extension == "tsv":
        df = pd.read_csv(file, sep="\t")
    elif extension == "xlsx":
        df = pd.read_excel(file)
    elif extension == "xml":
        df = pd.read_xml(file)
    elif extension == "json":
        df = pd.read_json(file)
    else:
        st.error("Unsupported file format")
        return None
    return df




def evaluate_model_cv(pipeline, X, y, cv=5):
    """Helper function to evaluate a pipeline using cross-validation."""
    if hasattr(pipeline.named_steps['model'], 'predict'):
        scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    else:
        scoring_metrics = ['neg_mean_squared_error', 'r2']
        
    scores = {}
    
    for metric in scoring_metrics:
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=metric)
        scores[metric] = np.mean(cv_scores)  # Average of cross-validation scores
    
    return scores

def apply_models_pipeline(X, y, test_size=0.2, random_state=42, cv=5, search_type='grid', models=[], param_grid_size=10, n_iter=10, custom_params={}):
    """
    Function to apply selected machine learning pipelines with cross-validation and hyperparameter search.
    
    Parameters:
    - X: Features (input data)
    - y: Labels (target data)
    - test_size: Proportion of the dataset to include in the test split
    - random_state: Seed for reproducibility
    - cv: Number of cross-validation folds
    - search_type: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
    - models: List of model names to apply
    - param_grid_size: Number of parameters to sample in GridSearchCV (not applicable for RandomizedSearchCV)
    - n_iter: Number of iterations for RandomizedSearchCV
    - custom_params: Dictionary of custom hyperparameters for each model
    
    Returns:
    A dictionary with the evaluation results for the selected pipelines and hyperparameter optimization.
    """
    
    results = {}

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    for model_name in models:
        # Ensure the selected model exists
        if model_name not in pipelines:
            st.error(f"Model '{model_name}' is not available. Please choose a valid model.")
            continue
        
        # Define pipeline and parameter grid for the selected model
        pipeline = pipelines[model_name]
        param_grid = param_grids.get(model_name, {})
        
        # Override param_grid with custom_params if provided
        if model_name in custom_params:
            param_grid.update(custom_params[model_name])
        
        if search_type == 'grid':
            # Limit the parameter grid size
            param_grid_limited = {k: v[:param_grid_size] for k, v in param_grid.items()}
            search = GridSearchCV(pipeline, param_grid_limited, cv=cv, scoring='accuracy' if hasattr(pipeline.named_steps['model'], 'predict') else 'neg_mean_squared_error', n_jobs=-1)
        elif search_type == 'random':
            search = RandomizedSearchCV(pipeline, param_grid, n_iter=n_iter, cv=cv, scoring='accuracy' if hasattr(pipeline.named_steps['model'], 'predict') else 'neg_mean_squared_error', n_jobs=-1, random_state=random_state)
        
        # Fit the search
        search.fit(X_train, y_train)
        
        # Get the best pipeline from the search
        best_pipeline = search.best_estimator_
        
        # Evaluate the best model on the test set
        test_scores = evaluate_model_cv(best_pipeline, X_test, y_test, cv=cv)
        
        # Cross-validation on the entire dataset with the best model
        cv_scores = evaluate_model_cv(best_pipeline, X, y, cv=cv)
        
        # Store the best hyperparameters and the results
        results[model_name] = {
            'best_params': search.best_params_,
            'test_set_scores': test_scores,
            'cross_val_scores': cv_scores
        }
    
    return results

# Define models and hyperparameters
pipelines = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ]),
    'Decision Tree Classifier': Pipeline([
        ('model', DecisionTreeClassifier())
    ]),
    'Random Forest Classifier': Pipeline([
        ('model', RandomForestClassifier())
    ]),
    'Support Vector Classifier': Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC())
    ]),
    'Naive Bayes': Pipeline([
        ('scaler', StandardScaler()),
        ('model', GaussianNB())
    ]),
    'K-Nearest Neighbors': Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier())
    ]),
    'Gradient Boosting Classifier': Pipeline([
        ('model', GradientBoostingClassifier())
    ]),
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    'Decision Tree Regressor': Pipeline([
        ('model', DecisionTreeRegressor())
    ]),
    'Random Forest Regressor': Pipeline([
        ('model', RandomForestRegressor())
    ]),
    'Support Vector Regressor': Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR())
    ]),
    'Lasso Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ]),
    'Ridge Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge())
    ]),
    'Gradient Boosting Regressor': Pipeline([
        ('model', GradientBoostingRegressor())
    ])
}

param_grids = {
    'Logistic Regression': {
        'model__C': np.logspace(-3, 3, 7),
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear']
    },
    'Decision Tree Classifier': {
        'model__max_depth': [3, 5, 10, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'Random Forest Classifier': {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'Support Vector Classifier': {
        'model__C': np.logspace(-3, 3, 7),
        'model__kernel': ['linear', 'rbf', 'poly'],
        'model__gamma': ['scale', 'auto']
    },
    'Naive Bayes': {},  # No hyperparameters for Naive Bayes in this example
    'K-Nearest Neighbors': {
        'model__n_neighbors': [3, 5, 7, 10],
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2]
    },
    'Gradient Boosting Classifier': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    },
    'Linear Regression': {},  # No hyperparameters for Linear Regression
    'Decision Tree Regressor': {
        'model__max_depth': [3, 5, 10, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'Random Forest Regressor': {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'Support Vector Regressor': {
        'model__C': np.logspace(-3, 3, 7),
        'model__kernel': ['linear', 'rbf', 'poly'],
        'model__gamma': ['scale', 'auto']
    },
    'Lasso Regression': {
        'model__alpha': np.logspace(-3, 3, 7)
    },
    'Ridge Regression': {
        'model__alpha': np.logspace(-3, 3, 7)
    },
    'Gradient Boosting Regressor': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    }
}


def load_tabular_data():
    """
    Function to handle file uploads and load tabular data in different formats.
    Provides preprocessing options and visualization features.
    """
    f = st.file_uploader("Upload Your Dataset 📤", type=["csv", "tsv", "xlsx", "xml", "json"])

    if f:
        df = return_df(f)
        if df is not None:
            st.success("File Uploaded!")
            st.dataframe(df)

            # Step 1: Column Removal
            df = delete_columns(df)

            # Step 2: Categorical Data Handling
            df = handle_categorical_data(df)
            
            # Step 3: Data Preprocessing and Cleaning Options
            df = handle_missing_values(df)
            df = apply_feature_scaling(df)
            df = feature_engineering(df)

            # Ensure the data is clean before generating EDA report
            df = df.dropna()  # Drop rows with missing values
            df = df.select_dtypes(include=[np.number])  # Ensure only numeric data for profiling

            # Add an expander for Pandas Profiling Report
            with st.expander("Pandas Profiling Report (Click to expand/collapse)"):
                st.subheader("Pandas Profiling Report")
                try:
                    pr = ProfileReport(df, explorative=True)
                    pr_html = pr.to_html()
                    html(pr_html, height=800)
                except ValueError as e:
                    st.error(f"An error occurred while generating the profile report: {e}")

            # Expander for visualizations
            with st.expander("Click to explore Visualizations"):
                st.subheader("Visualizations")
                create_visualizations(df)

            # Add an expander for machine learning
            with st.expander("Machine Learning Options (Click to expand/collapse)"):
                st.subheader("Machine Learning")

                # Select target column
                target = st.selectbox("Select target", df.columns.tolist())

                if target:
                    X = df.drop(columns=[target])
                    y = df[target]

                    # Limit the number of models to 3
                    selected_models = st.multiselect("Select models (up to 3)", options=list(pipelines.keys()), max_selections=3)

                    if selected_models:
                        st.write("### Configure Hyperparameters")

                        # Dictionary to store custom parameters for each model
                        custom_params = {}

                        # Collect hyperparameters for each selected model
                        for model_name in selected_models:
                            st.write(f"**{model_name}**")

                            params = param_grids.get(model_name, {})
                            model_params = {}
                            
                            for param, values in params.items():
                                # Provide input widgets based on parameter type
                                if isinstance(values[0], (int, float)):
                                    selected_values = st.multiselect(f"Choose values for {param}", options=values, default=values[:1])
                                elif isinstance(values[0], str):
                                    selected_values = st.multiselect(f"Choose values for {param}", options=values, default=[values[0]])
                                else:
                                    selected_values = st.selectbox(f"Choose value for {param}", options=values, index=0)
                                
                                model_params[param] = selected_values
                            
                            custom_params[model_name] = model_params
                        
                        # Select search type
                        search_type = st.selectbox("Select search type", options=['grid', 'random'])

                        # Interactive widgets for parameter grid size and random search iterations
                        param_grid_size = st.slider("Grid Search: Number of parameters to sample", min_value=1, max_value=20, value=10) if search_type == 'grid' else None
                        n_iter = st.slider("Random Search: Number of iterations", min_value=1, max_value=50, value=10) if search_type == 'random' else None

                        # Apply models and show results
                        if st.button("Run Models"):
                            with st.spinner("Running models..."):
                                results = apply_models_pipeline(
                                    X, y,
                                    search_type=search_type,
                                    models=selected_models,
                                    param_grid_size=param_grid_size,
                                    n_iter=n_iter,
                                    custom_params=custom_params
                                )
                                for model_name, result in results.items():
                                    st.write(f"### {model_name}")
                                    st.write(f"**Best parameters:** {result['best_params']}")
                                    st.write(f"**Test set scores:** {result['test_set_scores']}")
                                    st.write(f"**Cross-validation scores:** {result['cross_val_scores']}")
                                    st.write("---")



##################################
##################################
##################################
##################################


def tokenize_text(text):
    return word_tokenize(text)

def lowercase_text(text: str) -> str:
    return text.lower()

def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text: str, language: str = 'english') -> str:
    stop_words = set(stopwords.words(language))
    words = text.split()
    return ' '.join([word for word in words if word.lower() not in stop_words])

def remove_numbers(text: str) -> str:
    return re.sub(r'\d+', '', text)

def stem_text(text: str) -> str:
    stemmer = PorterStemmer()
    words = text.split()
    return ' '.join([stemmer.stem(word) for word in words])

def remove_whitespace(text: str) -> str:
    return text.strip()

def remove_emoji(text: str) -> str:
    return emoji.replace_emoji(text, replace='')

# Convert text into DataFrame
def text_to_dataframe(text: str) -> pd.DataFrame:
    sentences = text.split('\n')  # Split by newlines or customize the delimiter
    df = pd.DataFrame(sentences, columns=['Text'])
    return df

# Preprocess text based on user selection
def preprocess_text(text: str, options: dict) -> str:
    if options.get("lowercase"):
        text = lowercase_text(text)
    if options.get("remove_punctuation"):
        text = remove_punctuation(text)
    if options.get("remove_numbers"):
        text = remove_numbers(text)
    if options.get("remove_stopwords"):
        text = remove_stopwords(text)
    if options.get("stem_text"):
        text = stem_text(text)
    if options.get("remove_emoji"):
        text = remove_emoji(text)
    if options.get("remove_whitespace"):
        text = remove_whitespace(text)
    
        # Tokenize text
    tokens = tokenize_text(text)
    
    # Remove stopwords
    if options.get("remove_stopwords"):
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    # Stem words
    if options.get("stem_text"):
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)


# Preprocess the DataFrame in a vectorized manner
def preprocess_text_df(df: pd.DataFrame, options: dict) -> pd.DataFrame:
    # Apply preprocessing to each row using pandas apply function
    df['Text'] = df['Text'].apply(lambda x: preprocess_text(x, options))
    return df

def plot_wordcloud(text: str):
    wordcloud = WordCloud(background_color="white").generate(text)
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Augmentation functions using nlpaug library
def aug_synonym_replacement(text: str) -> str:
    aug = naw.SynonymAug(aug_src='wordnet')  
    return aug.augment(text)

def aug_character_swap(text: str) -> str:
    aug = nac.RandomCharAug(action="swap")  
    return aug.augment(text)

def augment_text_df(df: pd.DataFrame, options: dict, concat: bool = False) -> pd.DataFrame:
    def apply_augmentations(row: pd.Series) -> pd.Series:
        text = row['Text']
        augmented_texts = {}

        if 'synonym_replacement' in options:
            augmented_texts['Synonym Replacement'] = aug_synonym_replacement(text)
        if 'character_swap' in options:
            augmented_texts['Character Swap'] = aug_character_swap(text)
        
        return pd.Series(augmented_texts)
    
    # Apply selected augmentations to each row using pandas apply function
    augmented_df = df.apply(apply_augmentations, axis=1)
    
    if concat:
        # Combine all text columns into one column if concatenation is enabled
        combined_df = pd.concat([df, augmented_df], axis=1)
        combined_text = combined_df.apply(lambda row: ' '.join([str(row[col]) for col in combined_df.columns if 'Text' in col]), axis=1)
        final_df = pd.DataFrame(combined_text, columns=['Text'])
    else:
        # Return the DataFrame with original and augmented text columns
        final_df = pd.concat([df, augmented_df], axis=1)
    
    return final_df


# Augmentation options UI
def augmentation_options_ui() -> dict:
    with st.expander("🛠️ Augmentation Options"):
        augment_options = {
            'character_swap': st.checkbox("🔄 Swap random characters", key="character_swap"),
            'synonym_replacement': st.checkbox("📖 Replace words with synonyms", key="synonym_replacement"),
        }
        # Filter options to only those selected by the user
        selected_options = {k: v for k, v in augment_options.items() if v}
        return selected_options

def compute_tfidf(df: pd.DataFrame, text_column: str):
    tf_idf = TfidfVectorizer()
    tfidf_matrix = tf_idf.fit_transform(df[text_column])
    feature_names = tf_idf.get_feature_names_out()
    return tfidf_matrix, feature_names

def perform_clustering(embeddings_df: pd.DataFrame, num_clusters: int):
    X = embeddings_df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0)
    clusters = kmeans.fit_predict(X_scaled)
    embeddings_df['Cluster'] = clusters
    return embeddings_df, kmeans

def perform_tsne(embeddings_df: pd.DataFrame, perplexity: int = 30, learning_rate: float = 200.0, apply_pca: bool = True, n_pca_components: int = 50):
    X = embeddings_df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if apply_pca:
        # Reduce dimensionality first using PCA
        pca = PCA(n_components=n_pca_components)
        X_pca = pca.fit_transform(X_scaled)
    else:
        X_pca = X_scaled
    
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=0)
    X_tsne = tsne.fit_transform(X_pca)
    tsne_df = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
    tsne_df['Cluster'] = embeddings_df.get('Cluster', 'No Cluster')  # Include cluster labels if available
    return tsne_df

def load_nlp_data():
    uploaded_file = st.file_uploader("📄 Upload a text file", type="txt")

    if uploaded_file is not None:
        try:
            text = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            text = uploaded_file.read().decode("ISO-8859-1", errors="replace")

        st.write(f"### 📜 Original Content Preview\n{text[:300]}...")
        word_count = len(text.split())
        st.write(f"**📝 Word Count: {word_count}**")

        df = text_to_dataframe(text)
        df_original = df.copy()

        with st.expander("⚙️ Preprocessing Options"):
            options = {
                "lowercase": st.checkbox("🔤 Lowercase text", key="lowercase"),
                "remove_punctuation": st.checkbox("✂️ Remove punctuation", key="remove_punctuation"),
                "remove_numbers": st.checkbox("🔢 Remove numbers", key="remove_numbers"),
                "remove_stopwords": st.checkbox("🚫 Remove stopwords", key="remove_stopwords"),
                "stem_text": st.checkbox("🌱 Stem words", key="stem_text"),
                "remove_emoji": st.checkbox("😊 Remove emojis", key="remove_emoji"),
                "remove_whitespace": st.checkbox("🧹 Remove extra whitespace", key="remove_whitespace")
            }

        df_processed = preprocess_text_df(df, options)

        st.subheader("🔍 Original vs Processed Text")
        col1, col2 = st.columns(2)

        with col1:
            st.write("### 📝 Original Text")
            st.dataframe(df_original)

        with col2:
            st.write("### ✨ Processed Text")
            st.dataframe(df_processed)

        st.subheader("☁️ Word Cloud of Processed Text")
        combined_processed_text = " ".join(df_processed['Text'].tolist())
        plot_wordcloud(combined_processed_text)

        augment_options = augmentation_options_ui()
        
        with st.expander("🔄 Concatenate text with augmented"):
            concat_augmented = st.checkbox("📚 Concatenate augmented text columns", key="concat_augmented")

        st.subheader("🔄 Augmented Text DataFrame")
        df_augmented = augment_text_df(df_processed, augment_options, concat=concat_augmented)
        st.dataframe(df_augmented)

        st.write("Computing TF-IDF matrix...")
        tfidf_matrix, feature_names = compute_tfidf(df_processed, 'Text')

        st.subheader("💡 Sentence Embeddings with Pre-trained Models")

        model_names = [
            "all-MiniLM-L6-v2", 
            "paraphrase-MiniLM-L12-v2"
        ]

        selected_model = st.selectbox("Select a pre-trained model to generate embeddings:", model_names)

        st.write(f"Loading model: {selected_model}...")
        model = SentenceTransformer(selected_model)

        st.write("Generating embeddings...")
        embeddings = model.encode(df_processed['Text'].tolist(), convert_to_numpy=True)

        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.columns = [f"dim_{i}" for i in range(embeddings.shape[1])]

        st.write("### Sentence Embeddings")
        st.dataframe(embeddings_df)

        if st.button("Save Embeddings as CSV"):
            embeddings_df.to_csv("embeddings.csv", index=False)
            st.success("Embeddings saved successfully!")

        analysis_option = st.selectbox("Select Analysis Method:", ["None", "K-Means Clustering", "t-SNE Visualization"])

        if analysis_option == "K-Means Clustering":
            num_clusters = st.slider("Select number of clusters:", min_value=2, max_value=10, value=3)
            df_clustered, kmeans = perform_clustering(embeddings_df, num_clusters)
            
            # Visualize clustering results
            st.write("### Clustering Scatter Plot")
            fig, ax = plt.subplots()
            scatter = ax.scatter(df_clustered.iloc[:, 0], df_clustered.iloc[:, 1], c=df_clustered['Cluster'], cmap='viridis')
            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend)
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            st.pyplot(fig)

        elif analysis_option == "t-SNE Visualization":
            perplexity = st.slider("Select perplexity for t-SNE:", min_value=5, max_value=50, value=30)
            learning_rate = st.slider("Select learning rate for t-SNE:", min_value=10, max_value=1000, value=200)
            apply_pca = st.checkbox("Apply PCA before t-SNE", value=True)
            n_pca_components = st.slider("Select number of PCA components:", min_value=10, max_value=100, value=50)
            tsne_df = perform_tsne(embeddings_df, perplexity, learning_rate, apply_pca, n_pca_components)
            
            # Visualize t-SNE results
            st.write("### t-SNE Scatter Plot")
            fig, ax = plt.subplots()
            scatter = ax.scatter(tsne_df['Dim1'], tsne_df['Dim2'], c=tsne_df['Cluster'].astype('category').cat.codes, cmap='viridis')
            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend)
            plt.xlabel('Dim1')
            plt.ylabel('Dim2')
            st.pyplot(fig)


def main():
    st.title("Multi-Modal Dataset Uploader")

    # Sidebar for dataset selection
    dataset_type = st.sidebar.selectbox("Choose the type of dataset you want to process", 
                                        ["Tabular Data", "Text Data "])

    if dataset_type == "Tabular Data":
        st.header("Tabular Data Upload")
        load_tabular_data()
        
    elif dataset_type == "Text Data ":
        st.header("Text Data Processing")
        load_nlp_data()
        
if __name__ == "__main__":
    main()
