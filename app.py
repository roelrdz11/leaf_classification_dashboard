from dash import dcc, html
import dash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load your dataset
train_data_path = "C:/Users/roelr/OneDrive/Documents/ADAN/7431/leaf_classification_dashboard/train.csv.zip"
train_data = pd.read_csv(train_data_path)

# Step 1: Preprocessing
X = train_data.drop(columns=['id', 'species'])  # Features
y = train_data['species']  # Labels

# Convert categorical labels to numerical values
y = pd.factorize(y)[0]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 2: Train multiple classifiers
classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNeighbors": KNeighborsClassifier(n_neighbors=5),
    "SVC": SVC(kernel='linear')
}

accuracies = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

# Step 3: Plot validation accuracies as a horizontal bar graph with sky blue bars
def plot_accuracies(accuracies):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(accuracies.keys())
    values = list(accuracies.values())
    
    ax.barh(names, values, color='skyblue')  # Sky blue bars
    ax.set_xlabel('Accuracy')
    ax.set_title('Validation Accuracy of Different Classifiers')

    # Adjust scale to emphasize small differences
    ax.set_xlim([min(values) - 0.01, 1.0])  # Scale starts slightly below the lowest accuracy

    # Save plot as image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_bytes = base64.b64encode(buf.read()).decode()
    buf.close()
    return f"data:image/png;base64,{img_bytes}"

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    
    # Header Section (Title, Name, Affiliation)
    html.Div([
        html.H1("Leaf Classification Dashboard"),
        html.H3("Roel Rodriguez"),
        html.H4("Boston College, Woods College: Applied Analytics, ADAN 7399: Computer Vision"),
    ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#8a100b'}),
    
    # Main content: 4 columns
    html.Div([
        # Column 1: Abstract, Introduction & Significance
        html.Div([
            html.Div([
                html.H3("Abstract"),
                dcc.Markdown('''
                This project explores classifying various species of leaves using machine learning techniques. 
                Three classifiers (Random Forest, K-Neighbors, and SVC) were evaluated, and the validation accuracies 
                for each model were compared to determine the most effective approach for leaf classification.
                ''')
            ], style={'overflowY': 'scroll', 'height': '180px', 'padding': '10px'}),
            
            html.Div([
                html.H3("Introduction & Significance"),
                dcc.Markdown('''
                Accurate leaf classification is important in plant taxonomy and ecology. By applying machine learning models 
                to leaf margin and texture data, we can efficiently classify leaf species. This study highlights the potential 
                of machine learning to provide insights into biological datasets.
                ''')
            ], style={'overflowY': 'scroll', 'height': '180px', 'padding': '10px'}),
        ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'border': '1px solid gray'}),
        
        # Column 2: Methods
        html.Div([
            html.Div([
                html.H3("Methods"),
                dcc.Markdown('''
                - **Preprocessing**: 
                    1. Dropped the 'id' and 'species' columns.
                    2. Standardized features using StandardScaler.
                
                - **Classifiers**: 
                    1. RandomForest: A tree-based ensemble method.
                    2. KNeighbors: A distance-based model.
                    3. SVC: A linear support vector classifier.
                
                Each classifier was trained on the same data to ensure a fair comparison of validation accuracies.
                '''),
            ], style={'overflowY': 'scroll', 'height': '320px', 'padding': '10px'}),
        ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'border': '1px solid gray'}),
        
        # Column 3: Results (Bar graph of validation accuracies)
        html.Div([
            html.Div([
                html.H3("Results: Accuracy Comparison"),
                html.Img(src=plot_accuracies(accuracies), style={'width': '100%', 'height': 'auto'})
            ], style={'padding': '10px'}),
        ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'border': '1px solid gray'}),
        
        # Column 4: Discussion, Conclusion, and References
        html.Div([
            html.Div([
                html.H3("Discussion"),
                dcc.Markdown(f'''
                The SVC model achieved the highest validation accuracy, followed by the K-Neighbors and RandomForest classifiers. 
                This result suggests that a linear classifier like SVC may be more suitable for this dataset, as it outperformed 
                both tree-based and distance-based models.
                ''')
            ], style={'overflowY': 'scroll', 'height': '180px', 'padding': '10px'}),
            
            html.Div([
                html.H3("Conclusion"),
                dcc.Markdown('''
                This study demonstrates the effectiveness of machine learning models in leaf classification tasks. 
                The SVC model provided the best performance, indicating its suitability for this dataset. Future studies 
                could explore more complex models or additional feature engineering to improve classification results further.
                ''')
            ], style={'overflowY': 'scroll', 'height': '180px', 'padding': '10px'}),
            
            html.Div([
                html.H3("References"),
                dcc.Markdown('''
                - Kaggle Leaf Classification Dataset.
                - Scikit-learn: Machine learning library used for classifier implementation.
                ''')
            ], style={'overflowY': 'scroll', 'height': '180px', 'padding': '10px'}),
        ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'border': '1px solid gray'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),  # Flexbox layout to arrange columns
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)  # default port

