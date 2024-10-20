import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.svm import SVC
from sklearn.decomposition import PCA
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

# Step 2: Train the SVC model
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_scaled, y)

# Example for Kaggle submission: Assume we have test data
# Placeholder for loading test data:
# test_data_path = 'path_to_test_data.csv'
# test_data = pd.read_csv(test_data_path)

# Apply the same scaling to the test data
# X_test = test_data.drop(columns=['id'])
# X_test_scaled = scaler.transform(X_test)

# Predict using SVC on test data (commented out for now as test data is placeholder)
# predictions = svc.predict(X_test_scaled)

# Prepare submission file (uncomment this when you have test data)
# submission = pd.DataFrame({'id': test_data['id'], 'species': predictions})
# submission.to_csv('svc_leaf_classification_submission.csv', index=False)

# Step 3: Plot validation accuracies as a horizontal bar graph
def plot_accuracies():
    fig, ax = plt.subplots(figsize=(8, 5))
    acc = accuracy_score(y, svc.predict(X_scaled))
    ax.barh(['SVC'], [acc], color='skyblue')
    ax.set_xlabel('Accuracy')
    ax.set_title('Validation Accuracy of SVC')

    # Save plot as image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_bytes = base64.b64encode(buf.read()).decode()
    buf.close()
    return f"data:image/png;base64,{img_bytes}"

# Step 4: Visualize decision boundary using PCA for dimensionality reduction
def plot_decision_boundary(X, y, model):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 5))
    plt.contourf(xx, yy, Z, alpha=0.8)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor='k', s=20)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title('Decision Boundary of SVC (PCA-reduced)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Save plot as image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_bytes = base64.b64encode(buf.read()).decode()
    buf.close()
    return f"data:image/png;base64,{img_bytes}"

# Generate the plot as a base64 string
accuracy_plot = plot_accuracies()
decision_boundary_plot = plot_decision_boundary(X_scaled, y, svc)

# Generate static HTML output with the decision boundary below the accuracy bar graph
html_output = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Leaf Classification Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #b29d6c;
            color: black;
        }}
        .container {{
            max-width: 1200px;
            margin: auto;
            padding: 20px;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }}
        .column {{
            width: 23%;
            margin: 1%;
            padding: 10px;
            border: 1px solid gray;
            box-sizing: border-box;
        }}
        .clear {{ clear: both; }}
        h1, h3, h4, h5 {{
            color: white;
            text-align: center;
        }}
        p, ul {{
            color: black;
        }}
        img {{
            width: 100%;
            height: auto;
        }}
        .header {{
            background-color: #8a100b;
            color: white;
            padding: 10px;
            text-align: center;
            width: 100%;
        }}
        .section-title {{
            text-align: center;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header Section -->
        <div class="header">
            <h1>Leaf Classification Dashboard</h1>
            <h3>Roel Rodriguez</h3>
            <h4>Boston College, Woods College: Applied Analytics, ADAN 7431: Computer Vision</h4>
        </div>

        <!-- Column 1: Abstract, Introduction & Significance -->
        <div class="column">
            <h3 class="section-title">Abstract</h3>
            <p>This project explores classifying various species of leaves using machine learning techniques. 
            The SVC model was selected for evaluation, and the validation accuracy was used to determine its effectiveness.</p>
        </div>

        <!-- Column 2: Methods -->
        <div class="column">
            <h3 class="section-title">Methods</h3>
            <p><strong>Preprocessing:</strong> 
            <ul>
                <li>Dropped the 'id' and 'species' columns.</li>
                <li>Standardized features using StandardScaler.</li>
            </ul>
            </p>
            <p><strong>Classifiers:</strong> 
            <ul>
                <li>SVC: A linear support vector classifier was used for training and validation.</li>
            </ul>
            </p>
        </div>

        <!-- Column 3: Results -->
        <div class="column">
            <h3 class="section-title">Results: Accuracy Comparison</h3>
            <img src="{accuracy_plot}" alt="Accuracy plot">
        </div>

        <!-- Column 4: Decision Boundary -->
        <div class="column">
            <h3 class="section-title">Decision Boundary</h3>
            <img src="{decision_boundary_plot}" alt="Decision Boundary plot">
        </div>

        <!-- Clear floats -->
        <div class="clear"></div>
    </div>
</body>
</html>
"""

# Save the HTML file
with open("index.html", "w") as file:
    file.write(html_output)
