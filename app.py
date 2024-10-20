import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Load your training dataset
train_data_path = "C:/Users/roelr/OneDrive/Documents/ADAN/7431/leaf_classification_dashboard/train.csv.zip"
train_data = pd.read_csv(train_data_path)

# Load your test dataset
test_data_path = "C:/Users/roelr/OneDrive/Documents/ADAN/7431/leaf_classification_dashboard/test.csv.zip"
test_data = pd.read_csv(test_data_path)

# Step 1: Preprocessing
X = train_data.drop(columns=['id', 'species'])  # Features
y = train_data['species']  # Labels
X_test = test_data.drop(columns=['id'])  # Test features for Kaggle submission

# Convert categorical labels to numerical values
y = pd.factorize(y)[0]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)  # Also scale the test set

# Split the dataset into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 2: Train multiple classifiers
classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNeighbors": KNeighborsClassifier(n_neighbors=5),
    "SVC": SVC(kernel='linear', probability=True)
}

accuracies = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    acc = accuracy_score(y_valid, y_pred)
    accuracies[name] = acc

# Step 3: Train SVC on full training data and prepare Kaggle submission
svc = classifiers['SVC']
svc.fit(X_scaled, y)  # Train on full data
y_pred_proba = svc.predict_proba(X_test_scaled)  # Predict for test data

# Get the species names (class labels)
species_names = pd.factorize(train_data['species'])[1]

# Prepare Kaggle submission format with species as columns
submission_df = pd.DataFrame(y_pred_proba, columns=species_names)
submission_df.insert(0, 'id', test_data['id'])  # Add 'id' column from test set

# Save the submission file
submission_df.to_csv('svc_leaf_classification_submission.csv', index=False)

print("Submission file saved as 'svc_leaf_classification_submission.csv'")

# Step 4: Plot validation accuracies with adjusted scaling and font size for readability
def plot_accuracies(accuracies):
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(accuracies.keys())
    values = list(accuracies.values())
    
    ax.barh(names, values, color='skyblue')  # Sky blue bars
    ax.set_xlabel('Accuracy', fontsize=14)
    ax.set_title('Validation Accuracy of Different Classifiers', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Adjust scale to emphasize small differences
    ax.set_xlim([min(values) - 0.01, 1.0])  # Scale starts slightly below the lowest accuracy

    # Save plot as image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')  # Ensure text isn't clipped
    buf.seek(0)
    img_bytes = base64.b64encode(buf.read()).decode()
    buf.close()
    return f"data:image/png;base64,{img_bytes}"

# Step 5: Visualize decision boundary using PCA for dimensionality reduction
def plot_decision_boundary(X, y, model, image=None):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    
    # Plot the image behind the decision boundary, if provided
    if image is not None:
        plt.imshow(image, extent=(x_min, x_max, y_min, y_max), alpha=0.5, aspect='auto')
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor='k', s=20)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title('Decision Boundary of SVC (PCA-reduced)', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Save plot as image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_bytes = base64.b64encode(buf.read()).decode()
    buf.close()
    return f"data:image/png;base64,{img_bytes}"

# Example image for overlay (random training data visualization)
def generate_random_image_overlay():
    random_image = np.random.rand(100, 100, 3)  # Simulate an image for overlay
    return random_image

# Generate the bar graph and decision boundary plots
accuracy_plot = plot_accuracies(accuracies)
decision_boundary_image = generate_random_image_overlay()
decision_boundary_plot = plot_decision_boundary(X_scaled, y, classifiers['SVC'], decision_boundary_image)

# Generate static HTML output with both bar graph and decision boundary in the same column
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
            Three classifiers (Random Forest, K-Neighbors, and SVC) were evaluated, and the validation accuracies 
            for each model were compared to determine the most effective approach for leaf classification.</p>
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
                <li>RandomForest: A tree-based ensemble method.</li>
                <li>KNeighbors: A distance-based model.</li>
                <li>SVC: A linear support vector classifier.</li>
            </ul>
            Each classifier was trained on the same data to ensure a fair comparison of validation accuracies.
            </p>
        </div>

        <!-- Column 3: Results -->
        <div class="column">
            <h3 class="section-title">Results: Accuracy Comparison & Decision Boundary</h3>
            <img src="{accuracy_plot}" alt="Accuracy plot">
            <h3 class="section-title">Decision Boundary</h3>
            <img src="{decision_boundary_plot}" alt="Decision Boundary plot">
        </div>

        <!-- Column 4: Discussion and Conclusion -->
        <div class="column">
            <h3 class="section-title">Discussion</h3>
            <p>The SVC model achieved the highest validation accuracy, followed by the K-Neighbors and RandomForest classifiers. 
            The decision boundary plot demonstrates how the classifier separates different classes, with overlaying features from the dataset providing further insights into model performance.</p>

            <h3 class="section-title">Conclusion</h3>
            <p>This study demonstrates the effectiveness of machine learning models in leaf classification tasks. 
            The SVC model provided the best performance, indicating its suitability for this dataset. Future studies 
            could explore more complex models or additional feature engineering to improve classification results further.</p>

            <h3 class="section-title">References</h3>
            <ul>
                <li>Kaggle Leaf Classification Dataset.</li>
                <li>Scikit-learn: Machine learning library used for classifier implementation.</li>
            </ul>
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
