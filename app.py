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

# Step 3: Plot validation accuracies as a horizontal bar graph
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

# Generate the plot as a base64 string
accuracy_plot = plot_accuracies(accuracies)

# Generate static HTML output
html_output = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Leaf Classification Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; background-color: #b29d6c; }} /* Set background color */
        .container {{ max-width: 1200px; margin: auto; padding: 20px; }}
        .column {{ float: left; width: 23%; margin: 1%; padding: 10px; border: 1px solid gray; }}
        .clear {{ clear: both; }}
        h1, h3, h4, h5 {{ color: white; text-align: center; }}
        img {{ width: 100%; height: auto; }}
        .header {{ background-color: #8a100b; color: white; padding: 10px; text-align: center; }}
        .section-title {{ text-align: center; font-weight: bold; }}
        p {{ color: white; }}
        ul {{ color: white; }}
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
            
            <h3 class="section-title">Introduction & Significance</h3>
            <p>Accurate leaf classification is important in plant taxonomy and ecology. By applying machine learning models 
            to leaf margin and texture data, we can efficiently classify leaf species. This study highlights the potential 
            of machine learning to provide insights into biological datasets.</p>
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

        <!-- Column 3: Results (Bar graph of validation accuracies) -->
        <div class="column">
            <h3 class="section-title">Results: Accuracy Comparison</h3>
            <img src="{accuracy_plot}" alt="Accuracy plot">
        </div>

        <!-- Column 4: Discussion, Conclusion, and References -->
        <div class="column">
            <h3 class="section-title">Discussion</h3>
            <p>The SVC model achieved the highest validation accuracy, followed by the K-Neighbors and RandomForest classifiers. 
            This result suggests that a linear classifier like SVC may be more suitable for this dataset, as it outperformed 
            both tree-based and distance-based models.</p>

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
