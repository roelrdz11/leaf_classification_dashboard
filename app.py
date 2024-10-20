import pandas as pd
import numpy as np
import os
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from keras.preprocessing.image import load_img
from io import BytesIO

# Paths
train_data_path = "C:/Users/roelr/OneDrive/Documents/ADAN/7431/leaf_classification_dashboard/train.csv.zip"
test_data_path = "C:/Users/roelr/OneDrive/Documents/ADAN/7431/leaf_classification_dashboard/test.csv.zip"
images_dir = "C:/Users/roelr/OneDrive/Documents/ADAN/7431/leaf_classification_dashboard/images/images"
kaggle_image_path = "C:/Users/roelr/OneDrive/Pictures/leaf_kaggle.png"

# Step 1: Load training and test datasets
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Step 2: Preprocessing
X = train_data.drop(columns=['id', 'species'])  # Features
y = train_data['species']  # Labels
X_test = test_data.drop(columns=['id'])  # Test features for Kaggle submission

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train a RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y_encoded)

# Step 4: Predict probabilities for the test set
test_probabilities = rf_model.predict_proba(X_test_scaled)

# Step 5: Evaluate RandomForest performance (Kaggle score)
rf_logloss = log_loss(y_encoded, rf_model.predict_proba(X_scaled))
print(f"RandomForest LogLoss on Training Data: {rf_logloss}")

# Step 6: Prepare submission DataFrame
submission_df = pd.DataFrame(test_probabilities, columns=le.classes_)
submission_df.insert(0, 'id', test_data['id'])  # Insert 'id' column from test data

# Save the submission file
submission_df.to_csv('leaf_classification_submission.csv', index=False)
print("Submission file saved as 'leaf_classification_submission.csv'")

# Step 7: Generate HTML-embedded images
def generate_image_html():
    img_html = ""
    img_width_percentage = 100 // 5  # To display 5 images in the same row
    for i in range(5):
        img_file = np.random.choice(os.listdir(images_dir))
        img_path = os.path.join(images_dir, img_file)
        img = load_img(img_path)

        # Convert the image to base64 encoding
        img_buffer = BytesIO()
        img.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        # Embed the image into the HTML, making sure 5 images fit in the same row
        img_html += f'<img src="data:image/png;base64,{img_base64}" alt="Random Image {i+1}" style="width:{img_width_percentage}%; height:auto; display:inline-block;">\n'
    
    return img_html

# Step 8: Embed Kaggle image
def embed_kaggle_image():
    with open(kaggle_image_path, "rb") as image_file:
        kaggle_img_base64 = base64.b64encode(image_file.read()).decode()
    return f'<img src="data:image/png;base64,{kaggle_img_base64}" alt="Kaggle Submission Image" style="width:100%; height:auto;">'

# Step 9: Generate the HTML output including images in the Methods section and Kaggle image in Results section
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
            Three classifiers (Random Forest, K-Neighbors, and SVC) were evaluated, and submission logloss scores
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
            </p>

            <!-- Insert random images here, displayed in a row -->
            {generate_image_html()}
        </div>

        <!-- Column 3: Results -->
        <div class="column">
            <h3 class="section-title">Results: Kaggle Submission</h3>
            <p>We evaluated three models on the Kaggle platform:</p>
            <ul>
                <li><strong>SVC</strong>: 2.088 log loss</li>
                <li><strong>KNeighbors</strong>: 1.782 log loss</li>
                <li><strong>RandomForest</strong>: 0.687 log loss (best performance)</li>
            </ul>
            <p>As RandomForest achieved the best performance, we used it for the final model.</p>
            <!-- Embed Kaggle image -->
            {embed_kaggle_image()}
        </div>

        <!-- Column 4: Discussion and Conclusion -->
        <div class="column">
            <h3 class="section-title">Discussion</h3>
            <p>The Random Forest model achieved the best validation accuracy and the lowest log loss. 
            The Kaggle submission results are visualized in the figure above, demonstrating the model's performance compared to other classifiers.</p>

            <h3 class="section-title">Conclusion</h3>
            <p>This study demonstrates the effectiveness of machine learning models in leaf classification tasks. 
            The Random Forest model provided the best performance, indicating its suitability for this dataset. 
            Future studies could explore more complex models or additional feature engineering to improve classification results further.</p>

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

