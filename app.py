import pandas as pd
import numpy as np
import os
import base64
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from keras.preprocessing.image import load_img
from io import BytesIO

# Paths
train_data_path = "C:/Users/roelr/OneDrive/Documents/ADAN/7431/leaf_classification_dashboard/train.csv.zip"
test_data_path = "C:/Users/roelr/OneDrive/Documents/ADAN/7431/leaf_classification_dashboard/test.csv.zip"
images_dir = "C:/Users/roelr/OneDrive/Documents/ADAN/7431/leaf_classification_dashboard/images/images"
knn_image_path = "C:/Users/roelr/OneDrive/Pictures/knn_kaggle.png"
svc_image_path = "C:/Users/roelr/OneDrive/Pictures/svc_kaggle.png"
rf_image_path = "C:/Users/roelr/OneDrive/Pictures/rf_kaggle.png"

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

# Step 3: Train a KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_scaled, y_encoded)

# Step 4: Predict probabilities for the test set
test_probabilities = knn_model.predict_proba(X_test_scaled)

# Step 5: Evaluate KNeighbors performance
knn_logloss = log_loss(y_encoded, knn_model.predict_proba(X_scaled))
print(f"KNeighbors LogLoss on Training Data: {knn_logloss}")

# Step 6: Prepare submission DataFrame
submission_df = pd.DataFrame(test_probabilities, columns=le.classes_)
submission_df.insert(0, 'id', test_data['id'])  # Insert 'id' column from test data

# Save the submission file
submission_df.to_csv('kneighbors_leaf_classification_submission.csv', index=False)
print("Submission file saved as 'kneighbors_leaf_classification_submission.csv'")

# Step 7: Embed Kaggle images
def embed_kaggle_images():
    images_html = ""
    image_paths = [knn_image_path, svc_image_path, rf_image_path]
    labels = ['KNN', 'SVC', 'RandomForest']
    
    for i, img_path in enumerate(image_paths):
        with open(img_path, "rb") as image_file:
            img_base64 = base64.b64encode(image_file.read()).decode()
        images_html += f'<h5>{labels[i]} Kaggle Submission</h5>'
        images_html += f'<img src="data:image/png;base64,{img_base64}" alt="{labels[i]} Kaggle Image" style="width:100%; height:auto; margin-bottom:20px;">\n'
    
    return images_html

# Step 8: Generate HTML-embedded images for methods
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

# Step 9: Generate the HTML output including KNN modeling and Kaggle images
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
            Three classifiers (KNN, Random Forest, and SVC) were evaluated, and the Kaggle logloss scores 
            for each model were compared to determine the most effective approach for leaf classification.</p>
            
            <h3 class="section-title">Introduction & Significance</h3>
            <p>Accurate classification of leaves is essential in botanical studies, plant taxonomy, and environmental monitoring. 
            Machine learning techniques provide a scalable and efficient way to categorize plant species based on leaf characteristics, 
            which can significantly contribute to biodiversity conservation efforts and ecological studies.</p>
        </div>

        <!-- Column 2: Methods -->
        <div class="column">
            <h3 class="section-title">Methods</h3>
            <p><strong>Preprocessing:</strong> 
            <ul>
                <li>No null values were present in the dataset.</li>
                <li>The classes were balanced, with all species represented equally.</li>
                <li>Dropped the 'id' and 'species' columns from the feature set.</li>
                <li>Standardized features using StandardScaler to ensure fair comparison across models.</li>
            </ul>
            </p>
            <p><strong>Classifiers:</strong> 
            <ul>
                <li>KNN: A distance-based classifier.</li>
                <li>RandomForest: A tree-based ensemble method.</li>
                <li>SVC: A linear support vector classifier.</li>
            </ul>
            </p>

            <!-- Insert random training images here -->
            {generate_image_html()}
        </div>

        <!-- Column 3: Results -->
        <div class="column">
            <h3 class="section-title">Results: Kaggle Submissions</h3>
            <p>We evaluated three models on the Kaggle platform, and KNN achieved the best performance:</p>
            <ul>
                <li><strong>KNN</strong>: 0.14407 log loss</li>
                <li><strong>SVC</strong>: 2.08805 log loss</li>
                <li><strong>RandomForest</strong>: 0.68685 log loss</li>
            </ul>
            <p>As KNN achieved the best performance, we used it for the final model.</p>

            <!-- Embed Kaggle images -->
            {embed_kaggle_images()}
        </div>

        <!-- Column 4: Discussion and Conclusion -->
        <div class="column">
            <h3 class="section-title">Discussion</h3>
            <p>KNN performed exceptionally well due to its ability to capture distances between points effectively in this leaf classification problem. 
            Since leaf data often involves subtle differences between species, a distance-based method like KNN was able to distinguish between them more accurately than linear models or tree-based ensembles. 
            The results from Kaggle further reinforce this conclusion, with KNN outperforming SVC and RandomForest in terms of log loss.</p>

            <h3 class="section-title">Conclusion</h3>
            <p>This study demonstrates the effectiveness of machine learning models in leaf classification tasks. 
            The KNN model provided the best performance, indicating its suitability for this dataset. 
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

