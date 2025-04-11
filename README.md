# PersianNewsClassifier

A machine learning project for categorizing Persian news articles using Support Vector Machines (SVM). This project includes data preprocessing, feature extraction with TF-IDF, and supervised classification using Scikit-learn.

## 📰 Dataset

The dataset `per.csv` consists of Persian news articles with the following columns:
- `NewsID`: Unique identifier for each article.
- `Title`: The title of the news.
- `Body`: The full text of the news article.
- `Date` and `Time`: Timestamps.
- `Category`, `Category2`: Label(s) for classification.

## 🧹 Preprocessing

The preprocessing pipeline includes:
1. **Cleaning Text**:
   - Removal of carriage returns (`\r\n`), extra whitespaces.
2. **Text Normalization** using `hazm`:
   - Tokenization of Persian text.
   - Optionally, stemming and lemmatization (hazm supports this).
3. **Label Encoding**:
   - `Category2` column (primary label) is converted into numeric form using `LabelEncoder`.
4. **Train-Test Split**:
   - Stratified split for robust evaluation (typically 80/20 or 70/30).

## 🧠 Algorithm

### Model: **Support Vector Machine (SVM)**
- Kernel: `linear`
- Features: TF-IDF vectors extracted from the article `Title` or `Body`.
- Tools: `sklearn.svm.SVC`

### Steps:
1. **Feature Extraction**:
   - Used `TfidfVectorizer` to convert Persian text into numerical feature vectors.
2. **Model Training**:
   - Fitted a linear SVM on the training set.
3. **Evaluation**:
   - Used `classification_report` and `confusion_matrix` to assess performance.

