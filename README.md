# Data Preprocessing & Feature Selection Lab Notebook

## 📌 Overview
This Jupyter notebook contains a series of exercises demonstrating essential data preprocessing and feature selection techniques using Python's data science ecosystem. The exercises cover handling missing data, encoding categorical variables, dataset splitting, feature scaling, and feature selection methods.

## 🎯 Learning Objectives
- Handle missing data using various imputation strategies
- Encode categorical variables (ordinal and nominal)
- Split datasets while maintaining class distributions
- Apply feature scaling techniques
- Implement feature selection methods (SBS, Random Forest, L1 Regularization)

## 📂 Project Structure
```
├── Exercise 1: Handling Missing Data
├── Exercise 2: Encoding Categorical Data
├── Exercise 3: Dataset Splitting
├── Exercise 4: Feature Scaling
└── Exercise 5: Feature Selection Methods
```

## 🛠️ Technologies Used
- **Python 3.11**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and preprocessing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization

## 📊 Exercises Breakdown

### **Exercise 1: Handling Missing Data**
**1.1** - Identify missing values in a sample dataset
- Uses `pandas` to load CSV data with missing values
- Demonstrates `isnull().sum()` for missing value detection
- Each column (A, B, C, D) has 1 missing value

**1.2** - Various strategies for handling missing data:
- Drop rows with any missing values
- Drop columns with any missing values
- Drop rows where all values are missing
- Drop rows with fewer than specified non-missing values
- Drop rows based on specific column conditions

**1.3** - Imputation techniques using `SimpleImputer`:
- Mean imputation
- Median imputation
- Most frequent imputation (useful for categorical data)

### **Exercise 2: Encoding Categorical Data**
**2.1** - Ordinal encoding for size categories:
- Maps categorical sizes (S, M, L, XL) to numerical values (0, 1, 2, 3)
- Demonstrates inverse mapping to recover original labels

**2.2** - Error handling with mixed data types:
- Shows what happens when trying to fit a classifier with dictionary-type labels
- Demonstrates importance of proper label encoding

**2.3** - One-hot encoding for nominal features:
- Encodes 'color' and 'classlabel' columns using `pd.get_dummies()`
- Uses `drop_first=True` to avoid the dummy variable trap
- Explains why dropping one column prevents multicollinearity

### **Exercise 3: Dataset Splitting**
**3.1** - Stratified train-test split:
- Loads the Wine dataset from UCI repository
- Splits data (80% train, 20% test) with `stratify=y`
- Verifies class proportions are preserved in both sets
- Shows how stratification improves model evaluation reliability

### **Exercise 4: Feature Scaling**
**4.1** - Min-Max Scaling:
- Applies `MinMaxScaler` to normalize features to [0, 1] range
- Visualizes Alcohol feature distribution before/after scaling
- Uses matplotlib for comparative histogram plots

**4.2** - Standardization (Z-score normalization):
- Applies `StandardScaler` to standardize features (mean=0, std=1)
- Compares feature-wise means and standard deviations
- Discusses when to use standardization vs. normalization

### **Exercise 5: Feature Selection Methods**
**5.1** - L1 Regularization (Lasso):
- Trains logistic regression with L1 penalty across different C values
- Visualizes coefficient shrinkage as regularization increases
- Demonstrates feature elimination through L1 regularization

**5.2** - Sequential Backward Selection (SBS):
- Implements custom SBS class for feature selection
- Uses KNN classifier with SBS to select optimal features
- Identifies top 3 features: ['Alcohol', 'Flavanoids', 'Proanthocyanins']
- Achieves 91.67% test accuracy with selected features

**5.3** - Random Forest Feature Importance:
- Trains Random Forest classifier with 500 estimators
- Ranks features by importance scores
- Selects features with importance ≥ 0.1
- Compares RF-selected features with SBS-selected features

## 📈 Key Insights

### **Missing Data Handling:**
- Different strategies (deletion vs. imputation) suit different scenarios
- `most_frequent` imputation preserves mode for categorical data
- Mean/median imputation works well for normally distributed numerical data

### **Categorical Encoding:**
- **Ordinal encoding**: Preserves order relationships (S < M < L < XL)
- **One-hot encoding**: Essential for nominal data without intrinsic order
- **Dummy variable trap**: Always drop one category to avoid multicollinearity

### **Feature Scaling:**
- **Min-Max**: Best when bounded range is required (e.g., neural networks)
- **Standardization**: Better for algorithms assuming normal distribution
- Always fit scalers on training data only, then transform both train and test

### **Feature Selection:**
- **L1 Regularization**: Fast, embedded method that performs selection during training
- **SBS**: Model-agnostic but computationally expensive (O(n²))
- **Random Forest**: Provides importance scores but doesn't eliminate redundant features
- Different methods may select different optimal feature sets

## 🎯 Performance Comparison
| Method | Selected Features | Test Accuracy |
|--------|-------------------|---------------|
| All Features | 13 features | Baseline |
| SBS | Alcohol, Flavanoids, Proanthocyanins | **91.67%** |
| Random Forest | Top 5 features by importance | 52.78% |

**Key Finding**: SBS-selected subset (3 features) outperformed Random Forest selection (5 features), demonstrating that fewer, well-chosen features can sometimes yield better performance.

## 🚀 How to Run
1. Clone the repository
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib`
3. Open the Jupyter notebook: `jupyter notebook lb4.ipynb`
4. Run cells sequentially or as needed

## 📝 Dependencies
```bash
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
```

## 🔗 Dataset
- **Wine Dataset**: Loaded from UCI Machine Learning Repository
- Contains 178 samples, 13 features, 3 classes
- Used for classification tasks in exercises 3-5

## 🤝 Contributing
Feel free to fork this repository and submit pull requests with:
- Additional preprocessing techniques
- More feature selection methods
- Extended visualizations
- Performance comparisons

## 📚 References
1. scikit-learn Documentation
2. UCI Machine Learning Repository
3. "Python Machine Learning" by Sebastian Raschka

## 🏷️ License
This project is available for educational purposes. Please credit the original sources when using the code or concepts.

---
