import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, QuantileTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# -----------------------------
# STEP 1: Load & Clean Data
# -----------------------------
df = pd.read_csv("diabetes.csv")

cols_with_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)

# Fill missing values
df['Glucose'].fillna(df['Glucose'].median(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)

knn_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
imputer = KNNImputer(n_neighbors=3)
df[knn_features] = imputer.fit_transform(df[knn_features])

df.drop_duplicates(inplace=True)

# -----------------------------
# STEP 2: Outlier Removal
# -----------------------------
def remove_outliers_iqr(data, columns):
    for col in columns:
        Q1, Q3 = data[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data

df = remove_outliers_iqr(df, ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age'])

# -----------------------------
# STEP 3: Feature Transformation
# -----------------------------
for col in ['Insulin', 'SkinThickness', 'BMI', 'Age']:
    df[col] = np.log1p(df[col])

# Interaction features
df['Age_BMI'] = df['Age'] * df['BMI']
df['Glucose_BMI'] = df['Glucose'] / (df['BMI'] + 1)
df['Insulin_Glucose'] = df['Insulin'] / (df['Glucose'] + 1)
df['Preg_Age'] = df['Pregnancies'] / (df['Age'] + 1)
df['BP_Skin'] = df['BloodPressure'] / (df['SkinThickness'] + 1)

# -----------------------------
# STEP 4: Prepare Features and Labels
# -----------------------------
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale + polynomial features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
print(f"✅ Added {X_poly.shape[1] - X_scaled.shape[1]} interaction features\n")

# -----------------------------
# STEP 5: Train-Test Split + SMOTE
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42, stratify=y
)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# -----------------------------
# STEP 6: Model Training & Evaluation
# -----------------------------
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))
    return acc

# Logistic Regression
log_model = LogisticRegression(max_iter=2000, solver='lbfgs', C=2.0, class_weight='balanced')
log_model.fit(X_train_res, y_train_res)
log_acc = evaluate_model(log_model, X_test, y_test, "Logistic Regression")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=90, class_weight='balanced_subsample')
rf_model.fit(X_train_res, y_train_res)
rf_acc = evaluate_model(rf_model, X_test, y_test, "Random Forest")

# Linear Regression (rounded)
lin_reg = LinearRegression()
lin_reg.fit(X_train_res, y_train_res)
y_pred_lin = np.round(np.clip(lin_reg.predict(X_test), 0, 1))
lin_acc = accuracy_score(y_test, y_pred_lin)
print("\nLinear Regression Accuracy: {:.2f}%".format(lin_acc*100))

# Naive Bayes
qt = QuantileTransformer(output_distribution="normal", random_state=50)
X_train_res_nb = qt.fit_transform(X_train_res)
X_test_nb = qt.transform(X_test)

param_grid_nb = {'var_smoothing': np.logspace(-12, -1, 100)}
grid_nb = GridSearchCV(GaussianNB(), param_grid_nb, cv=10, scoring='accuracy', n_jobs=-1)
grid_nb.fit(X_train_res_nb, y_train_res)
best_smoothing = grid_nb.best_params_['var_smoothing']

nb_model = GaussianNB(var_smoothing=best_smoothing)
nb_model.fit(X_train_res_nb, y_train_res)
nb_acc = evaluate_model(nb_model, X_test_nb, y_test, "Naive Bayes")

# -----------------------------
# STEP 1: Feature Transformation for KNN
# -----------------------------
df_knn = df.copy()

# Log transform skewed features
for col in ['Insulin', 'SkinThickness', 'BMI', 'Age']:
    df_knn[col] = np.log1p(df_knn[col])

# Interaction features
df_knn['Age_BMI'] = df_knn['Age'] * df_knn['BMI']
df_knn['Glucose_BMI'] = df_knn['Glucose'] / (df_knn['BMI'] + 1)
df_knn['Insulin_Glucose'] = df_knn['Insulin'] / (df_knn['Glucose'] + 1)
df_knn['Preg_Age'] = df_knn['Pregnancies'] / (df_knn['Age'] + 1)
df_knn['BP_Skin'] = df_knn['BloodPressure'] / (df_knn['SkinThickness'] + 1)

# -----------------------------
# STEP 2: Prepare features and labels
# -----------------------------
X_knn = df_knn.drop('Outcome', axis=1)
y_knn = df_knn['Outcome']

# Balance dataset
sm = SMOTE(random_state=42)
X_res_knn, y_res_knn = sm.fit_resample(X_knn, y_knn)

# -----------------------------
# STEP 3: Scaling
# -----------------------------
# Use QuantileTransformer first to reduce skew and then StandardScaler
qt = QuantileTransformer(output_distribution='normal', random_state=42)
scaler = StandardScaler()

X_res_knn = qt.fit_transform(X_res_knn)
X_res_knn = scaler.fit_transform(X_res_knn)

X_test_knn = qt.transform(X_knn)
X_test_knn = scaler.transform(X_test_knn)

# -----------------------------
# STEP 4: Train-Test Split
# -----------------------------
X_train_knn, X_test_knn_split, y_train_knn, y_test_knn_split = train_test_split(
    X_res_knn, y_res_knn, test_size=0.2, random_state=42, stratify=y_res_knn
)

# -----------------------------
# STEP 5: Hyperparameter Tuning KNN
# -----------------------------
param_grid_knn = {
    'n_neighbors':[3,5,7,9,11],
    'weights':['distance'],
    'p':[1,2],
    'leaf_size':[15,25,35]
}

grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=10, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_train_knn, y_train_knn)

best_knn = grid_knn.best_estimator_
print("\n🔧 Best KNN parameters:", grid_knn.best_params_)

# -----------------------------
# STEP 6: Evaluate
# -----------------------------

y_pred_knn = best_knn.predict(X_test_knn_split)
knn_acc = accuracy_score(y_test_knn_split, y_pred_knn)
print("\n🏆 Enhanced Log-Transformed KNN Accuracy: {:.2f}%".format(knn_acc*100))
print(classification_report(y_test_knn_split, y_pred_knn))


# Decision Tree
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Assume X_train, y_train, X_test, y_test are pre-defined

# --- Apply SMOTE only on training data ---
sm = SMOTE(random_state=42)
X_train_res_dt, y_train_res_dt = sm.fit_resample(X_train, y_train)

# --- Define expanded parameter grid ---
param_grid_dt = {
    'criterion': ['gini', 'entropy'],  # Removed 'log_loss'
    'max_depth': [None, 5, 7, 9, 12, 15],
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 4, 6, 10],
    'max_features': [None, 'sqrt', 'log2']
}

# --- GridSearchCV for best hyperparameters ---
grid_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    param_grid_dt, cv=10, scoring='accuracy', n_jobs=-1
)
grid_dt.fit(X_train_res_dt, y_train_res_dt)

best_dt = grid_dt.best_estimator_
print("🔧 Best Decision Tree Parameters:", grid_dt.best_params_)

# --- Predict on original test set ---
y_pred_dt = best_dt.predict(X_test)
dt_acc = accuracy_score(y_test, y_pred_dt)

print(f"\n🌳 Decision Tree Accuracy: {dt_acc*100:.2f}%")
print(classification_report(y_test, y_pred_dt))


# Voting Ensemble (LR + RF + NB)
voting_clf = VotingClassifier(
    estimators=[('lr', log_model), ('rf', rf_model), ('nb', nb_model)],
    voting='soft'
)
voting_clf.fit(X_train_res_nb, y_train_res)
y_pred_vote = voting_clf.predict(X_test_nb)
vote_acc = accuracy_score(y_test, y_pred_vote)
print("\nVoting Ensemble Accuracy: {:.2f}%".format(vote_acc*100))

# -----------------------------
# STEP 7: Model Comparison Summary
# -----------------------------
print("\n🔍 Model Comparison Summary:")
print(f"Logistic Regression Accuracy: {log_acc*100:.2f}%")
print(f"Random Forest Accuracy:       {rf_acc*100:.2f}%")
print(f"Linear Regression Accuracy:   {lin_acc*100:.2f}%")
print(f"Naive Bayes Accuracy:         {nb_acc*100:.2f}%")
print(f"KNN Accuracy:                 {knn_acc*100:.2f}%")
print(f"Decision Tree Accuracy:       {dt_acc*100:.2f}%")
print(f"Voting Ensemble Accuracy:     {vote_acc*100:.2f}%")