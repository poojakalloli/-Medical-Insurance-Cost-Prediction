# Medical Insurance Cost Prediction 🏥💰

A regression-based machine learning project to predict medical insurance charges based on personal attributes like age, BMI, smoking status, and region.

## 📊 Dataset
The dataset includes 1,338 records with the following features:
- `age`, `sex`, `bmi`, `children`, `smoker`, `region` → `charges` (target)

## 🔍 Workflow
1. **EDA**: Distribution analysis, outlier detection (IQR method), correlation heatmap, scatter/box plots against charges
2. **Preprocessing**:
   - Numerical (`age`, `bmi`, `children`) → MinMax Scaling
   - Binary categorical (`sex`, `smoker`) → Label Encoding
   - Nominal categorical (`region`) → One-Hot Encoding
3. **Modeling**: Trained and evaluated 5 regression algorithms:
   - K-Nearest Neighbors
   - Linear Regression
   - Support Vector Regression
   - Decision Tree
   - Random Forest
4. **Evaluation**: R² Score, MAE, RMSE for each model

## 📈 Results

| Model | R² Score | MAE | RMSE |
|-------|----------|------|------|
| Linear Regression | 0.636 | 2600.25 | 4382.41 |
| Random Forest | 0.607 | 2523.38 | 4554.16 |
| KNN | 0.578 | 2781.76 | 4722.27 |
| Decision Tree | 0.236 | 3031.88 | 6353.62 |
| SVM | -0.063 | 5619.69 | 7493.21 |

## 🏆 Conclusion
Linear Regression achieved the best overall R² and RMSE, while Random Forest had the lowest MAE. This suggests the charges-feature relationship is fairly linear once smoker/age/bmi are properly encoded, though tree-based models could likely improve with hyperparameter tuning.

## 🛠️ Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## 🚀 Future Work
- Hyperparameter tuning (GridSearchCV) for Random Forest & SVM
- Feature engineering (interaction terms like smoker × bmi)
- Deploy as a Streamlit app for interactive predictions
