import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from typing import List
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


matplotlib.use('Agg')

def load_data() -> pd.DataFrame:
    """
    Loads and returns the dataset from the given URL as a Pandas DataFrame.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    df = pd.read_csv("CASP.csv")
    df = df.drop_duplicates()

    return df
    
def model(X: pd.DataFrame, y: pd.Series, cols: List[str]) -> (LinearRegression, pd.DataFrame):
    """
    Fits a Linear Regression model to the given data.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        cols (List): The names of columns.

    Returns:
        LinearRegression: The fitted Linear Regression model.
        DataFrame: The coefficients as DataFrame.
    """
    linreg = LinearRegression()
    linreg.fit(X, y)

    cdf = pd.DataFrame(linreg.coef_.round(5), cols, columns=['Coefficients'])
    cdf.loc['Intercept'] = linreg.intercept_.round(5)

    return linreg, cdf

def plot_residuals(y_test: pd.Series, y_pred: pd.Series) -> None:
    """
    Creates a residual plots.

    Args:
        y_test (pd.Series): The actual target values.
        y_pred (pd.Series): The predicted target values.
    """
    residuals = y_test - y_pred

    _, axs = plt.subplots(1, 2, figsize=(16, 6))

    axs[0].scatter(y_test, y_pred, alpha=0.5)
    axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', linewidth=2)
    axs[0].set_title('Actual vs. Predicted Values')
    axs[0].set_xlabel('Actual Values')
    axs[0].set_ylabel('Predicted Values')

    axs[1].scatter(y_pred, residuals, alpha=0.5)
    axs[1].axhline(y=0, color='k', linestyle='--', linewidth=2)
    axs[1].set_title('Residuals vs. Predicted Values')
    axs[1].set_xlabel('Predicted Values')
    axs[1].set_ylabel('Residuals')

    plt.savefig('images/residuals_3.png', dpi=300)

def plot_corr(X: pd.DataFrame) -> None:
    """
    Creates and saves a correlation heatmap plot.

    Args:
        X (pd.DataFrame): The feature matrix.
    """
    plt.figure(figsize=(12,10))
    sns.heatmap(X.corr(), annot=True, cmap="YlGnBu")
    plt.savefig('images/corr_3.png', dpi=300)

def plot_shap(shap_values: shap.Explanation, idx: int) -> None:
    """
    Creates and saves SHAP summary and bar plots.

    Args:
        shap_values (shap.Explanation): SHAP values.
        idx (int): Index for the SHAP waterfall plot.
    """
    plt.figure(figsize=(12,10))
    shap.summary_plot(shap_values, X_test, max_display=12)
    plt.tight_layout()
    plt.savefig('images/shap_summary_plot_3.png', dpi=300)

    plt.figure(figsize=(12,10))
    shap.plots.bar(shap_values, max_display=12)
    plt.tight_layout()
    plt.savefig('images/shap_bar_plot_3.png', dpi=300)

    plt.figure(figsize=(12,10))
    shap.plots.waterfall(shap_values[idx], max_display=12)
    plt.tight_layout()
    plt.savefig('images/shap_waterfall_plot_3.png', dpi=300)

def vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

df = load_data()

X = df.drop(['RMSD'], axis=1) 
y = df['RMSD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11)

linreg, params = model(X=X_train, y=y_train, cols=X.columns)
y_pred = linreg.predict(X_test)

explainer = shap.Explainer(linreg, X_train)
shap_values = explainer(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
train_score = linreg.score(X_train, y_train)
test_score = linreg.score(X_test, y_test)


print(vif(X_train))

print(mae)
print(mse)
print(train_score)
print(test_score)

print(params)

plot_corr(X=df)
plot_residuals(y_test=y_test, y_pred=y_pred)
plot_shap(shap_values=shap_values, idx=148)