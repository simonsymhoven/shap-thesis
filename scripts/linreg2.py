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

matplotlib.use('Agg')

def load_data() -> pd.DataFrame:
    """
    Loads and returns the dataset from the given URL as a Pandas DataFrame.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    url = "https://data.ub.uni-muenchen.de/2/1/miete03.asc"
    df = pd.read_excel("Concrete_Data.xls")
    df = df.rename(
        columns={
            'Cement (component 1)(kg in a m^3 mixture)':'cement',
            'Blast Furnace Slag (component 2)(kg in a m^3 mixture)':'blast',
            'Fly Ash (component 3)(kg in a m^3 mixture)':'ash',
            'Water  (component 4)(kg in a m^3 mixture)':'water',
            'Superplasticizer (component 5)(kg in a m^3 mixture)':'superplasticizer',
            'Coarse Aggregate  (component 6)(kg in a m^3 mixture)':'coarse',
            'Fine Aggregate (component 7)(kg in a m^3 mixture)':'fine',
            'Age (day)':'age',
            "Concrete compressive strength(MPa, megapascals) ": "strength"
        }
    )
    df = df.drop_duplicates()

    for column in df.columns:
        df[column] += 1
        df[column] = np.log(df[column])

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

    plt.savefig('images/residuals_2.png', dpi=300)

def plot_corr(X: pd.DataFrame) -> None:
    """
    Creates and saves a correlation heatmap plot.

    Args:
        X (pd.DataFrame): The feature matrix.
    """
    plt.figure(figsize=(12,10))
    sns.heatmap(X.corr(), annot=True, cmap="YlGnBu")
    plt.savefig('images/corr_2.png', dpi=300)

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
    plt.savefig('images/shap_summary_plot_2.png', dpi=300)

    plt.figure(figsize=(12,10))
    shap.plots.bar(shap_values, max_display=12)
    plt.tight_layout()
    plt.savefig('images/shap_bar_plot_2.png', dpi=300)

    plt.figure(figsize=(12,10))
    shap.plots.waterfall(shap_values[idx], max_display=12)
    plt.tight_layout()
    plt.savefig('images/shap_waterfall_plot_2.png', dpi=300)

df = load_data()

X = df.drop(['strength'], axis=1) 
y = df['strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linreg, params = model(X=X_train_scaled, y=y_train, cols=X.columns)
y_pred = linreg.predict(X_test_scaled)

explainer = shap.Explainer(linreg, X_train_scaled)
shap_values = explainer(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
train_score = linreg.score(X_train_scaled, y_train)
test_score = linreg.score(X_test_scaled, y_test)

print(mae)
print(mse)
print(train_score)
print(test_score)

print(params)

plot_corr(X=df)
plot_residuals(y_test=y_test, y_pred=y_pred)
plot_shap(shap_values=shap_values, idx=148)