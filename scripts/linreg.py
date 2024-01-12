import requests
import zipfile
from io import BytesIO
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import shap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

matplotlib.use('Agg')

def load_data() -> pd.DataFrame:
    """
    Loads and returns the dataset from the given URL as a Pandas DataFrame.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    url = "https://archive.ics.uci.edu/static/public/440/sgemm+gpu+kernel+performance.zip"

    r = requests.get(url)

    if r.ok:
        with zipfile.ZipFile(BytesIO(r.content)) as thezip:
            with thezip.open("sgemm_product.csv") as thefile:
                return pd.read_csv(thefile, sep=",")
    else:
        raise Exception("Something went wrong.")
    
def model(X: pd.DataFrame, y: pd.Series) -> (LinearRegression, pd.DataFrame):
    """
    Fits a Linear Regression model to the given data.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.

    Returns:
        LinearRegression: The fitted Linear Regression model.
        DataFrame: The coefficients as DataFrame.
    """
    model = LinearRegression()
    model.fit(X, y)

    cdf = pd.DataFrame(model.coef_.round(5), X.columns, columns=['Coefficients'])
    cdf.loc['Intercept'] = model.intercept_.round(5)

    return model, cdf

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

    plt.savefig('images/residuals_gpu.png', dpi=300)

def plot_corr(df: pd.DataFrame) -> None:
    """
    Creates and saves a correlation heatmap plot.

    Args:
        df (pd.DataFrame): The feature matrix.
    """
    plt.figure(figsize=(12,10))
    sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", fmt=".2f")
    plt.savefig('images/corr_gpu.png', dpi=300)

def plot_shap(shap_values: shap.Explanation, idx: int) -> None:
    """
    Creates and saves SHAP beeswarm, bar and waterfall plots.

    Args:
        shap_values (shap.Explanation): SHAP values.
        idx (int): Index for the SHAP waterfall plot.
    """
    plt.figure(figsize=(12,10))
    shap.plots.beeswarm(shap_values)
    plt.tight_layout()
    plt.savefig('images/shap_beeswarm_plot_gpu.png', dpi=300)

    plt.figure(figsize=(12,10))
    shap.plots.bar(shap_values)
    plt.tight_layout()
    plt.savefig('images/shap_bar_plot_gpu.png', dpi=300)

    plt.figure(figsize=(12,10))
    shap.plots.waterfall(shap_values[idx])
    plt.tight_layout()
    plt.savefig('images/shap_waterfall_plot_gpu.png', dpi=300)

def plot_dist(df: pd.DataFrame):
    """
    Creates and saves distribution and histrogramm plot over all
    features of df.

    Args:
        df (pd.DataFrame): The feature matrix.
    """
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes = axes.flatten()

    for i, var in enumerate(df.keys()):
        sns.histplot(df[var], ax=axes[i], kde=True)
        axes[i].set_title(var)

    plt.tight_layout()
    plt.savefig('images/dist_gpu.png', dpi=300)
    plt.show()


def plot_target_dist(df: pd.DataFrame, original_runtime: pd.Series):
    """
    Creates and saves distribution and histrogramm plot for targets variable 
    before log transformation and after.

    Args:
        df (pd.DataFrame): The feature matrix.
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    sns.histplot(original_runtime, ax=axs[0, 0], kde=True)
    axs[0, 0].set_ylabel("Count")
    axs[0, 0].set_title("Original")

    axs[1, 0].boxplot(original_runtime)
    axs[1, 0].set_ylabel("Values")
    axs[1, 0].set_xticks([1], ['Runtime'])

    sns.histplot(df["Runtime"], ax=axs[0, 1], kde=True)
    axs[0, 1].set_ylabel("Count")
    axs[0, 1].set_title("log-transformiert")

    axs[1, 1].boxplot(df["Runtime"])
    axs[1, 1].set_ylabel("Values")
    axs[1, 1].set_xticks([1], ['Runtime'])

    plt.tight_layout()
    plt.savefig('images/combined_runtime_plots.png', dpi=300)
    plt.show()


df = load_data()


# Preprocessing
mean_values = df.iloc[:, 14:18].mean(axis=1)
df.insert(14, "Runtime", mean_values)
df.drop(['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)'], axis='columns', inplace=True)

print(df.head())
print(df.isnull().sum())
print(df.describe().T)

original_runtime = df["Runtime"]
df["Runtime"] = np.log(df["Runtime"])

# Plots
plot_target_dist(df=df, original_runtime=original_runtime)
plot_dist(df=df)

# Train/Test Split and Model Initialisation/Fitting
X = df.drop(['Runtime'], axis=1) 
y = df['Runtime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2743)

linreg, coef = model(X=X_train, y=y_train)

y_pred = linreg.predict(X_test)
explainer = shap.LinearExplainer(linreg, X_train)
shap_values = explainer(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
train_score = linreg.score(X_train, y_train)
test_score = linreg.score(X_test, y_test)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {mse ** 0.5:.2f}")
print(f"Training Score (R^2): {train_score:.4f}")
print(f"Test Score (R^2): {test_score:.4f}")

print(coef)
plot_corr(X=df)
plot_residuals(y_test=y_test, y_pred=y_pred)
plot_shap(shap_values=shap_values, idx=0)