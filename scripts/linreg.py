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
from sklearn.inspection import permutation_importance

matplotlib.use('Agg')

def load_data() -> pd.DataFrame:
    """
    Loads and returns the dataset from the given URL as a Pandas DataFrame.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    url = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"

    r = requests.get(url)

    if r.ok:
        with zipfile.ZipFile(BytesIO(r.content)) as thezip:
            with thezip.open("Concrete_Data.xls") as thefile:
                return pd.read_excel(thefile, header=0)
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

    plt.savefig('images/residuals.png', dpi=300)

def plot_corr(df: pd.DataFrame) -> None:
    """
    Creates and saves a correlation heatmap plot.

    Args:
        df (pd.DataFrame): The feature matrix.
    """
    plt.figure(figsize=(12,10))
    sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", fmt=".2f")
    plt.savefig('images/corr.png', dpi=300)

def plot_shap(shap_values: shap.Explanation, idx: int, model: LinearRegression, X_test: pd.DataFrame) -> None:
    """
    Creates and saves SHAP beeswarm, bar and waterfall plots.

    Args:
        shap_values (shap.Explanation): SHAP values.
        idx (int): Index for the SHAP waterfall plot.
        model (LinearRegression): The Linear Regression model.
        X_test (pd.DataFrame): The dataset for partial dependence plot.
    """
    plt.figure(figsize=(12,10))
    shap.plots.beeswarm(shap_values)
    plt.tight_layout()
    plt.savefig('images/shap_beeswarm_plot.png', dpi=300)

    plt.figure(figsize=(12,10))
    shap.plots.bar(shap_values)
    plt.tight_layout()
    plt.savefig('images/shap_bar_plot.png', dpi=300)

    plt.figure(figsize=(12,10))
    shap.plots.waterfall(shap_values[idx])
    plt.tight_layout()
    plt.savefig('images/shap_waterfall_plot.png', dpi=300)

    plt.figure(figsize=(12,10))
    shap.plots.partial_dependence("cement", model.predict, X_test, 
                                  model_expected_value=True, 
                                  feature_expected_value=True,
                                  ice=False, 
                                  shap_values=shap_values[idx:idx+1,:])
    plt.tight_layout()
    plt.savefig('images/shap_dependence_plot.png', dpi=300)


def plot_dist(df: pd.DataFrame):
    """
    Creates and saves distribution and histrogramm plot over all
    features of df.

    Args:
        df (pd.DataFrame): The feature matrix.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i, var in enumerate(df.keys()):
        sns.histplot(df[var], ax=axes[i], kde=True)
        axes[i].set_title(var)

    plt.tight_layout()
    plt.savefig('images/dist.png', dpi=300)
    plt.show()

def plot_coef(coef: pd.DataFrame):
    """
    Plot the coefficients of a linear model as a horizontal bar chart.
    
    Args:
        coef (pd.DataFrame): A DataFrame containing the model's coefficients, 
                         with feature names as the index and a 'Coefficients' column.
    """
    coef = coef.drop('Intercept', errors='ignore')
    coef = coef.sort_values(by='Coefficients', ascending=True)
    plt.figure(figsize=(12, 10))
    plt.barh(coef.index, coef['Coefficients'])
    plt.axvline(0)
    plt.xlabel('Coefficients')
    plt.ylabel('Features')
    plt.savefig('images/coef.png', dpi=300)
    plt.show()

def plot_box(df: pd.DataFrame, df2: pd.DataFrame):
    """
    Plot the distribution of two dataframes as boxplot.
    
    Args:
        df (pd.DataFrame): first dataframe.
        df2 (pd.Dataframe): second dataframe.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    axs[0].boxplot(df2.values, vert=False)
    axs[0].set_title('original')
    axs[0].set_yticklabels(df2.columns)
    
    axs[1].boxplot(df.values, vert=False)
    axs[1].set_title('log-transformed')
    axs[1].set_yticklabels(df.columns)
    
    plt.tight_layout()
    plt.savefig('images/boxplot.png', dpi=300)
    plt.show()

def plot_permutation_importance(perm_importance_train: permutation_importance, 
                                perm_importance_test: permutation_importance,
                                columns: np.ndarray):
    """
    Plots the permutation importance of features for both training and test datasets.

    Args:
        perm_importance_train (object): A fitted PermutationImportance instance for the training dataset.
                                    It contains importances_mean attribute, representing the decrease in 
                                    model score when a feature is randomly shuffled.
        perm_importance_test (object): A fitted PermutationImportance instance for the test dataset, 
                                   similar to perm_importance_train.
        columns (list or array): An array or list of feature names corresponding to the indices in 
                             the perm_importance objects.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    sorted_idx_train = perm_importance_train.importances_mean.argsort()
    axs[0].barh(range(len(sorted_idx_train)), perm_importance_train.importances_mean[sorted_idx_train], align='center')
    axs[0].set_title('train')
    axs[0].set_xlabel('Change in MSE')
    axs[0].set_yticks(range(len(sorted_idx_train)), np.array(columns)[sorted_idx_train])
    
    sorted_idx_test = perm_importance_test.importances_mean.argsort()
    axs[1].barh(range(len(sorted_idx_test)), perm_importance_test.importances_mean[sorted_idx_test], align='center')
    axs[1].set_title('test')
    axs[1].set_xlabel('Change in MSE')
    axs[1].set_yticks(range(len(sorted_idx_test)), np.array(columns)[sorted_idx_test])

    plt.tight_layout()
    plt.savefig('images/permutation_importance.png', dpi=300)
    plt.show()

df = load_data()

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
        'Concrete compressive strength(MPa, megapascals) ': 'strength'
    }
)
df = df.drop_duplicates()
df_original = df.copy()

print(df.head())
print(df.isnull().sum())
print(df.describe().T)

for column in df.columns:
    df[column] += 1
    df[column] = np.log(df[column])

plot_dist(df=df)
plot_box(df=df, df2=df_original)

X = df.drop(['strength'], axis=1) 
y = df['strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

linreg, coef = model(X=X_train, y=y_train)
y_pred = linreg.predict(X_test)

perm_importance_train = permutation_importance(linreg, X_train, y_train)
perm_importance_test = permutation_importance(linreg, X_test, y_test)
plot_permutation_importance(perm_importance_train, perm_importance_test, X.columns)

explainer = shap.Explainer(linreg, X_train)
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

plot_coef(coef=coef)
print(coef)
plot_corr(df=df)
plot_residuals(y_test=y_test, y_pred=y_pred)
plot_shap(shap_values=shap_values, idx=0, model=linreg, X_test=X_test)