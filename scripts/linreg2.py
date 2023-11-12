import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from zipfile import ZipFile
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_data() -> pd.DataFrame:
    url = "https://data.ub.uni-muenchen.de/2/1/miete03.asc"
    df = pd.read_csv(url, sep='\t')
    return df

def model(X, y):
    linreg = LinearRegression()
    linreg.fit(X, y)

    return linreg

def plot_residuals(y_test, y_pred):
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

    plt.savefig('images/residuals2.png', dpi=300)

def plot_corr(X):
    plt.figure(figsize=(12,10))
    sns.heatmap(X.corr(), annot=True)
    plt.savefig('images/corr2.png', dpi=300)

def shap_explainer(linreg, X_train):
    explainer = shap.Explainer(linreg, X_train, algorithm='linear')
    return explainer

def vif(X):
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

df = load_data()

X = df.drop(['nm'], axis=1)
y = df['nm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

vif = vif(X=X_train)
print(vif)

linreg = model(X=X, y=y)
y_pred = linreg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
train_score = linreg.score(X_train, y_train)
test_score = r2_score(y_test, y_pred)

plot_residuals(y_test=y_test, y_pred=y_pred)
plot_corr(X=df)

explainer = shap_explainer(linreg=linreg, X_train=X_train)
shap_values = explainer(X_test)
