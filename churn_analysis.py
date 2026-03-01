import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


# load a dataset
df = pd.read_csv("Churn.csv")

print("data load sucsesfully")

print(df.head())

print("\n data shape : ", df.shape)
print("\n colums :", df.columns)

print("\n find nissing value \n")
print(df.isnull().sum())

print("\n tyoe of the data")
print(df.dtypes)


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")
# converting TotalCharges may introduce NaN values (empty strings, etc.)
# drop any rows containing nulls so that downstream ML code doesn't fail
nan_count = df['TotalCharges'].isna().sum()
if nan_count > 0:
    print(
        f"TotalCharges conversion generated {nan_count} NaNs; dropping those rows.")
    df = df.dropna(subset=['TotalCharges'])

# remove identifier column
df.drop("customerID", axis=1, inplace=True)
print("\n Final Dataset Shape : ", df.shape)


plt.figure(figsize=(6, 4))
sns.countplot(x="Churn",  data=df)

plt.title("Customer Churn Distribution")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x="Contract", hue="Churn", data=df)

plt.title("Churn by Contract size")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxenplot(x="Churn", y="MonthlyCharges", data=df)

plt.title("Monthly Charges  VS Contract")
plt.show()


# encode the churn column as binary 1/0
# strip whitespace just in case the values include extra spaces
df["Churn"] = (df["Churn"].astype(str)
                        .str.strip()          # remove whitespace
                        .map({"Yes": 1, "No": 0}))
churn_nan = df["Churn"].isna().sum()
if churn_nan > 0:
    print(
        f"Found {churn_nan} rows where Churn could not be encoded; dropping them.")
    df = df.dropna(subset=["Churn"])

# for remaining object columns, convert other yes/no flags to 1/0
for Columm in df.columns:
    if df[Columm].dtype == "object" and Columm != "Churn":
        df[Columm] = df[Columm].apply(
            lambda x: 1 if str(x).strip() == "Yes" else 0 if str(x).strip() == "No" else x)

# create dummy variables for any remaining categorical columns
_df = pd.get_dummies(df)
# ensure churn column carried over correctly
if "Churn" not in _df.columns:
    _df["Churn"] = df["Churn"]
df = _df

print("\n Data Ready For Machine Learning")
print(df.head())

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Training Data Shape", X_train)
print("Testing Data Shape", X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model Training Complete")


prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print("Model Accuracy : ", accuracy)


cm = confusion_matrix(y_test , prediction)

print("\nCounfution Matrix\n", cm)

print("classification_report")
print(classification_report(y_test,prediction))


plt.figure(figsize=(6,4))
sns.heatmap(cm , annot=True , fmt="d" , cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Counfution matrix")
plt.show()


import joblib

joblib.dump(model , "churn_model.pkl")
print("Model Saved Successfully!")