from sklearn import datasets
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# creating ordinal categoriy
def categorize_petal_len(x):
    '''
    '''
    if x <= 1.6:
        return 'LOW'
    elif x <= 5.1:
        return 'AVERAGE'
    else:
        return 'HIGH'


# load iris dataset
iris = datasets.load_iris()
iris_df=pd.DataFrame(iris.data)
iris_df['class']=iris.target

iris_df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
iris_df.dropna(how="all", inplace=True) # remove any empty lines
iris_df['petal_len_cats'] = iris_df['petal_len'].apply(categorize_petal_len) # create ordinal type variable

print(iris_df.head())

missing_values = iris_df.isnull().sum()
print("Missing Values:\n", missing_values)

# 2.2 Summary statistics
summary_statistics = iris_df.describe()
print("Summary Statistics:\n", summary_statistics)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.histplot(iris_df['sepal_len'], kde=True)
plt.title('Sepal Length Distribution')

plt.subplot(2, 2, 2)
sns.histplot(iris_df['sepal_wid'], kde=True)
plt.title('Sepal Width Distribution')

plt.subplot(2, 2, 3)
sns.histplot(iris_df['petal_len'], kde=True)
plt.title('Petal Length Distribution')

plt.subplot(2, 2, 4)
sns.histplot(iris_df['petal_wid'], kde=True)
plt.title('Petal Width Distribution')

plt.tight_layout()
plt.savefig("distributions.png")

# class distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=iris_df, x='class')
plt.title('Distribution of Iris Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig("class_distr.png")



correlation_matrix = iris_df[['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']].corr()

# 3. Create a correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig("heatmap.png")





X = iris_df[['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']]
y = iris_df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# Perform one-way ANOVA for each feature
for feature in X.columns:
    f_statistic, p_value = stats.f_oneway(
        *[X[feature][y == cls] for cls in y.unique()]
    )
    print(f"Feature: {feature}")
    print(f"F-statistic: {f_statistic}")
    print(f"P-value: {p_value}")
    if p_value < 0.05:
        print("Result: Significant difference exists between classes.")
    else:
        print("Result: No significant difference between classes.")
    print("\n")



# Pearson Correlation test using Scipy's stats
# corr, p_values = scipy.stats.pearsonr(iris_df['sepal_len'], iris_df['sepal_wid'])
# print(corr, p_values)

# width = 12
# height = 10
# plt.figure(figsize=(width, height))
# sns.regplot(x="sepal_wid", y="class", data=iris_df)
# plt.ylim(0,)
# plt.savefig("linreg_iris.png")