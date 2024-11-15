from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Fetch the dataset
mushroom = fetch_ucirepo(id=73)
X = mushroom.data.features
y = mushroom.data.targets

# Handle missing values in 'stalk-root' column
X.loc[:, 'stalk-root'] = X['stalk-root'].fillna('missing')

# Encode categorical features and the target variable
le = LabelEncoder()
for column in X.columns:
    X.loc[:, column] = le.fit_transform(X[column])

# Encode the target variable
y = le.fit_transform(y.values.ravel())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model accuracy:", accuracy_score(y_test, y_pred))



