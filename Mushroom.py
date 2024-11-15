from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Fetch the dataset
mushroom = fetch_ucirepo(id=73)
X = mushroom.data.features
y = mushroom.data.targets

# Step 2: Handle missing values in 'stalk-root' column
X.loc[:, 'stalk-root'] = X['stalk-root'].fillna('missing')

# Step 3: Encode categorical features and the target variable
le = LabelEncoder()
for column in X.columns:
    X.loc[:, column] = le.fit_transform(X[column])

# Encode the target variable
y = le.fit_transform(y.values.ravel())

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("Model accuracy:", accuracy_score(y_test, y_pred))



