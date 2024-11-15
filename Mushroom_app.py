import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Fetch the dataset
mushroom = fetch_ucirepo(id=73)
X = mushroom.data.features
y = mushroom.data.targets

# Handle missing values
X.loc[:, 'stalk-root'] = X['stalk-root'].fillna('missing')

# Encode categorical features and target variable
encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X.loc[:, column] = le.fit_transform(X[column])
    encoders[column] = le

y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y.values.ravel())

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Feature mappings for human-readable options
feature_mappings = {
    "cap-shape": {"b": "bell", "c": "conical", "x": "convex", "f": "flat", "k": "knobbed", "s": "sunken"},
    "cap-surface": {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"},
    "cap-color": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "r": "green", "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"},
    "bruises": {"t": "bruises", "f": "no bruises"},
    "odor": {"a": "almond", "l": "anise", "c": "creosote", "y": "fishy", "f": "foul", "m": "musty", "n": "none", "p": "pungent", "s": "spicy"},
    "gill-attachment": {"a": "attached", "d": "descending", "f": "free", "n": "notched"},
    "gill-spacing": {"c": "close", "w": "crowded", "d": "distant"},
    "gill-size": {"b": "broad", "n": "narrow"},
    "gill-color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "g": "gray", "r": "green", "o": "orange", "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"},
    "stalk-shape": {"e": "enlarging", "t": "tapering"},
    "stalk-root": {"b": "bulbous", "c": "club", "u": "cup", "e": "equal", "z": "rhizomorphs", "r": "rooted", "?": "missing", "missing": "missing"},  # Added 'missing' entry
    "stalk-surface-above-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-surface-below-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-color-above-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange", "p": "pink", "e": "red", "w": "white", "y": "yellow"},
    "stalk-color-below-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange", "p": "pink", "e": "red", "w": "white", "y": "yellow"},
    "veil-type": {"p": "partial", "u": "universal"},
    "veil-color": {"n": "brown", "o": "orange", "w": "white", "y": "yellow"},
    "ring-number": {"n": "none", "o": "one", "t": "two"},
    "ring-type": {"c": "cobwebby", "e": "evanescent", "f": "flaring", "l": "large", "n": "none", "p": "pendant", "s": "sheathing", "z": "zone"},
    "spore-print-color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "r": "green", "o": "orange", "u": "purple", "w": "white", "y": "yellow"},
    "population": {"a": "abundant", "c": "clustered", "n": "numerous", "s": "scattered", "v": "several", "y": "solitary"},
    "habitat": {"g": "grasses", "l": "leaves", "m": "meadows", "p": "paths", "u": "urban", "w": "waste", "d": "woods"}
}

# Tooltip dictionary for each feature
tooltips = {
    "cap-shape": "The shape of the mushroom cap (e.g., bell, conical, convex).",
    "cap-surface": "The texture of the mushroom cap surface (e.g., fibrous, smooth, scaly).",
    "cap-color": "The color of the mushroom cap.",
    "bruises": "Whether the mushroom bruises easily (yes/no).",
    "odor": "The odor of the mushroom (e.g., almond, anise, foul).",
    "gill-attachment": "How the gills are attached to the stem.",
    "gill-spacing": "The spacing of the gills (close, crowded, distant).",
    "gill-size": "The size of the gills (broad or narrow).",
    "gill-color": "The color of the gills.",
    "stalk-shape": "The shape of the stalk (enlarging or tapering).",
    "stalk-root": "The type of stalk root.",
    "stalk-surface-above-ring": "The surface texture of the stalk above the ring.",
    "stalk-surface-below-ring": "The surface texture of the stalk below the ring.",
    "stalk-color-above-ring": "The color of the stalk above the ring.",
    "stalk-color-below-ring": "The color of the stalk below the ring.",
    "veil-type": "Type of veil (partial or universal).",
    "veil-color": "The color of the veil.",
    "ring-number": "The number of rings on the stalk.",
    "ring-type": "The type of ring on the stalk.",
    "spore-print-color": "The color of the spore print.",
    "population": "The population density of the mushrooms.",
    "habitat": "The habitat where the mushroom grows (e.g., woods, urban)."
}

# Prediction function with handling for unseen labels
def predict_mushroom(features):
    encoded_features = []
    for column, value in zip(X.columns, features):
        # Ensure value is cast to string for comparison
        value = str(value)
        # Check if the value is in the known classes of the encoder
        if value in encoders[column].classes_:
            # Transform the known value
            encoded_value = encoders[column].transform([value])[0]
        else:
            # Handle unseen label by showing a warning and using a default known value
            st.warning(f"'{value}' is not recognized for {column}. Using default value '{encoders[column].classes_[0]}' instead.")
            encoded_value = encoders[column].transform([encoders[column].classes_[0]])[0]
        
        encoded_features.append(encoded_value)
    
    prediction = model.predict([encoded_features])
    return "Edible" if y_encoder.inverse_transform(prediction)[0] == 'e' else "Poisonous"

# Streamlit UI
st.title("Mushroom Classification App")
st.write("Input the characteristics of the mushroom to classify it as edible or poisonous.")

# Collect user input with human-readable options
feature_inputs = []
for column in X.columns:
    # Get the human-readable options
    options = [feature_mappings[column][value] for value in encoders[column].inverse_transform(range(len(encoders[column].classes_)))]
    input_value = st.selectbox(
        f"Select {column}",
        options,
        help=tooltips.get(column, "No tooltip available for this feature")
    )
    # Map back to encoded value for prediction
    encoded_value = [key for key, val in feature_mappings[column].items() if val == input_value][0]
    feature_inputs.append(encoded_value)

# Display user choices in human-readable format
st.write("### Selected Characteristics")
for column, choice in zip(X.columns, feature_inputs):
    st.write(f"**{column}**: {feature_mappings[column][choice]}")

# Prediction button and result
if st.button("Classify"):
    result = predict_mushroom(feature_inputs)
    if result == "Edible":
        st.success(f"The mushroom is likely: **{result}**")
    else:
        st.error(f"The mushroom is likely: **{result}**")




# Evaluate the model on the test set
y_pred = model.predict(X_test)
st.write("### Model Performance")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred, target_names=["Edible", "Poisonous"]))

