import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("CAR_DETAILS.csv")
    return df

df = load_data()

# Preprocess the data
categorical_columns = ['name', 'fuel', 'seller_type', 'transmission', 'owner']
x = df.drop('selling_price', axis=1)
y = df['selling_price']

# Apply one-hot encoding to the categorical columns
preprocessor = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), categorical_columns)],
    remainder='passthrough'
)

@st.cache_data
def preprocess_data(x):
    return preprocessor.fit_transform(x)

x_encoded = preprocess_data(x)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=42)

# Train the model
@st.cache_resource
def train_model(x_train, y_train):
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(x_train, y_train)
    return rf_regressor

rf_regressor = train_model(x_train, y_train)

# Create the Streamlit app
st.title('Car Selling Price Prediction')

# User input for feature values
user_input = {}
for column in x.columns:
    if column in categorical_columns:
        unique_values = df[column].unique()
        user_input[column] = st.selectbox(column, unique_values)
    else:
        user_input[column] = st.number_input(column, value=0)

# Transform user input to one-hot encoding
user_input_encoded = preprocessor.transform(pd.DataFrame(user_input, index=[0]))

# Make predictions using the trained model
prediction = rf_regressor.predict(user_input_encoded)

# Display the prediction
st.subheader('Prediction')
st.write(f'The predicted selling price is: {prediction[0]:.2f}')
