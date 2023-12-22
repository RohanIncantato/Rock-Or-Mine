import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Save the model
#filename = 'D:/projects/PracticePRo/trained_model.sav'
#pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# Load the trained model
#filename = 'D:/projects/PracticePRo/trained_model.sav'
#loaded_model = pickle.load(open(filename, 'rb'))

# Streamlit App
def main():
    st.title("Rock vs. Mine Prediction App")
    st.write("Enter the sonar data values for prediction.")

    # User input
    input_data = []
    for i in range(60):
        input_data.append(st.number_input(f"Feature {i+1}:", min_value=0.0, max_value=1.0))

    if st.button("Predict"):
        # Reshape the input data for prediction
        input_data_array = np.asarray(input_data)
        input_data_reshaped = input_data_array.reshape(1, -1)

        # Make prediction
        prediction = loaded_model.predict(input_data_reshaped)

        # Display result
        if prediction[0] == 'R':
            st.success('The object is a Rock.')
        else:
            st.success('The object is a Mine.')

# Run the app
if __name__ == "__main__":
    main()
