# streamlit_app.py
import pandas as pd
import streamlit as st
import plotly.express as px
from Generator import train_GAN_test

def main():
    st.title('Cross Sectional Data Generator')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    target_count = st.number_input("Enter the target count of values", min_value=1, value=1000, step=10)
    randomness_degree = st.slider("Degree of randomness", min_value=0.0, max_value=1.0, value=0.5)
    epochs = st.number_input("Enter the number of training epochs", min_value=1, value=5000, step=100)

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write('Original Data')
        st.dataframe(data)

        # Visualize original data
        column_to_plot = st.selectbox("Select a column to visualize", data.columns.tolist())
        if pd.api.types.is_numeric_dtype(data[column_to_plot]):
            plot_types = ["Histogram", "Box Plot", "Violin Plot", "Scatter Plot", "Line Plot"]
        else:
            plot_types = ["Bar Plot", "Pie Chart"]
        plot_type = st.selectbox("Select a type of plot", plot_types)
        
        if plot_type == "Histogram":
            fig = px.histogram(data, x=column_to_plot)
        elif plot_type == "Box Plot":
            fig = px.box(data, y=column_to_plot)
        elif plot_type == "Violin Plot":
            fig = px.violin(data, y=column_to_plot)
        elif plot_type == "Scatter Plot":
            fig = px.scatter(data, x=column_to_plot)
        elif plot_type == "Line Plot":
            fig = px.line(data, x=column_to_plot)
        elif plot_type == "Bar Plot":
            fig = px.bar(data, x=column_to_plot)
        elif plot_type == "Pie Chart":
            fig = px.pie(data, names=column_to_plot)
        
        st.plotly_chart(fig)

        # processed_data = train_GAN(data, target_count, int(randomness_degree*100), epochs)  # Convert randomness_degree to integer
        processed_data = train_GAN_test(data, target_count, int(randomness_degree*100))  # Convert randomness_degree to integer
        st.write('Processed Data')
        st.dataframe(processed_data)

        # Visualize processed data
        column_to_plot = st.selectbox("Select a column to visualize (processed data)", processed_data.columns.tolist())
        if pd.api.types.is_numeric_dtype(processed_data[column_to_plot]):
            plot_types = ["Histogram", "Box Plot", "Violin Plot", "Scatter Plot", "Line Plot"]
        else:
            plot_types = ["Bar Plot", "Pie Chart"]
        plot_type = st.selectbox("Select a type of plot (processed data)", plot_types)
        
        if plot_type == "Histogram":
            fig = px.histogram(processed_data, x=column_to_plot)
        elif plot_type == "Box Plot":
            fig = px.box(processed_data, y=column_to_plot)
        elif plot_type == "Violin Plot":
            fig = px.violin(processed_data, y=column_to_plot)
        elif plot_type == "Scatter Plot":
            fig = px.scatter(processed_data, x=column_to_plot)
        elif plot_type == "Line Plot":
            fig = px.line(processed_data, x=column_to_plot)
        elif plot_type == "Bar Plot":
            fig = px.bar(processed_data, x=column_to_plot)
        elif plot_type == "Pie Chart":
            fig = px.pie(processed_data, names=column_to_plot)
        
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
