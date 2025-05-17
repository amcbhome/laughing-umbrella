# portfolio_analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_statistics(data):
    mean_x = data['X'].mean()
    mean_y = data['Y'].mean()
    std_x = data['X'].std(ddof=0)  # Population standard deviation
    std_y = data['Y'].std(ddof=0)
    correlation = data['X'].corr(data['Y'])
    return mean_x, mean_y, std_x, std_y, correlation

def calculate_portfolio(data, weights):
    mean_x, mean_y, std_x, std_y, corr = calculate_statistics(data)
    results = []

    for i, w_x in enumerate(weights):
        w_y = 1 - w_x
        portfolio_return = (w_x * mean_x) + (w_y * mean_y)
        portfolio_variance = (w_x ** 2 * std_x ** 2) + \
                             (w_y ** 2 * std_y ** 2) + \
                             (2 * w_x * w_y * corr * std_x * std_y)
        portfolio_risk = np.sqrt(portfolio_variance)
        results.append({
            'Portfolio': chr(65 + i),  # A, B, C, D, E, F
            'Weight_X': round(w_x, 2),
            'Weight_Y': round(w_y, 2),
            'Return': round(portfolio_return, 2),
            'Risk (Std Dev)': round(portfolio_risk, 2)
        })

    return pd.DataFrame(results)

def plot_efficient_frontier(results_df):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Ensure correct order of points from A to F
    results_df_sorted = results_df.sort_values('Portfolio')

    x = results_df_sorted['Risk (Std Dev)']
    y = results_df_sorted['Return']

    # Plot points
    ax.scatter(x, y, color='blue', zorder=5)

    # Plot line through the points A to F
    ax.plot(x, y, color='red', linestyle='--', marker='o', zorder=4)

    # Annotate each point with its Portfolio label
    for _, row in results_df_sorted.iterrows():
        ax.annotate(row['Portfolio'],
                    (row['Risk (Std Dev)'], row['Return']),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha='center',
                    fontsize=9,
                    weight='bold')

    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Risk (Standard Deviation)')
    ax.set_ylabel('Expected Return')
    ax.grid(True)
    st.pyplot(fig)

def main():
    st.title("📈 Portfolio Efficient Frontier Calculator")
    st.write("Upload a CSV file with historical data columns named **'X'** and **'Y'**.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            
            if 'X' not in data.columns or 'Y' not in data.columns:
                st.error("CSV must contain 'X' and 'Y' columns.")
                return

            mean_x, mean_y, std_x, std_y, corr = calculate_statistics(data)

            st.subheader("📊 Basic Statistics")
            st.write(f"Mean of X: {mean_x:.2f}")
            st.write(f"Mean of Y: {mean_y:.2f}")
            st.write(f"Population Std Dev of X: {std_x:.2f}")
            st.write(f"Population Std Dev of Y: {std_y:.2f}")
            st.write(f"Pearson Correlation Coefficient: {corr:.2f}")

            weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            results_df = calculate_portfolio(data, weights)

            st.subheader("📌 Efficient Frontier Results")
            st.dataframe(results_df.style.format({
                'Weight_X': '{:.2f}', 
                'Weight_Y': '{:.2f}', 
                'Return': '{:.2f}', 
                'Risk (Std Dev)': '{:.2f}'
            }), use_container_width=True)

            st.subheader("📉 Efficient Frontier Plot")
            plot_efficient_frontier(results_df)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
