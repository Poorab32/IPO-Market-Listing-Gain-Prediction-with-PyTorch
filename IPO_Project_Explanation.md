# 📈 IPO Prediction Project: Complete Explanation Guide

This guide is designed to help you explain the IPO Listing Gain Predictor project to your professor and share the technical details with your teammates. It covers the architecture, the machine learning model, and all the financial/technical terminology used.

---

## 🚀 1. Project Overview

**What does this project do?**
This project is an **AI-powered Web Application** that predicts the "Listing Gain" of an Initial Public Offering (IPO) in the Indian Stock Market. When a company goes public, it offers shares at an "Offer Price." On the day those shares actually start trading on the stock exchange, they open at a "List Price." The difference between these two is the Listing Gain (or loss). 

Our application uses historical data from 561 Indian IPOs (from 2010 to 2025) to predict what this listing gain will be based on factors like issue size, offer price, and how much different categories of investors subscribed to the IPO.

---

## 💻 2. Technical Stack (How the website works)

You can divide the project into three main technical pillars:

1. **Frontend / UI (Streamlit):** We used Streamlit, a Python framework, to build the user interface. It allows us to turn Python scripts into interactive web applications without needing HTML/JavaScript. We added custom CSS (styling) to give it a premium, modern "dark mode" look with glassmorphism effects.
2. **Data & Visualizations (Pandas & Plotly):** The historical data (`Initial Public Offering.xlsx`) is loaded and cleaned using the `pandas` library. The interactive charts (bar charts, scatter plots, gauge meters) are built using `Plotly`.
3. **AI / Machine Learning Backend (PyTorch & Scikit-learn):** The core intelligence of the app is a Deep Learning model built using `PyTorch`. We also use `joblib` and `scikit-learn` to load our data scaler.

---

## 🧠 3. Machine Learning Terms Explained

When explaining the AI part to your professor, use these terms:

*   **Neural Network / MLP (Multilayer Perceptron):** Our model is an MLP, a type of artificial neural network. It consists of:
    *   **Input Layer:** Takes the 9 features (Issue Size, Subscriptions, Offer Price, etc.).
    *   **Hidden Layers:** Two layers (with 64 and 32 "neurons" or nodes) that process the data and find complex, non-linear patterns. Activation function used is ReLU (Rectified Linear Unit) which helps the model learn complex relationships.
    *   **Output Layer:** 1 node that outputs the final predicted listing gain percentage.
*   **StandardScaler:** Before feeding data into the neural network, all numbers must be on a similar scale. The `scaler.pkl` file ensures that a huge number like "Issue Size: 4000 crores" doesn't overpower a small number like "RII Subscription: 2x". It converts everything to a standardized scale.
*   **Epochs & Optimizer:** During training, the model went through the data multiple times (Epochs = 100). We used the **Adam Optimizer**, which mathematically adjusts the weights of the neurons to minimize errors over time.
*   **MSE Loss (Mean Squared Error):** This is how the model learned. It made a prediction, checked how far off it was from the actual answer, squared that error, and tried to make this error as small as possible in the next epoch.
*   **Metrics on Analytics Page:**
    *   **RMSE (Root Mean Squared Error):** Tells us the average error of our predictions in the same units as our target (percentage points). Lower is better.
    *   **MAE (Mean Absolute Error):** The absolute average difference between predicted and actual listing gain.
    *   **R² Score (R-Squared):** Measures how well the model explains the variance in the data. A score closer to 1.0 means the model is highly accurate. 

---

## 📊 4. IPO & Stock Market Terms Explained

These are the financial terms your app takes as inputs. It is crucial to understand what they mean:

*   **IPO (Initial Public Offering):** The process by which a private company offers its shares to the public for the first time to raise capital.
*   **Issue Size (₹ Crores):** The total monetary value of shares the company is trying to sell. (e.g., A 500 crore issue size means the company is raising ₹500 crores).
*   **Offer Price (₹):** The fixed price at which the company is selling its shares to investors during the IPO period.
*   **List Price (₹):** The price at which the shares actually open on the stock market on listing day. It is decided by market demand and supply.
*   **Listing Gain / Loss (%):** The percentage difference between the Offer Price and the List Price. 
    *   *Example: Offer price = ₹100, List price = ₹120. Listing Gain = 20%.*
*   **CMP (Current Market Price):** The price the stock is trading at right now on the exchange (BSE/NSE), long after the listing day.
*   **Subscription (x):** Shows demand. If a company offers 1 million shares, but investors apply for 10 million shares, the IPO is "10x Oversubscribed." High oversubscription usually implies a higher listing gain.
    *   **QIB (Qualified Institutional Buyers):** Large institutional investors like mutual funds, banks, and insurance companies. High QIB indicates "smart money" trusts the company.
    *   **HNI (High Net-worth Individuals) / NII (Non-Institutional Investors):** Wealthy individuals who invest large amounts (typically over ₹2 lakhs in India).
    *   **RII (Retail Individual Investors):** Everyday, small-scale investors who invest up to ₹2 lakhs. 
    *   **Total Subscription:** The combined overall demand for the IPO across all categories.

---

## 🗣️ 5. How to present this to the Professor

When giving your presentation or demonstrating the project, follow this flow:

1.  **The Problem:** "Predicting IPO listing gains is difficult because it relies on human psychology, hype, and complex financial metrics."
2.  **Our Solution:** "We built a PyTorch-based Deep Learning model that analyzes historical subscription patterns, issue sizes, and pricing to predict the listing gain."
3.  **App Demonstration:** 
    *   Open the app (Page 1: Predictor). Run a dummy prediction. Emphasize the "Similar Historical IPOs" feature to prove the model's logic is grounded in history.
    *   Go to Page 2 (Dashboard). Show the historical trends. "We aren't just making predictions; we've built a data exploration tool for evaluating 561 past IPOs natively inside the app."
    *   Go to Page 3 (Analytics). This is the "Technical flex" page. Show the professor the correlation heatmaps and the model performance metrics (RMSE/R²). Mention the neural network's architecture (9 → 64 → 32 → 1).
4.  **Conclusion:** "By combining financial domain knowledge with Deep Learning and a modern Streamlit frontend, we created a comprehensive, end-to-end AI product."

---
*Good luck with your presentation!*
