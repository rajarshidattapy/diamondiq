# PRD — DiamondIQ

## Explainable Diamond Price Prediction + Price Trend Simulation System

---

# 1. Product Overview

## Product Name

**DiamondIQ — Explainable AI-Powered Diamond Valuation System**

---

## Problem Statement

Most existing diamond price prediction applications only provide a final predicted price without helping users understand *why* that price was generated. This lack of transparency reduces user trust and limits the usefulness of the system in real-world valuation decisions.

Additionally, users cannot analyze how changes in important features like **carat weight** affect the final price, making the system less interactive and less useful for decision support.

The current application already predicts diamond prices using machine learning, but it lacks:

* Explainability of predictions
* Interactive price trend simulation
* Better decision-support insights

This project aims to solve these limitations by upgrading the existing application into a more intelligent and explainable valuation platform.

---

# 2. Objective

Enhance the existing diamond price prediction app by implementing:

### Feature 1:

## Diamond Price Prediction Engine

A robust real-time prediction engine using the trained XGBoost regression model to accurately predict diamond prices based on user inputs.

### Feature 2:

## Price Trend Simulation

An interactive simulation module that allows users to dynamically vary the **carat value** and visualize how the predicted price changes while keeping all other features constant.

This transforms the system from a simple prediction tool into an interactive valuation intelligence platform.

---

# 3. Functional Requirements

---

# Module A — Diamond Price Prediction Engine

## Description

The system should accept diamond characteristics from the user and predict the estimated market price using the trained machine learning model.

---

## Input Features

User should provide:

* carat
* cut
* color
* clarity
* depth
* table
* x
* y
* z

---

## Processing Flow

### Step 1:

Take user input from Streamlit UI

### Step 2:

Pass input through preprocessing pipeline

Includes:

* missing value handling
* categorical encoding
* feature scaling

### Step 3:

Send processed data to trained XGBoost model

### Step 4:

Generate predicted diamond price

### Step 5:

Display final output clearly in UI

Example:

```text
Predicted Diamond Price: ₹85,240
```

---

# Module B — Price Trend Simulation

## Description

The system should allow users to interactively simulate how diamond price changes when only the **carat value** changes.

This helps users understand price sensitivity and valuation trends.

---

## Functional Flow

### Step 1:

After prediction, show:

## Price Trend Simulation

### Step 2:

Add Streamlit slider:

```text
Select Carat Range:
0.5 ct → 2.5 ct
```

Example:

```python
st.slider()
```

### Step 3:

Generate multiple carat values within selected range

Example:

```python
0.5, 0.7, 0.9, 1.1, 1.3 ...
```

### Step 4:

Keep all other features constant

Only modify:

```text
carat
```

### Step 5:

Run prediction for each generated carat value

### Step 6:

Plot dynamic line graph:

```text
X-axis → Carat
Y-axis → Predicted Price
```

### Step 7:

Display graph in Streamlit

This should update instantly when slider changes.

---

# 4. UI Requirements

---

# Main Layout

## Section 1 — User Input Form

Use:

```python
st.form()
```

Fields:

* carat
* cut
* color
* clarity
* depth
* table
* x
* y
* z

Button:

```text
Predict Price
```

---

## Section 2 — Prediction Output

Large highlighted card:

```text
Predicted Price: ₹XX,XXX
```

Should look premium and clear.

---

## Section 3 — Price Trend Simulation

Below prediction:

### Include:

* carat range slider
* line graph visualization
* optional price insight text

Example:

```text
Price increases sharply after 1.5 carat
```

---

# 5. Non-Functional Requirements

System should provide:

### Fast Response

Prediction should complete within 1–2 seconds

### Smooth Visualization

Trend simulation should update without lag

### Scalable Architecture

Existing modular structure should remain intact

### Clean UI/UX

Should feel production-grade, not like a basic ML demo

---

# 6. Existing Project Structure

Use existing architecture:

```text
app.py
src/components/
prediction_pipeline.py
training_pipeline.py
artifacts/
```

Do NOT rebuild project from scratch.

Only extend the existing codebase.

---

# 7. Implementation Constraints

Important:

### Must use existing trained model

Use current:

```text
XGBoost model
```

### Must use current preprocessing pipeline

Do not replace current pipeline

### Must preserve current Streamlit architecture

Only improve it

### Avoid unnecessary complexity

Keep implementation production-clean

---

# 8. Success Criteria

Project is successful if:

* accurate price prediction works
* trend simulation updates correctly
* graph is responsive
* UI looks polished
* feature is presentation-ready for placements/hackathons

---

# 9. Final Expected Result

Instead of:

```text
Basic ML prediction app
```

The final product should feel like:

```text
Production-grade Diamond Valuation Intelligence Platform
```

that predicts price and visually explains pricing behavior through interactive simulation.
