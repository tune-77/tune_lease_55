# Memory - Long Term

## Projects
### 温水式リース審査AI (Warm Water Style Lease Screening AI)
- **Status**: Production Ready (Streamlit app)
- **Current Version**: `lease_logic_sumaho3.py` (2026-02-10 Fix: Indentation & Variable Scope repaired)
- **External Access**: `https://lora-gyrational-trebly.ngrok-free.dev` (via ngrok)
- **Key Features**:
  - **Multi-Model Scoring**: Automatic model selection (Service, Manufacturing, Transport, Overall) with CSV-loaded coefficients.
  - **Visualization**: Radar Chart, Positioning Scatter, BEP Graph.
  - **Self-Improvement**: Coefficient Analysis Mode (Logistic Regression on saved logs).
  - **Yield Prediction**: Regression model with market rate adjustment (Base date: 2025-03).
  - **UI Optimization**: Smartphone-friendly layout (fewer columns, larger inputs).
  - **AI Debate Mode**: "Pro" vs "Con" agents (Qwen2.5) debating deal risks.

## Technical Notes
- **Active Script**: `lease_logic_sumaho3.py` (Replaced `lease_logic.py` as the main driver).
- **Unit Handling**:
  - **Logarithmic Terms** (Sales, Credit): `np.log1p(Thousands of Yen)`.
  - **Linear Terms** (Profits, Assets): Scaled to Millions (`/1000`) for scoring model matching.
- **Ratios**: Calculated using raw "Thousands" values for precision.
- **Safety**: `safe_sigmoid` implemented.

## Preferences
- **User**: Kobayashi
- **Persona**: "Lease Pro" (Senior Officer).
- **Design Policy**:
  - **Adversarial Evaluation**: Use debate to uncover hidden risks.
  - **Visual Evidence**: Use charts (Radar, Scatter) to persuade.
  - **Mobile First**: Optimize for smartphone usage via Streamlit adjustments (Flet was discarded).
