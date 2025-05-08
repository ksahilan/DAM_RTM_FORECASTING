# DAM_RTM_FORECASTING
BTP - 1 project: LSTM model to forecast DAM and RTM prices in India

LSTM models are trained on CSV files to:
1. Take the previous day's MCP to predict the next day's MCP (eg, Wednesday's data to predict Friday's MCPs)
2. Forecast RTM MCP using the last 24 hours or 96-timestep MCP data before that slot's auction

The best trained weights are present in the best_model_weights folder.
Scaler is used to scale the input, and models are trained on scaled inputs (Scaling is done to prevent overfitting).
Codes to use the scaler and to test are given in dam_rtm_forecasting.ipynb.
