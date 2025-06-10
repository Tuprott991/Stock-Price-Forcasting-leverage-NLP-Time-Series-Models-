# Stock-Price-Forcasting-leverage-NLP-Time-Series-Models

This project explores whether combining historical stock data with financial news sentiment can improve stock price predictions. Most forecasting models focus only on past trends, but this project looks at how market sentiment captured from news might impact predictions.

Thanks Fintech Student team from UEF (University of Economics and Finance) for providing a robust & quality dataset (Stock data + News sentiment)

## About the Dataset

...

## LSTM with masked attention + finBERT

The model is the combination of LSTM for learning sequential feature, Masked Self-Attention with Q,K,V = lstm_output and news sentiment (Extract by FinBERT model). This Deep learning model pipeline wiil learn both feature from stock time series & news 