import yfinance as yf

tickers = ["MSFT"]
start = "2024-01-01"
end = "2024-06-01"

data = yf.download(tickers, start=start, end=end)
print(data.head())  # If this works, the problem is likely in Dash
