import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import minimize

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Portfolio Optimization Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        dcc.Input(id='tickers', type='text', placeholder='Enter Tickers (comma-separated)', 
                 style={'width': '30%', 'margin': '10px'}),
        dcc.Input(id='start-date', type='text', placeholder='Start Date (YYYY-MM-DD)',
                 style={'width': '20%', 'margin': '10px'}),
        dcc.Input(id='end-date', type='text', placeholder='End Date (YYYY-MM-DD)',
                 style={'width': '20%', 'margin': '10px'}),
        dcc.Input(id='risk-free', type='number', value=0.02, placeholder='Risk-Free Rate',
                 style={'width': '15%', 'margin': '10px'}, step=0.01),
        html.Button('Optimize Portfolio', id='optimize-btn', 
                   style={'margin': '10px', 'padding': '10px'})
    ], style={'textAlign': 'center'}),
    
    dcc.Loading(
        id="loading",
        type="default",
        children=html.Div([
            dcc.Graph(id='efficient-frontier'),
            html.Div(id='portfolio-stats', style={'padding': '20px'}),
            dcc.Graph(id='portfolio-weights')
        ])
    )
])

def calculate_metrics(weights, mu, Sigma, risk_free_rate):
    ret = np.dot(weights, mu)
    vol = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
    sharpe = (ret - risk_free_rate) / vol
    return ret, vol, sharpe

@app.callback(
    [Output('efficient-frontier', 'figure'),
     Output('portfolio-stats', 'children'),
     Output('portfolio-weights', 'figure')],
    [Input('optimize-btn', 'n_clicks')],
    [State('tickers', 'value'),
     State('start-date', 'value'),
     State('end-date', 'value'),
     State('risk-free', 'value')]
)
def optimize_portfolio(n_clicks, tickers, start, end, risk_free):
    print(f"Button Clicked: {n_clicks}")
    print(f"Tickers: {tickers}")
    print(f"Start Date: {start}")
    print(f"End Date: {end}")
    print(f"Risk-Free Rate: {risk_free}")

    if n_clicks is None or not all([tickers, start, end]):
        return go.Figure(), [], go.Figure()


    tickers = [t.strip().upper() for t in tickers.split(',')]
    risk_free = risk_free or 0.02

    try:
        tickers = [tickers] if isinstance(tickers, str) else tickers
        
        print(f"Tickers received: {tickers}")  # Ensure tickers are correctly passed

        # Fetch historical data
        data = yf.download(tickers, start=start, end=end)
        data = data.get('Adj Close', data['Close'])  # Use 'Close' if 'Adj Close' is missing

        returns = data.pct_change().dropna()
        
        # Calculate expected returns and covariance matrix
        mu = returns.mean() * 252
        Sigma = returns.cov() * 252
        
        # Monte Carlo simulation
        n_portfolios = 10000
        results = np.zeros((3, n_portfolios))
        weights = []
        
        for i in range(n_portfolios):
            w = np.random.random(len(tickers))
            w /= w.sum()
            ret, vol, _ = calculate_metrics(w, mu, Sigma, risk_free)
            results[0,i] = ret
            results[1,i] = vol
            results[2,i] = (ret - risk_free) / vol
            weights.append(w)
        
        # Portfolio optimization
        def negative_sharpe(w):
            ret, vol, _ = calculate_metrics(w, mu, Sigma, risk_free)
            return - (ret - risk_free) / vol
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0,1) for _ in range(len(tickers)))
        initial = np.ones(len(tickers)) / len(tickers)
        
        opt_result = minimize(negative_sharpe, initial, 
                            method='SLSQP', bounds=bounds, 
                            constraints=constraints)
        
        opt_ret, opt_vol, opt_sharpe = calculate_metrics(opt_result.x, mu, Sigma, risk_free)
        
        # Create figures
        ef_fig = go.Figure()
        ef_fig.add_trace(go.Scatter(
            x=results[1,:], y=results[0,:],
            mode='markers',
            marker=dict(
                color=results[2,:],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sharpe Ratio')
            ),
            name='Possible Portfolios'
        ))
        ef_fig.add_trace(go.Scatter(
            x=[opt_vol], y=[opt_ret],
            mode='markers',
            marker=dict(color='red', size=15),
            name='Optimal Portfolio'
        ))
        ef_fig.update_layout(
            coloraxis_colorbar=dict(
                title="Sharpe Ratio",
                thickness=15,   # Slimmer color bar
                len=0.75,       # Adjust length
                x=1.05,         # Move slightly right
            ),
        legend=dict(
            x=0.8,         # Move legend to the right
            y=1,           # Adjust legend placement
            bgcolor="rgba(255,255,255,0.5)"  # Semi-transparent background
        ),
        margin=dict(l=40, r=40, t=40, b=40)  # Avoid excess overlap
)

        
        stats = html.Div([
            html.H3("Optimal Portfolio Statistics"),
            html.P(f"Expected Return: {opt_ret:.2%}"),
            html.P(f"Volatility: {opt_vol:.2%}"),
            html.P(f"Sharpe Ratio: {opt_sharpe:.2f}"),
            html.P(f"Risk-Free Rate: {risk_free:.2%}")
        ])
        
        weights_fig = go.Figure(
            data=[go.Pie(
                labels=tickers,
                values=opt_result.x,
                hole=0.4,
                textinfo='percent+label'
            )]
        )
        weights_fig.update_layout(
            title='Optimal Portfolio Weights',
            height=400
        )
        
        return ef_fig, stats, weights_fig
    
    except Exception as e:
        return go.Figure(), html.Div(f"Error: {str(e)}", style={'color': 'red'}), go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)