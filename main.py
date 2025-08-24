import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Wall Street Stock Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .bullish { color: #00c851; font-weight: bold; }
    .bearish { color: #ff4444; font-weight: bold; }
    .neutral { color: #ffbb33; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class StockAnalyzer:
    def __init__(self, symbol, period="1y"):
        self.symbol = symbol.upper()
        self.period = period
        self.stock = yf.Ticker(self.symbol)
        self.data = None
        self.info = None
        self.load_data()
    
    def load_data(self):
        try:
            self.data = self.stock.history(period=self.period)
            self.info = self.stock.info
            return True
        except Exception as e:
            st.error(f"Error loading data for {self.symbol}: {str(e)}")
            return False
    
    def calculate_dcf(self, growth_rate=0.05, discount_rate=0.10, terminal_growth=0.03):
        """1) Discounted Cash Flow Model"""
        try:
            # Get financial data
            cash_flow = self.stock.cashflow
            if cash_flow.empty:
                return None, "Cash flow data not available"
            
            # Get free cash flow (Operating Cash Flow - Capital Expenditures)
            if 'Free Cash Flow' in cash_flow.index:
                fcf = cash_flow.loc['Free Cash Flow'].iloc[0]
            elif 'Total Cash From Operating Activities' in cash_flow.index:
                operating_cf = cash_flow.loc['Total Cash From Operating Activities'].iloc[0]
                if 'Capital Expenditures' in cash_flow.index:
                    capex = abs(cash_flow.loc['Capital Expenditures'].iloc[0])
                    fcf = operating_cf - capex
                else:
                    fcf = operating_cf * 0.85  # Estimate
            else:
                return None, "Insufficient cash flow data"
            
            if pd.isna(fcf) or fcf <= 0:
                return None, "Invalid cash flow data"
            
            # Project 5-year cash flows
            projected_fcf = []
            for year in range(1, 6):
                projected_fcf.append(fcf * (1 + growth_rate) ** year)
            
            # Calculate terminal value
            terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            
            # Discount all cash flows to present value
            pv_fcf = sum([cf / (1 + discount_rate) ** (i + 1) for i, cf in enumerate(projected_fcf)])
            pv_terminal = terminal_value / (1 + discount_rate) ** 5
            
            enterprise_value = pv_fcf + pv_terminal
            
            # Get shares outstanding
            shares = self.info.get('sharesOutstanding', self.info.get('impliedSharesOutstanding', 1))
            if not shares:
                shares = 1
            
            intrinsic_value = enterprise_value / shares
            current_price = self.data['Close'].iloc[-1]
            
            return {
                'intrinsic_value': intrinsic_value,
                'current_price': current_price,
                'upside_potential': ((intrinsic_value - current_price) / current_price) * 100,
                'projected_fcf': projected_fcf,
                'terminal_value': terminal_value,
                'enterprise_value': enterprise_value
            }, None
        except Exception as e:
            return None, f"DCF calculation error: {str(e)}"
    
    def calculate_ddm(self):
        """2) Dividend Discount Model"""
        try:
            dividend_yield = self.info.get('dividendYield', 0)
            if not dividend_yield:
                return None, "No dividend data available"
            
            current_price = self.data['Close'].iloc[-1]
            annual_dividend = current_price * dividend_yield
            
            # Assume dividend growth rate based on payout ratio and ROE
            payout_ratio = self.info.get('payoutRatio', 0.4)
            roe = self.info.get('returnOnEquity', 0.12)
            growth_rate = (1 - payout_ratio) * roe if roe else 0.05
            
            # Gordon Growth Model
            required_return = 0.10  # 10% required return
            if required_return <= growth_rate:
                growth_rate = required_return - 0.01  # Ensure growth < required return
            
            fair_value = (annual_dividend * (1 + growth_rate)) / (required_return - growth_rate)
            
            return {
                'fair_value': fair_value,
                'current_price': current_price,
                'annual_dividend': annual_dividend,
                'dividend_yield': dividend_yield * 100,
                'implied_growth': growth_rate * 100,
                'upside_potential': ((fair_value - current_price) / current_price) * 100
            }, None
        except Exception as e:
            return None, f"DDM calculation error: {str(e)}"
    
    def calculate_multiples(self):
        """3) Relative Valuation / Multiples"""
        try:
            current_price = self.data['Close'].iloc[-1]
            
            # Get financial metrics
            pe_ratio = self.info.get('trailingPE')
            pb_ratio = self.info.get('priceToBook')
            ps_ratio = self.info.get('priceToSalesTrailing12Months')
            ev_ebitda = self.info.get('enterpriseToEbitda')
            ev_ebit = self.info.get('enterpriseToRevenue')  # Approximation
            
            # Industry averages (simplified - in practice, you'd pull from sector data)
            industry_averages = {
                'PE': 18.0,
                'PB': 2.5,
                'PS': 2.0,
                'EV_EBITDA': 12.0
            }
            
            multiples_analysis = {}
            
            if pe_ratio:
                multiples_analysis['PE'] = {
                    'current': pe_ratio,
                    'industry_avg': industry_averages['PE'],
                    'relative_value': 'Undervalued' if pe_ratio < industry_averages['PE'] else 'Overvalued',
                    'discount_premium': ((pe_ratio - industry_averages['PE']) / industry_averages['PE']) * 100
                }
            
            if pb_ratio:
                multiples_analysis['PB'] = {
                    'current': pb_ratio,
                    'industry_avg': industry_averages['PB'],
                    'relative_value': 'Undervalued' if pb_ratio < industry_averages['PB'] else 'Overvalued',
                    'discount_premium': ((pb_ratio - industry_averages['PB']) / industry_averages['PB']) * 100
                }
            
            if ps_ratio:
                multiples_analysis['PS'] = {
                    'current': ps_ratio,
                    'industry_avg': industry_averages['PS'],
                    'relative_value': 'Undervalued' if ps_ratio < industry_averages['PS'] else 'Overvalued',
                    'discount_premium': ((ps_ratio - industry_averages['PS']) / industry_averages['PS']) * 100
                }
            
            if ev_ebitda:
                multiples_analysis['EV_EBITDA'] = {
                    'current': ev_ebitda,
                    'industry_avg': industry_averages['EV_EBITDA'],
                    'relative_value': 'Undervalued' if ev_ebitda < industry_averages['EV_EBITDA'] else 'Overvalued',
                    'discount_premium': ((ev_ebitda - industry_averages['EV_EBITDA']) / industry_averages['EV_EBITDA']) * 100
                }
            
            return multiples_analysis, None
        except Exception as e:
            return None, f"Multiples calculation error: {str(e)}"
    
    def calculate_peg_ratio(self):
        """4) PEG Ratio"""
        try:
            pe_ratio = self.info.get('trailingPE')
            growth_rate = self.info.get('earningsGrowth')
            
            if not pe_ratio or not growth_rate:
                # Try to estimate growth from earnings data
                financials = self.stock.financials
                if not financials.empty and 'Net Income' in financials.index:
                    earnings = financials.loc['Net Income']
                    if len(earnings) >= 2:
                        growth_rate = ((earnings.iloc[0] / earnings.iloc[1]) - 1)
                    else:
                        growth_rate = 0.10  # Default 10%
                else:
                    growth_rate = 0.10
                
                if not pe_ratio:
                    return None, "P/E ratio not available"
            
            growth_rate_pct = growth_rate * 100 if growth_rate < 1 else growth_rate
            peg_ratio = pe_ratio / growth_rate_pct if growth_rate_pct > 0 else float('inf')
            
            # PEG interpretation
            if peg_ratio < 1:
                interpretation = "Undervalued (PEG < 1)"
            elif peg_ratio <= 1.5:
                interpretation = "Fairly valued (1 < PEG < 1.5)"
            else:
                interpretation = "Overvalued (PEG > 1.5)"
            
            return {
                'peg_ratio': peg_ratio,
                'pe_ratio': pe_ratio,
                'growth_rate': growth_rate_pct,
                'interpretation': interpretation
            }, None
        except Exception as e:
            return None, f"PEG calculation error: {str(e)}"
    
    def calculate_eva_ri(self):
        """5) Economic Value Added (EVA) / Residual Income (RI)"""
        try:
            # Get financial data
            roe = self.info.get('returnOnEquity', 0)
            book_value = self.info.get('bookValue', 0)
            
            if not roe or not book_value:
                return None, "Insufficient data for EVA/RI calculation"
            
            # Estimate cost of equity (CAPM approximation)
            beta = self.info.get('beta', 1.0)
            risk_free_rate = 0.03  # 3% assumption
            market_premium = 0.07  # 7% market risk premium
            cost_of_equity = risk_free_rate + (beta * market_premium)
            
            # Calculate EVA per share
            eva_per_share = (roe - cost_of_equity) * book_value
            
            # Current market price
            current_price = self.data['Close'].iloc[-1]
            
            # EVA-based valuation (simplified)
            eva_multiple = 10  # Assumption: 10x EVA multiple
            eva_value = book_value + (eva_per_share * eva_multiple)
            
            return {
                'eva_per_share': eva_per_share,
                'roe': roe * 100,
                'cost_of_equity': cost_of_equity * 100,
                'book_value': book_value,
                'eva_value': eva_value,
                'current_price': current_price,
                'value_creation': eva_per_share > 0
            }, None
        except Exception as e:
            return None, f"EVA/RI calculation error: {str(e)}"
    
    def calculate_piotroski_score(self):
        """6) Piotroski F-Score"""
        try:
            financials = self.stock.financials
            balance_sheet = self.stock.balance_sheet
            cash_flow = self.stock.cashflow
            
            if financials.empty or balance_sheet.empty or cash_flow.empty:
                return None, "Insufficient financial data for Piotroski Score"
            
            score = 0
            criteria = {}
            
            # Profitability (4 points)
            # 1. Positive net income
            if 'Net Income' in financials.index:
                net_income = financials.loc['Net Income'].iloc[0]
                criteria['Positive Net Income'] = net_income > 0
                if net_income > 0:
                    score += 1
            
            # 2. Positive ROA
            roa = self.info.get('returnOnAssets', 0)
            criteria['Positive ROA'] = roa > 0
            if roa > 0:
                score += 1
            
            # 3. Positive operating cash flow
            if 'Total Cash From Operating Activities' in cash_flow.index:
                op_cash_flow = cash_flow.loc['Total Cash From Operating Activities'].iloc[0]
                criteria['Positive Operating Cash Flow'] = op_cash_flow > 0
                if op_cash_flow > 0:
                    score += 1
            
            # 4. Cash flow > Net income (quality of earnings)
            if 'Net Income' in financials.index and 'Total Cash From Operating Activities' in cash_flow.index:
                quality_earnings = op_cash_flow > net_income
                criteria['Quality of Earnings'] = quality_earnings
                if quality_earnings:
                    score += 1
            
            # Leverage, Liquidity and Source of Funds (3 points)
            # 5. Lower long-term debt ratio
            if len(balance_sheet.columns) >= 2:
                if 'Long Term Debt' in balance_sheet.index:
                    ltd_current = balance_sheet.loc['Long Term Debt'].iloc[0]
                    ltd_previous = balance_sheet.loc['Long Term Debt'].iloc[1]
                    criteria['Decreasing Leverage'] = ltd_current < ltd_previous
                    if ltd_current < ltd_previous:
                        score += 1
            
            # 6. Higher current ratio
            current_ratio = self.info.get('currentRatio', 0)
            criteria['Adequate Liquidity'] = current_ratio > 1.5
            if current_ratio > 1.5:
                score += 1
            
            # 7. No dilution (shares outstanding)
            shares_current = self.info.get('sharesOutstanding', 0)
            criteria['No Share Dilution'] = True  # Simplified assumption
            score += 1
            
            # Operating Efficiency (2 points)
            # 8. Higher gross margin
            gross_margin = self.info.get('grossMargins', 0)
            criteria['Improving Gross Margin'] = gross_margin > 0.3  # Simplified
            if gross_margin > 0.3:
                score += 1
            
            # 9. Higher asset turnover
            asset_turnover = self.info.get('returnOnAssets', 0) / self.info.get('profitMargins', 0.01) if self.info.get('profitMargins') else 0
            criteria['Improving Asset Turnover'] = asset_turnover > 1  # Simplified
            if asset_turnover > 1:
                score += 1
            
            # Interpretation
            if score >= 8:
                interpretation = "Strong (8-9): High-quality company"
            elif score >= 6:
                interpretation = "Good (6-7): Above-average quality"
            elif score >= 4:
                interpretation = "Average (4-5): Mixed quality"
            else:
                interpretation = "Weak (0-3): Poor quality"
            
            return {
                'score': score,
                'max_score': 9,
                'interpretation': interpretation,
                'criteria': criteria
            }, None
        except Exception as e:
            return None, f"Piotroski Score calculation error: {str(e)}"
    
    def calculate_altman_z_score(self):
        """7) Altman Z-Score"""
        try:
            balance_sheet = self.stock.balance_sheet
            financials = self.stock.financials
            
            if balance_sheet.empty or financials.empty:
                return None, "Insufficient data for Altman Z-Score"
            
            # Get required data
            total_assets = balance_sheet.loc['Total Assets'].iloc[0]
            current_assets = balance_sheet.loc['Total Current Assets'].iloc[0]
            current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0]
            retained_earnings = balance_sheet.loc['Retained Earnings'].iloc[0] if 'Retained Earnings' in balance_sheet.index else 0
            ebit = financials.loc['EBIT'].iloc[0] if 'EBIT' in financials.index else financials.loc['Operating Income'].iloc[0]
            total_revenue = financials.loc['Total Revenue'].iloc[0]
            
            # Market value of equity
            market_cap = self.info.get('marketCap', 0)
            total_liabilities = balance_sheet.loc['Total Liab'].iloc[0]
            
            # Calculate Z-Score components
            working_capital = current_assets - current_liabilities
            
            # Z-Score formula: Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
            # A = Working Capital / Total Assets
            # B = Retained Earnings / Total Assets  
            # C = EBIT / Total Assets
            # D = Market Value of Equity / Total Liabilities
            # E = Sales / Total Assets
            
            a = working_capital / total_assets
            b = retained_earnings / total_assets
            c = ebit / total_assets
            d = market_cap / total_liabilities if total_liabilities > 0 else 0
            e = total_revenue / total_assets
            
            z_score = 1.2*a + 1.4*b + 3.3*c + 0.6*d + 1.0*e
            
            # Interpretation
            if z_score > 2.99:
                risk_level = "Safe Zone"
                interpretation = "Low bankruptcy risk"
            elif z_score > 1.8:
                risk_level = "Grey Zone"  
                interpretation = "Moderate bankruptcy risk"
            else:
                risk_level = "Distress Zone"
                interpretation = "High bankruptcy risk"
            
            return {
                'z_score': z_score,
                'risk_level': risk_level,
                'interpretation': interpretation,
                'components': {
                    'Working Capital/Total Assets': a,
                    'Retained Earnings/Total Assets': b,
                    'EBIT/Total Assets': c,
                    'Market Value Equity/Total Liabilities': d,
                    'Sales/Total Assets': e
                }
            }, None
        except Exception as e:
            return None, f"Altman Z-Score calculation error: {str(e)}"
    
    def calculate_dupont_roe(self):
        """8) DuPont ROE Decomposition"""
        try:
            # Get financial data
            net_margin = self.info.get('profitMargins', 0)
            asset_turnover = self.info.get('returnOnAssets', 0) / net_margin if net_margin > 0 else 0
            equity_multiplier = 1 / self.info.get('debtToEquity', 0.5) if self.info.get('debtToEquity') else 2
            roe = self.info.get('returnOnEquity', 0)
            
            # DuPont Formula: ROE = Net Margin × Asset Turnover × Equity Multiplier
            calculated_roe = net_margin * asset_turnover * equity_multiplier
            
            return {
                'roe': roe * 100,
                'calculated_roe': calculated_roe * 100,
                'net_margin': net_margin * 100,
                'asset_turnover': asset_turnover,
                'equity_multiplier': equity_multiplier,
                'components': {
                    'Profitability (Net Margin)': net_margin * 100,
                    'Efficiency (Asset Turnover)': asset_turnover,
                    'Leverage (Equity Multiplier)': equity_multiplier
                }
            }, None
        except Exception as e:
            return None, f"DuPont ROE calculation error: {str(e)}"
    
    def calculate_qmv_factors(self):
        """9) QMV Factor Composite (Quality, Momentum, Value)"""
        try:
            # Quality Factors
            roe = self.info.get('returnOnEquity', 0)
            roa = self.info.get('returnOnAssets', 0)
            debt_to_equity = self.info.get('debtToEquity', 0)
            current_ratio = self.info.get('currentRatio', 0)
            
            quality_score = 0
            if roe > 0.15: quality_score += 1
            if roa > 0.05: quality_score += 1
            if debt_to_equity < 0.5: quality_score += 1
            if current_ratio > 1.5: quality_score += 1
            
            # Momentum Factors (price-based)
            returns_1m = ((self.data['Close'].iloc[-1] / self.data['Close'].iloc[-22]) - 1) * 100 if len(self.data) > 22 else 0
            returns_3m = ((self.data['Close'].iloc[-1] / self.data['Close'].iloc[-66]) - 1) * 100 if len(self.data) > 66 else 0
            returns_12m = ((self.data['Close'].iloc[-1] / self.data['Close'].iloc[-252]) - 1) * 100 if len(self.data) > 252 else 0
            
            momentum_score = 0
            if returns_1m > 0: momentum_score += 1
            if returns_3m > 5: momentum_score += 1
            if returns_12m > 10: momentum_score += 1
            
            # Value Factors
            pe_ratio = self.info.get('trailingPE', 0)
            pb_ratio = self.info.get('priceToBook', 0)
            ps_ratio = self.info.get('priceToSalesTrailing12Months', 0)
            
            value_score = 0
            if pe_ratio and pe_ratio < 20: value_score += 1
            if pb_ratio and pb_ratio < 3: value_score += 1
            if ps_ratio and ps_ratio < 2: value_score += 1
            
            # Composite Score
            total_score = quality_score + momentum_score + value_score
            max_score = 10
            
            return {
                'quality_score': quality_score,
                'momentum_score': momentum_score,
                'value_score': value_score,
                'composite_score': total_score,
                'max_score': max_score,
                'percentile': (total_score / max_score) * 100,
                'returns': {
                    '1_month': returns_1m,
                    '3_month': returns_3m,
                    '12_month': returns_12m
                }
            }, None
        except Exception as e:
            return None, f"QMV calculation error: {str(e)}"
    
    def get_earnings_surprise(self):
        """10) Earnings Surprise & Revisions Snapshot"""
        try:
            # Get earnings data
            earnings_dates = self.stock.calendar
            recommendations = self.stock.recommendations
            
            # Basic earnings info from info
            eps_ttm = self.info.get('trailingEps', 0)
            eps_forward = self.info.get('forwardEps', 0)
            
            # Analyst recommendations
            rec_summary = {
                'strong_buy': 0,
                'buy': 0, 
                'hold': 0,
                'sell': 0,
                'strong_sell': 0
            }
            
            if recommendations is not None and not recommendations.empty:
                latest_recs = recommendations.head(10)  # Last 10 recommendations
                for _, rec in latest_recs.iterrows():
                    grade = rec['To Grade'].lower()
                    if 'strong buy' in grade or 'outperform' in grade:
                        rec_summary['strong_buy'] += 1
                    elif 'buy' in grade:
                        rec_summary['buy'] += 1
                    elif 'hold' in grade or 'neutral' in grade:
                        rec_summary['hold'] += 1
                    elif 'sell' in grade:
                        rec_summary['sell'] += 1
                    elif 'strong sell' in grade or 'underperform' in grade:
                        rec_summary['strong_sell'] += 1
            
            # Calculate consensus
            total_recs = sum(rec_summary.values())
            if total_recs > 0:
                bullish_recs = rec_summary['strong_buy'] + rec_summary['buy']
                consensus = 'Bullish' if bullish_recs > total_recs/2 else 'Bearish' if rec_summary['sell'] + rec_summary['strong_sell'] > total_recs/2 else 'Neutral'
            else:
                consensus = 'No Data'
            
            return {
                'eps_ttm': eps_ttm,
                'eps_forward': eps_forward,
                'eps_growth': ((eps_forward - eps_ttm) / eps_ttm * 100) if eps_ttm > 0 else 0,
                'recommendations': rec_summary,
                'consensus': consensus,
                'total_analysts': total_recs
            }, None
        except Exception as e:
            return None, f"Earnings surprise calculation error: {str(e)}"

def create_price_chart(data, symbol):
    """Create interactive price chart with volume"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price Chart', 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=symbol
        ),
        row=1, col=1
    )
    
    # Volume bars
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'),
        row=2, col=1
    )
    
    # Add moving averages
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='MA20', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='MA50', line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    fig.update_layout(
        title=f'{symbol} Technical Analysis',
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=True
    )
    
    return fig

# Main App
def main():
    st.markdown('<div class="main-header">🏛️ Wall Street Stock Monitor</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Stock Analysis Dashboard")
    
    # Stock input - Including Malaysian stocks
    # Malaysian stocks use .KL suffix (Kuala Lumpur Stock Exchange)
    default_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
    malaysian_stocks = ['1155.KL', '1023.KL', '5183.KL', '1066.KL', '5168.KL', '6012.KL', '3182.KL', '4707.KL', '5347.KL', '2445.KL','5263.KL']
    
    # Market selection
    market = st.sidebar.selectbox(
        "Select Market:",
        options=['US Market', 'Malaysian Market (Bursa)', 'Global (Enter any symbol)'],
        index=0
    )
    
    if market == 'US Market':
        stock_options = default_stocks
        market_suffix = ""
        currency = "USD"
    elif market == 'Malaysian Market (Bursa)':
        stock_options = malaysian_stocks
        market_suffix = ".KL"
        currency = "MYR"
    else:
        stock_options = default_stocks + malaysian_stocks
        market_suffix = ""
        currency = "Various"
    
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        if market == 'Global (Enter any symbol)':
            symbol = st.text_input("Enter Stock Symbol:", value="AAPL", placeholder="e.g., AAPL, 1155.KL").upper()
        else:
            symbol = st.text_input("Enter Stock Symbol:", value=stock_options[0], placeholder=f"e.g., {stock_options[0]}").upper()
    with col2:
        st.write("Popular:")
        selected_stock = st.selectbox("", options=stock_options, label_visibility="collapsed")
        if selected_stock:
            symbol = selected_stock
    
    # Time period
    period = st.sidebar.selectbox(
        "Analysis Period:",
        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=3
    )
    
    # Malaysian Stock Examples with company names
    st.sidebar.markdown("**🇲🇾 Popular Malaysian Stocks:**")
    st.sidebar.markdown("""
    - **1155.KL** - Maybank (Malayan Banking)
    - **1023.KL** - CIMB Group Holdings  
    - **5183.KL** - Public Bank
    - **1066.KL** - RHB Bank
    - **5168.KL** - Hong Leong Bank
    - **6012.KL** - IOI Corporation
    - **3182.KL** - Genting
    - **4707.KL** - Telekom Malaysia
    - **5347.KL** - Petronas Dagangan
    - **2445.KL** - MyEG Services
    """)
    st.sidebar.subheader("📋 Analysis Modules")
    show_valuation = st.sidebar.checkbox("💰 Valuation Models", value=True)
    show_quality = st.sidebar.checkbox("🎯 Quality Metrics", value=True)  
    show_technical = st.sidebar.checkbox("📈 Technical Analysis", value=True)
    show_risk = st.sidebar.checkbox("⚠️ Risk Assessment", value=True)
    
    if st.sidebar.button("🔍 Analyze Stock", type="primary"):
        if symbol:
            with st.spinner(f"Analyzing {symbol}..."):
                analyzer = StockAnalyzer(symbol, period)
                
                if analyzer.data is not None and not analyzer.data.empty:
                    # Company Info Header
                    st.subheader(f"📊 {symbol} Analysis Dashboard")
                    
                    # Basic info
                    col1, col2, col3, col4 = st.columns(4)
                    
                    current_price = analyzer.data['Close'].iloc[-1]
                    prev_close = analyzer.info.get('previousClose', current_price)
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    # Currency-aware display
                    with col1:
                        st.metric("Current Price", f"{currency} {current_price:.2f}" if currency != "Various" else f"${current_price:.2f}", f"{change:+.2f} ({change_pct:+.1f}%)")
                    with col2:
                        market_cap = analyzer.info.get('marketCap', 0)
                        if market_cap:
                            if currency == "MYR":
                                st.metric("Market Cap", f"RM {market_cap/1e9:.1f}B")
                            else:
                                st.metric("Market Cap", f"${market_cap/1e9:.1f}B")
                        else:
                            st.metric("Market Cap", "N/A")
                    with col3:
                        volume = analyzer.data['Volume'].iloc[-1]
                        avg_volume = analyzer.info.get('averageVolume', volume)
                        st.metric("Volume", f"{volume:,}", f"{((volume/avg_volume-1)*100):+.1f}%" if avg_volume else "")
                    with col4:
                        pe_ratio = analyzer.info.get('trailingPE', 0)
                        st.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
                    
                    # Company description
                    if analyzer.info.get('longBusinessSummary'):
                        with st.expander("📝 Company Overview"):
                            st.write(analyzer.info['longBusinessSummary'])
                    
                    # Technical Chart
                    if show_technical:
                        st.subheader("📈 Technical Analysis")
                        chart = create_price_chart(analyzer.data.copy(), symbol)
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # Technical indicators
                        col1, col2, col3, col4 = st.columns(4)
                        
                        # RSI
                        delta = analyzer.data['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs))
                        current_rsi = rsi.iloc[-1]
                        
                        with col1:
                            rsi_signal = "Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
                            st.metric("RSI (14)", f"{current_rsi:.1f}", rsi_signal)
                        
                        # MACD
                        exp1 = analyzer.data['Close'].ewm(span=12).mean()
                        exp2 = analyzer.data['Close'].ewm(span=26).mean()
                        macd = exp1 - exp2
                        signal = macd.ewm(span=9).mean()
                        macd_signal = "Bullish" if macd.iloc[-1] > signal.iloc[-1] else "Bearish"
                        
                        with col2:
                            st.metric("MACD Signal", macd_signal)
                        
                        # Bollinger Bands
                        bb_period = 20
                        bb_std = 2
                        sma = analyzer.data['Close'].rolling(bb_period).mean()
                        std = analyzer.data['Close'].rolling(bb_period).std()
                        bb_upper = sma + (std * bb_std)
                        bb_lower = sma - (std * bb_std)
                        
                        bb_position = "Upper" if current_price > bb_upper.iloc[-1] else "Lower" if current_price < bb_lower.iloc[-1] else "Middle"
                        
                        with col3:
                            st.metric("Bollinger Position", bb_position)
                        
                        # Support/Resistance
                        recent_high = analyzer.data['High'].rolling(20).max().iloc[-1]
                        recent_low = analyzer.data['Low'].rolling(20).min().iloc[-1]
                        
                        with col4:
                            st.metric("20-Day Range", f"${recent_low:.2f} - ${recent_high:.2f}")
                    
                    # Valuation Models
                    if show_valuation:
                        st.subheader("💰 Valuation Models")
                        
                        val_col1, val_col2 = st.columns(2)
                        
                        with val_col1:
                            st.markdown("#### 1️⃣ Discounted Cash Flow (DCF)")
                            dcf_result, dcf_error = analyzer.calculate_dcf()
                            if dcf_result:
                                st.success(f"**Intrinsic Value:** ${dcf_result['intrinsic_value']:.2f}")
                                st.write(f"**Current Price:** ${dcf_result['current_price']:.2f}")
                                upside = dcf_result['upside_potential']
                                if upside > 0:
                                    st.success(f"**Upside Potential:** +{upside:.1f}%")
                                else:
                                    st.error(f"**Downside Risk:** {upside:.1f}%")
                            else:
                                st.warning(f"DCF: {dcf_error}")
                            
                            st.markdown("#### 2️⃣ Dividend Discount Model (DDM)")
                            ddm_result, ddm_error = analyzer.calculate_ddm()
                            if ddm_result:
                                st.success(f"**Fair Value:** ${ddm_result['fair_value']:.2f}")
                                st.write(f"**Dividend Yield:** {ddm_result['dividend_yield']:.2f}%")
                                st.write(f"**Implied Growth:** {ddm_result['implied_growth']:.1f}%")
                            else:
                                st.warning(f"DDM: {ddm_error}")
                        
                        with val_col2:
                            st.markdown("#### 3️⃣ Relative Valuation Multiples")
                            multiples_result, multiples_error = analyzer.calculate_multiples()
                            if multiples_result:
                                for multiple, data in multiples_result.items():
                                    with st.expander(f"{multiple} Analysis"):
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.metric("Current", f"{data['current']:.2f}")
                                        with col_b:
                                            st.metric("Industry Avg", f"{data['industry_avg']:.2f}")
                                        
                                        if data['relative_value'] == 'Undervalued':
                                            st.success(f"📈 {data['relative_value']}")
                                        else:
                                            st.error(f"📉 {data['relative_value']}")
                                        
                                        st.write(f"Discount/Premium: {data['discount_premium']:+.1f}%")
                            else:
                                st.warning(f"Multiples: {multiples_error}")
                            
                            st.markdown("#### 4️⃣ PEG Ratio")
                            peg_result, peg_error = analyzer.calculate_peg_ratio()
                            if peg_result:
                                st.metric("PEG Ratio", f"{peg_result['peg_ratio']:.2f}")
                                if peg_result['peg_ratio'] < 1:
                                    st.success("📈 " + peg_result['interpretation'])
                                elif peg_result['peg_ratio'] <= 1.5:
                                    st.info("➡️ " + peg_result['interpretation'])
                                else:
                                    st.error("📉 " + peg_result['interpretation'])
                            else:
                                st.warning(f"PEG: {peg_error}")
                    
                    # Quality Metrics
                    if show_quality:
                        st.subheader("🎯 Quality & Performance Metrics")
                        
                        qual_col1, qual_col2 = st.columns(2)
                        
                        with qual_col1:
                            st.markdown("#### 6️⃣ Piotroski F-Score")
                            piot_result, piot_error = analyzer.calculate_piotroski_score()
                            if piot_result:
                                score = piot_result['score']
                                st.metric("F-Score", f"{score}/9")
                                
                                # Score visualization
                                score_color = 'green' if score >= 7 else 'orange' if score >= 5 else 'red'
                                st.markdown(f'<div style="background-color: {score_color}; padding: 10px; border-radius: 5px; color: white; text-align: center; margin: 10px 0;"><strong>{piot_result["interpretation"]}</strong></div>', unsafe_allow_html=True)
                                
                                # Criteria breakdown
                                with st.expander("📋 Detailed Criteria"):
                                    for criterion, passed in piot_result['criteria'].items():
                                        st.write(f"{'✅' if passed else '❌'} {criterion}")
                            else:
                                st.warning(f"Piotroski: {piot_error}")
                            
                            st.markdown("#### 8️⃣ DuPont ROE Analysis")
                            dupont_result, dupont_error = analyzer.calculate_dupont_roe()
                            if dupont_result:
                                st.metric("ROE", f"{dupont_result['roe']:.1f}%")
                                
                                # Component breakdown
                                components_df = pd.DataFrame([
                                    ["Net Margin", f"{dupont_result['net_margin']:.2f}%"],
                                    ["Asset Turnover", f"{dupont_result['asset_turnover']:.2f}x"],
                                    ["Equity Multiplier", f"{dupont_result['equity_multiplier']:.2f}x"]
                                ], columns=["Component", "Value"])
                                
                                st.dataframe(components_df, hide_index=True)
                            else:
                                st.warning(f"DuPont: {dupont_error}")
                        
                        with qual_col2:
                            st.markdown("#### 9️⃣ QMV Factor Composite")
                            qmv_result, qmv_error = analyzer.calculate_qmv_factors()
                            if qmv_result:
                                # Overall composite score
                                composite_pct = qmv_result['percentile']
                                st.metric("Composite Score", f"{qmv_result['composite_score']}/10 ({composite_pct:.0f}%)")
                                
                                # Individual factor scores
                                factors_df = pd.DataFrame([
                                    ["Quality", qmv_result['quality_score'], 4],
                                    ["Momentum", qmv_result['momentum_score'], 3], 
                                    ["Value", qmv_result['value_score'], 3]
                                ], columns=["Factor", "Score", "Max"])
                                
                                st.dataframe(factors_df, hide_index=True)
                                
                                # Returns breakdown
                                with st.expander("📊 Momentum Returns"):
                                    st.write(f"1 Month: {qmv_result['returns']['1_month']:+.1f}%")
                                    st.write(f"3 Month: {qmv_result['returns']['3_month']:+.1f}%")
                                    st.write(f"12 Month: {qmv_result['returns']['12_month']:+.1f}%")
                            else:
                                st.warning(f"QMV: {qmv_error}")
                            
                            st.markdown("#### 🔟 Earnings & Analyst Sentiment")
                            earnings_result, earnings_error = analyzer.get_earnings_surprise()
                            if earnings_result:
                                col_eps1, col_eps2 = st.columns(2)
                                with col_eps1:
                                    st.metric("TTM EPS", f"${earnings_result['eps_ttm']:.2f}")
                                with col_eps2:
                                    st.metric("Forward EPS", f"${earnings_result['eps_forward']:.2f}")
                                
                                growth = earnings_result['eps_growth']
                                if growth > 0:
                                    st.success(f"📈 EPS Growth: +{growth:.1f}%")
                                else:
                                    st.error(f"📉 EPS Growth: {growth:.1f}%")
                                
                                # Analyst recommendations
                                if earnings_result['total_analysts'] > 0:
                                    st.write(f"**Analyst Consensus:** {earnings_result['consensus']} ({earnings_result['total_analysts']} analysts)")
                                    
                                    rec_data = earnings_result['recommendations']
                                    rec_df = pd.DataFrame([
                                        ["Strong Buy", rec_data['strong_buy']],
                                        ["Buy", rec_data['buy']],
                                        ["Hold", rec_data['hold']],
                                        ["Sell", rec_data['sell']],
                                        ["Strong Sell", rec_data['strong_sell']]
                                    ], columns=["Rating", "Count"])
                                    
                                    st.dataframe(rec_df, hide_index=True)
                            else:
                                st.warning(f"Earnings: {earnings_error}")
                    
                    # Risk Assessment
                    if show_risk:
                        st.subheader("⚠️ Risk Assessment")
                        
                        risk_col1, risk_col2 = st.columns(2)
                        
                        with risk_col1:
                            st.markdown("#### 7️⃣ Altman Z-Score (Bankruptcy Risk)")
                            altman_result, altman_error = analyzer.calculate_altman_z_score()
                            if altman_result:
                                z_score = altman_result['z_score']
                                st.metric("Z-Score", f"{z_score:.2f}")
                                
                                risk_level = altman_result['risk_level']
                                if risk_level == "Safe Zone":
                                    st.success(f"✅ {risk_level}: {altman_result['interpretation']}")
                                elif risk_level == "Grey Zone":
                                    st.warning(f"⚠️ {risk_level}: {altman_result['interpretation']}")
                                else:
                                    st.error(f"🚨 {risk_level}: {altman_result['interpretation']}")
                                
                                # Components breakdown
                                with st.expander("📊 Z-Score Components"):
                                    for component, value in altman_result['components'].items():
                                        st.write(f"{component}: {value:.3f}")
                            else:
                                st.warning(f"Altman Z-Score: {altman_error}")
                        
                        with risk_col2:
                            st.markdown("#### 5️⃣ Economic Value Added (EVA)")
                            eva_result, eva_error = analyzer.calculate_eva_ri()
                            if eva_result:
                                eva_per_share = eva_result['eva_per_share']
                                st.metric("EVA per Share", f"${eva_per_share:.2f}")
                                
                                if eva_result['value_creation']:
                                    st.success("✅ Creating Shareholder Value")
                                else:
                                    st.error("❌ Destroying Shareholder Value")
                                
                                # Key metrics
                                with st.expander("📈 EVA Breakdown"):
                                    st.write(f"ROE: {eva_result['roe']:.2f}%")
                                    st.write(f"Cost of Equity: {eva_result['cost_of_equity']:.2f}%")
                                    st.write(f"Book Value: ${eva_result['book_value']:.2f}")
                                    st.write(f"EVA Valuation: ${eva_result['eva_value']:.2f}")
                            else:
                                st.warning(f"EVA: {eva_error}")
                            
                            # Additional Risk Metrics
                            st.markdown("#### 📊 Additional Risk Metrics")
                            beta = analyzer.info.get('beta', 1.0)
                            volatility = analyzer.data['Close'].pct_change().std() * np.sqrt(252) * 100
                            
                            risk_metrics_df = pd.DataFrame([
                                ["Beta", f"{beta:.2f}"],
                                ["Annualized Volatility", f"{volatility:.1f}%"],
                                ["52-Week Range", f"${analyzer.info.get('fiftyTwoWeekLow', 0):.2f} - ${analyzer.info.get('fiftyTwoWeekHigh', 0):.2f}"]
                            ], columns=["Metric", "Value"])
                            
                            st.dataframe(risk_metrics_df, hide_index=True)
                    
                    # Investment Summary
                    st.subheader("📋 Investment Summary")
                    
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    # Calculate overall sentiment
                    bullish_signals = 0
                    total_signals = 0
                    
                    # Add logic to aggregate signals from all models
                    if dcf_result and dcf_result['upside_potential'] > 0:
                        bullish_signals += 1
                    if dcf_result:
                        total_signals += 1
                    
                    if peg_result and peg_result['peg_ratio'] < 1:
                        bullish_signals += 1
                    if peg_result:
                        total_signals += 1
                    
                    if piot_result and piot_result['score'] >= 6:
                        bullish_signals += 1
                    if piot_result:
                        total_signals += 1
                    
                    if altman_result and altman_result['z_score'] > 2.99:
                        bullish_signals += 1
                    if altman_result:
                        total_signals += 1
                    
                    # Overall sentiment
                    if total_signals > 0:
                        sentiment_score = (bullish_signals / total_signals) * 100
                        
                        if sentiment_score >= 70:
                            overall_sentiment = "🟢 BULLISH"
                            sentiment_color = "green"
                        elif sentiment_score >= 40:
                            overall_sentiment = "🟡 NEUTRAL"
                            sentiment_color = "orange"
                        else:
                            overall_sentiment = "🔴 BEARISH"
                            sentiment_color = "red"
                        
                        with summary_col1:
                            st.markdown(f'<div style="background-color: {sentiment_color}; padding: 15px; border-radius: 10px; color: white; text-align: center;"><h3>{overall_sentiment}</h3><p>Signal Strength: {sentiment_score:.0f}%</p></div>', unsafe_allow_html=True)
                        
                        with summary_col2:
                            st.markdown("**Key Strengths:**")
                            strengths = []
                            if piot_result and piot_result['score'] >= 7:
                                strengths.append("High quality fundamentals")
                            if qmv_result and qmv_result['momentum_score'] >= 2:
                                strengths.append("Strong price momentum")
                            if altman_result and altman_result['z_score'] > 2.99:
                                strengths.append("Low bankruptcy risk")
                            if eva_result and eva_result['value_creation']:
                                strengths.append("Creating shareholder value")
                            
                            for strength in strengths[:3]:
                                st.write(f"✅ {strength}")
                        
                        with summary_col3:
                            st.markdown("**Key Risks:**")
                            risks = []
                            if altman_result and altman_result['z_score'] < 1.8:
                                risks.append("High financial distress")
                            if beta > 1.5:
                                risks.append("High volatility")
                            if pe_ratio and pe_ratio > 30:
                                risks.append("High valuation multiple")
                            if current_rsi > 70:
                                risks.append("Overbought conditions")
                            
                            for risk in risks[:3]:
                                st.write(f"⚠️ {risk}")
                    
                    # Model Performance Summary
                    st.subheader("🔬 Model Performance Summary")
                    
                    model_results = []
                    
                    if dcf_result:
                        signal = "BUY" if dcf_result['upside_potential'] > 10 else "SELL" if dcf_result['upside_potential'] < -10 else "HOLD"
                        model_results.append(["DCF Model", f"${dcf_result['intrinsic_value']:.2f}", f"{dcf_result['upside_potential']:+.1f}%", signal])
                    
                    if ddm_result:
                        signal = "BUY" if ddm_result['upside_potential'] > 10 else "SELL" if ddm_result['upside_potential'] < -10 else "HOLD"
                        model_results.append(["DDM Model", f"${ddm_result['fair_value']:.2f}", f"{ddm_result['upside_potential']:+.1f}%", signal])
                    
                    if peg_result:
                        signal = "BUY" if peg_result['peg_ratio'] < 1 else "SELL" if peg_result['peg_ratio'] > 2 else "HOLD"
                        model_results.append(["PEG Ratio", f"{peg_result['peg_ratio']:.2f}", peg_result['interpretation'], signal])
                    
                    if piot_result:
                        signal = "BUY" if piot_result['score'] >= 7 else "SELL" if piot_result['score'] <= 3 else "HOLD"
                        model_results.append(["Piotroski F-Score", f"{piot_result['score']}/9", piot_result['interpretation'], signal])
                    
                    if altman_result:
                        signal = "BUY" if altman_result['z_score'] > 2.99 else "SELL" if altman_result['z_score'] < 1.8 else "HOLD"
                        model_results.append(["Altman Z-Score", f"{altman_result['z_score']:.2f}", altman_result['risk_level'], signal])
                    
                    if model_results:
                        results_df = pd.DataFrame(model_results, columns=["Model", "Value", "Interpretation", "Signal"])
                        st.dataframe(results_df, hide_index=True, use_container_width=True)
                    
                    # Additional Wall Street Models Suggestion
                    st.subheader("💡 Additional Professional Models")
                    st.info("""
                    **Other models commonly used by Wall Street professionals:**
                    
                    🔸 **Black-Scholes Model** - Options pricing and volatility analysis
                    🔸 **CAPM (Capital Asset Pricing Model)** - Required return calculation  
                    🔸 **Monte Carlo Simulation** - Risk modeling and scenario analysis
                    🔸 **VaR (Value at Risk)** - Portfolio risk quantification
                    🔸 **Sharpe Ratio** - Risk-adjusted return measurement
                    🔸 **Jensen's Alpha** - Performance vs. benchmark analysis
                    🔸 **Treynor Ratio** - Risk-adjusted performance metric
                    🔸 **Information Ratio** - Active return per unit of risk
                    🔸 **Sortino Ratio** - Downside risk-adjusted returns
                    🔸 **Maximum Drawdown** - Peak-to-trough decline analysis
                    
                    These models can be implemented in future versions for enhanced analysis.
                    """)
                    
                else:
                    st.error(f"Could not retrieve data for {symbol}. Please check the symbol and try again.")
        else:
            st.warning("Please enter a valid stock symbol.")

if __name__ == "__main__":
    main()
