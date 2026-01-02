import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

def find_pivots(df, window=5):
    """
    Finds local highs and lows.
    Returns two lists: lows (date, price), highs (date, price)
    """
    highs = []
    lows = []
    
    for i in range(window, len(df) - window):
        current_low = df['Low'].iloc[i]
        current_high = df['High'].iloc[i]
        
        # Check Low
        is_low = True
        for j in range(i - window, i + window + 1):
            if j == i: continue
            if df['Low'].iloc[j] <= current_low:
                 is_low = False
                 break
        if is_low:
            lows.append((df.index[i], current_low))
            
        # Check High
        is_high = True
        for j in range(i - window, i + window + 1):
            if j == i: continue
            if df['High'].iloc[j] >= current_high:
                is_high = False
                break
        if is_high:
            highs.append((df.index[i], current_high))
            
    return lows, highs

def check_n_pattern(df, lows, highs):
    """
    Checks for N-pattern: Higher Highs and Higher Lows.
    Returns: has_signal (bool), pattern_info (dict or None), msg (str)
    pattern_info: {'l_prev': (date, price), 'h_prev': .., 'l_last': ..}
    """
    if len(lows) < 2 or len(highs) < 1:
        return False, None, "Insufficient data for pattern"

    l_last = lows[-1]      # (Date, Price)
    l_prev = lows[-2]
    
    # 1. Higher Lows (底底高)
    if l_last[1] <= l_prev[1]:
        return False, None, f"無訊號: 未出現底底高 (前底 {l_prev[1]:.2f} >= 近底 {l_last[1]:.2f})"

    # Find the high between these two lows
    relevant_high = None
    # We look for a high that occurred strictly between the two lows in time
    candidates = [h for h in highs if l_prev[0] < h[0] < l_last[0]]
    
    h_prev = None
    if candidates:
        # Usually the highest point between them is the peak
        h_prev = max(candidates, key=lambda x: x[1])
    else:
        # Fallback: look for high before l_last if not strictly between? 
        # But N pattern implies Up-Down-Up, so there SHOULD be a high between.
        # Sometimes the 'high' is before l_prev? No, that's not the N pivot.
        # Pivot structure: Low1 -> High1 -> Low2. Low2 > Low1.
        return False, None, "無訊號: 兩次低點間無顯著高點"

    # 2. Higher Highs (頭頭高)
    # Check if we have broken h_prev
    current_close = df['Close'].iloc[-1]
    
    # Check if there is a newer completed high
    recent_highs = [h for h in highs if h[0] > l_last[0]]
    
    breakout = False
    if recent_highs:
        h_curr = recent_highs[-1]
        if h_curr[1] > h_prev[1]:
            breakout = True
    else:
        if current_close > h_prev[1]:
            breakout = True

    pattern_data = {
        'l_prev': l_prev,
        'h_prev': h_prev,
        'l_last': l_last
    }

    if breakout:
        return True, pattern_data, "✅ 買進訊號 (符合 N 型向上: 底底高 + 頭頭高)"
    else:
        return False, pattern_data, f"觀察中: 底底高成立，但尚未過前高 ({h_prev[1]:.2f})"

def plot_enhanced_candlestick(df, title, pattern_data=None):
    """
    Creates a candlestick chart with MAs and Pattern annotations.
    """
    if df.empty:
        return None
    
    # -- 1. Candlestick --
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='K Line')])

    # -- 2. Moving Averages --
    # Calculate MAs
    ma20 = df['Close'].rolling(window=20).mean()
    ma240 = df['Close'].rolling(window=240).mean() # Approx yearly

    fig.add_trace(go.Scatter(x=df.index, y=ma20, 
                             mode='lines', name='MA20 (月線)', line=dict(color='orange', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=ma240, 
                             mode='lines', name='MA240 (年線)', line=dict(color='blue', width=1.5)))

    # -- 3. Pattern Visualization --
    if pattern_data:
        l_prev = pattern_data['l_prev']
        h_prev = pattern_data['h_prev']
        l_last = pattern_data['l_last']
        
        # Draw lines: l_prev -> h_prev -> l_last
        # And maybe -> current price?
        
        # Coordinates for the N-shape
        x_coords = [l_prev[0], h_prev[0], l_last[0]]
        y_coords = [l_prev[1], h_prev[1], l_last[1]]
        
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode='lines+markers',
            name='N 型形態',
            line=dict(color='purple', width=3),
            marker=dict(size=8, color='purple')
        ))
        
        # Annotations (Text)
        annotations = [
            dict(x=h_prev[0], y=h_prev[1], xref="x", yref="y",
                 text="前高", showarrow=True, arrowhead=1, ax=0, ay=-40, bgcolor="white", bordercolor="purple"),
            dict(x=l_prev[0], y=l_prev[1], xref="x", yref="y",
                 text="前底 (1)", showarrow=True, arrowhead=1, ax=0, ay=40, bgcolor="white"),
             dict(x=l_last[0], y=l_last[1], xref="x", yref="y",
                 text="前底 (2) - 止損點", showarrow=True, arrowhead=1, ax=0, ay=40, bgcolor="#ffcccb", bordercolor="red")
        ]
        fig.update_layout(annotations=annotations)
        
        # Add Stop Loss Line extending to the right
        fig.add_shape(type="line",
            x0=l_last[0], y0=l_last[1], x1=df.index[-1], y1=l_last[1],
            line=dict(color="Red", width=2, dash="dash"),
            name="賣出界線"
        )

    
    return fig

def analyze_stock_data(symbol):
    """
    Fetches data and performs N-pattern analysis for a single stock.
    Returns a dictionary with results and the dataframe.
    """
    full_symbol = f"{symbol}.TW"
    try:
        stock = yf.Ticker(full_symbol)
        df = stock.history(period="max")
        
        if df.empty:
            return None
            
        # Use data for analysis
        df_analysis = df[-300:].copy() 
        lows, highs = find_pivots(df_analysis, window=5)
        
        has_signal, pattern_data, msg = check_n_pattern(df_analysis, lows, highs)
        
        stop_loss = pattern_data['l_last'][1] if pattern_data else None
        current_price = df['Close'].iloc[-1]
        
        return {
            "Symbol": symbol,
            "Price": current_price,
            "Signal": "✅ Buy" if has_signal else ("⚠️ Watch" if "觀察中" in msg else "None"),
            "Message": msg,
            "StopLoss": stop_loss,
            "PatternData": pattern_data,
            "DataFrame": df,
            "HasSignal": has_signal
        }
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None

def main():
    st.set_page_config(page_title="台股個股 K 線檢視器", layout="wide")
    st.title("台股個股 K 線檢視器 (Taiwan Stock Viewer)")
    st.caption("功能：日 K 線 + MA20/MA240 + N 型形態辨識 | 支援批次分析 (輸入多個代碼用逗號分隔)")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        symbol_input = st.text_input("輸入股票代碼 (Ex: 2330, 2317)", value="2330")
        btn = st.button("查看 K 線 & 分析")
        
    if btn:
        if not symbol_input:
            st.error("請輸入股票代碼")
            return
        
        # Parse inputs
        symbols = [s.strip() for s in symbol_input.split(',') if s.strip()]
        
        if not symbols:
             st.error("無效的代碼輸入")
             return

        results = []
        
        # Batch Processing
        if len(symbols) > 1:
            st.write("🔄 正在進行批次分析...")
            progress_bar = st.progress(0)
            
            for i, sym in enumerate(symbols):
                res = analyze_stock_data(sym)
                if res:
                    results.append(res)
                progress_bar.progress((i + 1) / len(symbols))
            
            if not results:
                st.error("無法取得任何輸入股票的資料")
                return
                
            st.session_state['analysis_mode'] = 'batch'
            st.session_state['analysis_results'] = results
            st.session_state['selected_symbol'] = results[0]['Symbol'] # Default to first

        else:
            # Single Stock Processing
            st.write(f"正在取得 {symbols[0]}.TW 的資料...")
            target_result = analyze_stock_data(symbols[0])
            if not target_result:
                st.error("找不到資料，請確認股票代碼是否正確。")
                return
            
            st.session_state['analysis_mode'] = 'single'
            st.session_state['analysis_results'] = [target_result]
            st.session_state['selected_symbol'] = target_result['Symbol']

    # --- Render UI based on Session State ---
    if 'analysis_results' in st.session_state and st.session_state['analysis_results']:
        results = st.session_state['analysis_results']
        mode = st.session_state.get('analysis_mode', 'single')
        
        # If Batch Mode, Show Table & Selector
        if mode == 'batch':
            # Create Summary Table
            summary_data = []
            for r in results:
                summary_data.append({
                    "代碼": r["Symbol"],
                    "現價": f"{r['Price']:.2f}",
                    "訊號": r["Signal"],
                    "止損點 (前低)": f"{r['StopLoss']:.2f}" if r['StopLoss'] else "-",
                    "詳細訊息": r["Message"]
                })
            
            st.subheader("📋 批次分析結果")
            st.dataframe(pd.DataFrame(summary_data))
            
            st.divider()
            st.subheader("📉 個股詳細圖表")
            
            # Use selectbox to update state
            selected_symbol = st.selectbox(
                "選擇要查看的股票", 
                [r["Symbol"] for r in results],
                index=[r["Symbol"] for r in results].index(st.session_state.get('selected_symbol', results[0]['Symbol']))
            )
            st.session_state['selected_symbol'] = selected_symbol # Update state manually if needed, though streamlit handles key='...' usually. 
            # Simple variable assignment works here because rerun uses this value.
        
        else:
            # Single Mode
            selected_symbol = results[0]['Symbol']

        # Find the selected result data
        target_result = next((r for r in results if r["Symbol"] == selected_symbol), None)

        # --- Render Chart for Target Result ---
        if target_result:
            df = target_result["DataFrame"]
            pattern_data = target_result["PatternData"]
            msg = target_result["Message"]
            has_signal = target_result["HasSignal"]
            stop_loss = target_result["StopLoss"]
            symbol = target_result["Symbol"]

            # Display Signal Status Details
            if mode == 'single': 
                st.divider()
                
            if has_signal:
                st.success(f"### {msg}")
            elif pattern_data and "觀察中" in msg:
                st.warning(f"### {msg}")
            else:
                st.info(f"### {msg}")
            
            if stop_loss:
                st.markdown(f"**建議賣出界線 (前低)**: `{stop_loss:.2f}`")

            # --- Visualization ---
            # Show last 3 months
            subset_df = df[-65:] 
            
            fig = plot_enhanced_candlestick(subset_df, f"{symbol} 日 K 線圖 (Daily Chart - 近 3 個月)", pattern_data)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
