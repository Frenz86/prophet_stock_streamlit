import pandas as pd
from matplotlib import pyplot as plt
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import streamlit as st
import yfinance
import plotly.express as px

@st.cache
def load_data():
    components = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return components.drop('SEC filings', axis=1).set_index('Symbol')


#st.cache(ignore_hash=True)
def load_quotes(asset):
    return yfinance.download(asset)


def main():
    components = load_data()
    title = st.empty()
    st.sidebar.title("Options")

  
    #st.markdown("""
    #<style>
    #body {
    #    color: #fff;
    #    background-color: #0A3648;
    #}
    #</style>
    #""", unsafe_allow_html=True) #071433   0A3648

    def label(symbol):
        a = components.loc[symbol]
        return symbol + ' - ' + a.Security

    if st.sidebar.checkbox('View companies list'):
        st.dataframe(components[['Security',
                                 'GICS Sector',
                                 'Date first added',
                                 'Founded']])

    st.sidebar.subheader('Select Companies')
    asset = st.sidebar.selectbox('Click below to select the company from the list',
                                 components.index.sort_values(), index=3,
                                 format_func=label)
    title.title(components.loc[asset].Security)
    if st.sidebar.checkbox('View company info', True):
        st.table(components.loc[asset])
    data0 = load_quotes(asset)
    data = data0.copy().dropna()
    data.index.name = None

    section = st.sidebar.slider('Number of observations', min_value=30,
                        max_value=min([2000, data.shape[0]]),
                        value=500,  step=10)

    data2 = data[-section:]['Adj Close'].to_frame('Price')
    data2['Date'] = data2.index

    fig2 = px.line(data2,x='Date', y="Price",title = 'the Title ')
    fig2.update_xaxes(
        rangeslider_visible= True,
        rangeselector=dict(
                            buttons = list([
                            dict(count = 3,label = '1y',step='year',stepmode = "backward"),
                            dict(count = 9,label = '3y',step='year',stepmode = "backward"),
                            dict(count = 15,label = '5y',step='year',stepmode = "backward"),
                            dict(step= 'all')
                                ])        
                            )
                    )
    st.plotly_chart(fig2)


    data2 = (
        data2['Price']
        .dropna()
        .to_frame()
        .reset_index()
        .rename(columns={"index": "ds", 'Price': "y"})
    )

    model = Prophet()
    model.fit(data2)
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    fig = plot_plotly(model, forecast)
    fig.update_layout(
        title='title', yaxis_title=asset , xaxis_title="Date",
    )
    
    st.write("# Forecast Prices")
    st.plotly_chart(fig)

    ##########################################
    model.plot_components(forecast)
    #plt.title("Global Products")
    plt.legend()
    st.pyplot()


if __name__ == '__main__':
    main()




