import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf


###############################################################

# Streamlit style, title etc

###############################################################

#st.set_page_config(layout="wide")
st.title("Reverse FCFF")

# Inject CSS overrides
st.markdown("""
    <style>
    html, body, [class*="css"] {
                font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
                font-style: italic !important;
        background-color: #F5F5F5;
        color: #2B2B2B;
    }

    h1, h2, h3, p, span, div, label {
        font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
        font-style: italic !important;
    }

    /* Title */
    h1 {
        color: #E74C3C !important;
    }

    /* Radio buttons styling */
    div.row-widget.stRadio > div {
        gap: 10px;
    }

    div.row-widget.stRadio input[type="radio"] {
        accent-color: #E74C3C !important;
        width: 20px;
        height: 20px;
    }

    div.row-widget.stRadio label {
        font-size: 16px;
    }

    /* Sliders */
    .stSlider > div[data-baseweb="slider"] > div {
        color: #E74C3C !important;
    }

    /* Hide Streamlit footer */
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)



###############################################################

#Store data for the two companies

###############################################################


# Historical Values for MSFT
roic_MSFT = [5.546953187,5.311710224,5.621878916,5.225808229,5.805422943,5.955487071,5.426696553,5.693854569,
        5.937830441,5.475903856,6.131046865,5.49180667,5.800291238,5.532331315,5.296852395,6.090548716,
        6.065807128,5.454773777,6.095507782,5.433733106]

revenue_MSFT = [100000,107894.7368,115789.4737,123684.2105,131578.9474,139473.6842,147368.4211,155263.1579,
           163157.8947,171052.6316,178947.3684,186842.1053,194736.8421,202631.5789,210526.3158,218421.0526,
           226315.7895,234210.5263,242105.2632,250000]

profit_MSFT = [0.871497155,1.254074711,2.291838416,3.211910018,2.598487315,0.432022911,2.327160062,2.029561728,
          0.332523987,2.294174782,0.080403138,1.949679477,1.307938532,1.976721989,0.918487011,0.190879182,
          0.786448517,1.238367894,0.824876251,1.26435397]


#Actual Data from MSFT in Dictionary for easy access
MSFT =  {"Revenue": 245122, 
        "EBIT": 109433,
        "TaxRate": 0.1820,
        "Debt": 97852,
        "Cash": 75543,
        "SharesOut": 7469,
        "LongtermDebtRate": 0.043,
       "Historic ROIC":roic_MSFT,
       "Historic Revenue":revenue_MSFT,
       "Historic Profit": profit_MSFT,
        "Ticker": "MSFT"}

#Actual Data fro NVDA in Dictionary for easy access
NVDA = {"Revenue": 245, 
        "EBIT": 109,
        "TaxRate": 0.1820,
        "Debt": 97,
        "Cash": 75,
        "SharesOut": 7,
        "LongtermDebtRate": 0.043,
        "Ticker": "NVDA"}



#Iterable Dictionary with company name and var for com
companies = {"Microsoft": MSFT,
             "Nvidia": NVDA}


#############################################################

# Build a plotting function to plot estimate against historical share value

#############################################################

def plot_saeulendiagramm(categories, values, title):
    """
    Plottet ein Säulendiagramm mit farblicher Hervorhebung der letzten 10 Werte.
    Verwendet Streamlit zur Anzeige.
    """
    highlight_count = 10
    total_values = len(values)
    
    # Make sure we don't exceed bounds
    highlight_count = min(highlight_count, total_values)
    colors = ['gray'] * (len(values) - highlight_count) + ['orange'] * highlight_count

    # Create figure for Streamlit
    fig, ax = plt.subplots()
    ax.bar(categories, values, color=colors)
    ax.set_title(f" {title} (Estimated)")
    ax.set_xlabel("Year")
    #ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Display in Streamlit
    st.pyplot(fig)

years = list(range(2005,2025+10))



###############################################################

#FCF Calculator function that also calls streamlit and other function

###############################################################


def FCFF_func():

    #set end value for revenue growth to longterm debt rate for the selected company
    revenue_growth_end = current_company["LongtermDebtRate"]
    
    #calcualte decrease in revenue growth
    revenue_subtract = (revenue_growth_start-revenue_growth_end)/5
    
    revenue_growth = []
    
    for i in range(10):
        if i < 5:
            revenue_growth.append(revenue_growth_start)
        else:
            revenue_growth.append(revenue_growth_start-(i-4)*revenue_subtract)
    
    revenue = []
    revenue_start = current_company["Revenue"]
    for i in range(len(revenue_growth)):
        revenue.append(revenue_start*(1+revenue_growth[i]))
        revenue_start = revenue[i]    
    
    #Fill list for operational margin with input margin by looping
    op_margin = []
    for i in range(10):
        op_margin.append(op_margin_input)
    
    #turn revenue, op_margin into array for calculations
    revenue = np.array(revenue)
    op_margin = np.array(op_margin)
    
    #calculate Ebit
    ebit = op_margin * revenue
    
    #calculate after Tax EBit
    ebit_after_tax = ebit * (1 - current_company["TaxRate"])
    
    #Set Input for reinvestment rate !!!USE STREAMLIT FOR USER INPUT!!!
    reinvestment_rate_input = .2
    
    #Fill list for reinvestment rate
    reinvestment_rate = []
    for i in range(10):
        reinvestment_rate.append(reinvestment_rate_input)
    
    #Calculate free cashflow to the firm
    ebit_after_tax = np.array(ebit_after_tax)
    reinvestment_rate = np.array(reinvestment_rate)
    FCFF = ebit_after_tax*(1-reinvestment_rate)
    
    #Calculate Cost of Capital
    
    Capital_Cost_end = 0.084
    Capital_Cost_subtract = (Capital_Cost_start-Capital_Cost_end)/5
    Capital_Cost = []
    for i in range(10):
        if i < 5:
            Capital_Cost.append(Capital_Cost_start)
        else:
            Capital_Cost.append(Capital_Cost_start-(i-4)*Capital_Cost_subtract)
    
    Discount_Factor = []
    for i in range(10):
        x = 1 / (Capital_Cost[i]+1)**(i+1)
        Discount_Factor.append(x)
            
    PV_FCFF = FCFF * Discount_Factor
    NPV_10 = sum(PV_FCFF)
    
    
    ###############################################################
    
    #Model TV Period
    
    ###############################################################
    
    revenue_TV = revenue[-1] * (1 + revenue_growth[-1])
    ebit_TV = revenue_TV * op_margin[-1]
    ebit_after_tax_TV = ebit_TV * (1 - current_company["TaxRate"])
    reinvestment_TV = 61301 #!!!! NEED TO CHECK HIS CALCULATION IF WE NEED TO DO THiS IN CODE OR IF WE CAN SET FIX
    FCFF_TV = ebit_after_tax_TV - reinvestment_TV
    CoC_TV = Capital_Cost[-1]
    growth_TV = revenue_growth_end
    
    TV = FCFF_TV / (CoC_TV - growth_TV)
    NPV_TV = TV * Discount_Factor[-1]
    
    ###############################################################
    
    #Calculating Firm Value, Equity Value and Share Price
    
    ###############################################################
    
    NPV_total = NPV_10 + NPV_TV
    
    Equity_Value = NPV_total - current_company["Debt"] + current_company["Cash"]
    
    PPS = Equity_Value / current_company["SharesOut"]
    
    
    #############################################################
    
    # Reverse FCF Part
    
    #############################################################
    
    reinvestment_amt = []
    for i in range(10):
        x = ebit_after_tax[i] * reinvestment_rate[i]
        reinvestment_amt.append(x)
    
    invested_cap = [352524]
    
    for i in range(10):
        x = invested_cap[i] + reinvestment_amt[i]
        invested_cap.append(x)
    
    avg_invested_cap = []
    
    for i in range(len(invested_cap)-1):
        x = (invested_cap[i] + invested_cap[i+1])/ 2
        avg_invested_cap.append(x)
    
    roic = ebit_after_tax / avg_invested_cap
    
    tax_rate = []
    for i in range(10):
        tax_rate.append(current_company["TaxRate"])

    #############################################################
    
    # Bulding Out Put DataFrames
    
    ############################################################# 

    #First DF for the first 10 years
    FCCF_model = [revenue_growth, revenue, op_margin, ebit, tax_rate, ebit_after_tax, reinvestment_rate, reinvestment_amt, FCFF, Capital_Cost, PV_FCFF]
    return_df = pd.DataFrame(FCCF_model, 
                             columns = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5", "Year 6", "Year 7", "Year 8", "Year 9", "Year 10"], 
                             index = [
        "Umsatzwachstum",          # revenue_growth
        "Umsätze",                 # revenue
        "Operative Marge",         # op_margin
        "Operativer Gewinn (EBIT)",# ebit
        "Steuerquote t",           # tax_rate
        "EBIT(1-t)",               # ebit_after_tax
        "Reinvestitionsquote",     # reinvestment_rate
        "- Reinvestment",          # reinvestment_amt
        "FCFF",                    # FCFF
        "Kapitalkosten",           # Capital_Cost
        "Barwert der FCFF"         # PV_FCFF
    ])

    # Define which rows should be formatted as percentages
    percent_rows = [
        "Umsatzwachstum",
        "Operative Marge",
        "Steuerquote t",
        "Reinvestitionsquote",
        "Kapitalkosten"
    ]
    
    # Apply formatting to percentage rows
    return_df.loc[percent_rows] = return_df.loc[percent_rows].applymap(lambda x: f"{x:.1%}")
    
    # round other rows to percentages
    for row in return_df.index:
        if row not in percent_rows:
            return_df.loc[row] = return_df.loc[row].apply(lambda x: f"{round(x):,}")
    
    # Second Data Frame with the final Scalar Values    
    TV_period_Data = [
        ("I. Summe der Barwert der FCFF in den nächsten 10 Jahren", NPV_10),
        ("FCFF in TV-Periode (TV-FCFF)", FCFF_TV),
        ("Kapitalkosten in TV-Periode (k)", CoC_TV),
        ("Wachstumsrate in TV-Periode (g)", growth_TV),
        ("Terminal Value = TV-FCFF / (k-g)", TV),
        ("II. Barwert des Terminal Values", NPV_TV),
        ("I + II: Summe der Barwerte (=Wert des operativen Vermögens)", NPV_total),
        ("- Fremdkapital", current_company["Debt"]),
        ("+ Cash", current_company["Cash"]),
        ("Wert des Eigenkapitals", Equity_Value),
        ("Anzahl der ausstehenden Aktien", current_company["SharesOut"]),
        ("Fairer, fundamentaler Wert pro Aktie", PPS)
        
    ]

    return_df2 = pd.DataFrame(TV_period_Data, columns=["Label", "Value"]).set_index("Label")



        # Define which rows are percentages
    percent_rows2 = [
        "Kapitalkosten in TV-Periode (k)",
        "Wachstumsrate in TV-Periode (g)"
    ]
    
    # Add 2 decimals for fair share price
    float_2dec_rows = [
        "Fairer, fundamentaler Wert pro Aktie"
    ]
    
    # Apply formatting
    for label in return_df2.index:
        value = return_df2.loc[label, "Value"]
        
        if label in percent_rows2:
            return_df2.loc[label, "Value"] = f"{value:.2%}"
        
        elif label in float_2dec_rows:
            return_df2.loc[label, "Value"] = f"{value:.2f}"
        
        else:
            return_df2.loc[label, "Value"] = f"{round(value):,}"

    #Add the None values for roic and avg_invested_cap so that they have same len as invested_cap
    avg_invested_cap.insert(0, None)
    roic = np.insert(roic, 0, None)

    # Build 3rd Return df
    implizierte_Annahmen = [invested_cap, avg_invested_cap, roic]
    return_df3 = pd.DataFrame(implizierte_Annahmen,  columns = ["Base Year", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5", "Year 6", "Year 7", "Year 8", "Year 9", "Year 10"], index=["Invested Capital", "Average Invested Capital","Return on Invested Capital"])

    # Define percent row
    percent_row3 = "Return on Invested Capital"
    
    # Format rows
    for row in return_df3.index:
        if row == percent_row3:
            # Format as percentage
            return_df3.loc[row] = return_df3.loc[row].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else ""
            )
        else:
            # Format as rounded whole number with dot as thousands separator
            return_df3.loc[row] = return_df3.loc[row].apply(
                lambda x: f"{round(x):,}" if pd.notna(x) else ""
            )

    #
    st.markdown(f"## :green[Fairer Wert pro Aktie:] **{PPS:.2f} €**")

    ticker = current_company["Ticker"]
    stock_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    st.markdown(f"## :green[Aktueller Preis pro Aktie:] **{stock_price:.2f} €**")

    if PPS < stock_price:
        filler = "higher"
    else: 
        filler = "lower"

    st.markdown(f"### The current stock price is {abs(stock_price/PPS-1)*100:.2f}% {filler} then its fair value")

    
    #return the dfs
    st.dataframe(return_df)
    st.dataframe(return_df2, width = 500)
    st.dataframe(return_df3, height = 154)

    #call plot function with correct values to plot Revenue estimation
    revenue_h = pd.Series(current_company["Historic Revenue"])
    revenue_new = pd.concat([revenue_h, pd.Series(revenue)])
    plot_saeulendiagramm(years, revenue_new, "Revenue")

    
    roic_h = pd.Series(current_company["Historic ROIC"])
    roic_new = pd.concat([roic_h, pd.Series(roic[1:])*100])
    plot_saeulendiagramm(years, roic_new, "Return on Invested Capital")
###############################################################

# Build site with streamlit 

###############################################################

#select company via streamlit 
select_company = st.radio("Select a company", list(companies.keys())) 

#set current company variable to selected
current_company = companies[select_company]

#get user input for revenue_growth via slider
revenue_growth_start_percent = st.slider(
    "Initial Revenue Growth Rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=15.0,
    step=0.5,
    format="%.1f%%"
)

#get user input for op margin via slider 
op_margin_input_percent = st.slider(
    "Operating Margin (%)",
    min_value=0.0,
    max_value=100.0,
    value=15.0,
    step=0.5,
    format="%.1f%%"
)

#get user input for CoC via slider 
Capital_Cost_start_percent = st.slider(
    "Initial Cost of Capital (%)",
    min_value=0.0,
    max_value=100.0,
    value=15.0,
    step=0.5,
    format="%.1f%%"
)


# set inputs to the decimal value
op_margin_input = op_margin_input_percent / 100
revenue_growth_start = revenue_growth_start_percent / 100
Capital_Cost_start = Capital_Cost_start_percent / 100


#call functions to rerun script on change
FCFF_func()










