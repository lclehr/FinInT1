import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

###############################################################

# Streamlit style, title etc

###############################################################

#st.set_page_config(layout="wide")
st.title("Reverse FCFF")

# Inject CSS overrides to get colors of our favourite private bank ;)
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


# Historical Values for MSFT and NVDA

revenue_MSFT = [
    93580,
    91154,
    96571,
    110360,
    125843,
    143015,
    168088,
    198270,
    211915,
    245122
]

revenue_NVDA = [
    5010,
    6910,
    9714,
    11716,
    10918,
    16675,
    26914,
    26974,
    60922,
    130497
]

# EBIT-Werte für MSFT (30.06.x)
msft_ebit = [
    18161,
    26078,
    29025,
    35058,
    42959,
    52959,
    69916,
    83383,
    88523,
    109433
]

# EBIT-Werte für NVDA (Ende Januar x)
nvda_ebit = [
    747,
    1934,
    3210,
    3804,
    2846,
    4532,
    10041,
    4224,
    32972,
    81453
]
#ROIC for MSFT and NVDA
msft_roic = [
    10.2488,
    15.7513,
    14.6551,
    8.8075,
    21.4688,
    22.9883,
    28.8729,
    31.7364,
    28.2877,
    28.7142
]

nvda_roic = [
    10.8561,
    23.8825,
    33.6136,
    39.4485,
    21.2026,
    23.1672,
    31.9539,
    12.7490,
    72.3859,
    109.4474
]


#Current Data from MSFT in Dictionary for easy access
MSFT =  {"Revenue": 245122, 
        "EBIT": 109433,
        "TaxRate": 0.1820,
        "Debt": 97852,
        "Cash": 75543,
        "SharesOut": 7469,
        "Invested Capital": 352524,
        "Sales to Capital": 2.00,
       "Historic ROIC":msft_roic,
       "Historic Revenue":revenue_MSFT,
       "Historic EBIT": msft_ebit,
        "Ticker": "MSFT"}

#Current Data fro NVDA in Dictionary for easy access. Sources Bloomberg and Annual Statement
NVDA = {"Revenue": 130497, 
        "EBIT": 75605,
        "TaxRate": 0.1326,
        "Debt": 10270,
        "Cash": 43210,
        "SharesOut": 24477,
        "Sales to Capital": 2.77,
        "Invested Capital": 80385,
        "Historic ROIC":nvda_roic,
        "Historic Revenue":revenue_NVDA,
        "Historic EBIT": nvda_ebit,
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
    values = values / 1000

    highlight_count = min(highlight_count, total_values)
    colors = ['green'] * (len(values) - highlight_count) + ['red'] * highlight_count

    fig, ax = plt.subplots()
    ax.bar(categories, values, color=colors)
    ax.tick_params(axis='both', labelsize=9)
    if title == "ROIC":
        formatter = FuncFormatter(lambda x, _: f'{x*100000:.0f}%')
        ax.yaxis.set_major_formatter(formatter)
    # Set x-ticks and labels for all categories (years)
    ax.set_xticks(categories)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
    ax.set_title(f"{title} (estimated)", fontsize=15, fontweight='normal', pad=10)
    ax.set_ylabel(f"{title}", fontsize=12, fontweight='normal')
    plt.tight_layout()
    green_patch = mpatches.Patch(color='green', label='Historical Values')
    red_patch = mpatches.Patch(color='red', label='Estimated Values')
    ax.legend(handles=[green_patch, red_patch], fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    st.pyplot(fig)

years = list(range(2005,2025+10))



###############################################################

#FCF Calculator function that also calls streamlit and other function

###############################################################


def FCFF_func():
    
    #Revenue Calculations
    #######################################
   
    #initialize lists related to revenue
    revenue = []
    revenue_growth_amt = []
    revenue_growth = []

    #calcualte decrease in revenue growth for years 6 to 10
    revenue_subtract = (revenue_growth_start-revenue_growth_end)/5
    
    #fill revenue growth list
    for i in range(10):
        if i < 5:
            revenue_growth.append(revenue_growth_start)
        else:
            revenue_growth.append(revenue_growth_start-(i-4)*revenue_subtract)
    
    
    revenue_start = current_company["Revenue"]
    for i in range(len(revenue_growth)):
        revenue.append(revenue_start*(1+revenue_growth[i]))
        revenue_start = revenue[i]    
        revenue_growth_amt.append(revenue_start*(1+revenue_growth[i])-revenue_start)
    
    
    
    #Fill list for operational margin with input margin by looping
    op_margin = []
    for i in range(10):
        op_margin.append(op_margin_input)
    
    #turn revenue, op_margin into array for calculations
    revenue = np.array(revenue)
    op_margin = np.array(op_margin)
    revenue_growth_amt = np.array(revenue_growth_amt)
    
    #calculate Ebit
    ebit = op_margin * revenue
    
    #calculate after Tax EBit
    ebit_after_tax = ebit * (1 - current_company["TaxRate"])
    
    #Reinvestment Calculations
    ####################################
    reinvestment_amt = []
    sales_to_capital = []
    for i in range(10):
        sales_to_capital.append(current_company["Sales to Capital"])
    
    reinvestment_amt = np.array(reinvestment_amt)
    reinvestment_amt = revenue_growth_amt / sales_to_capital
    
    #Calculate free cashflow to the firm
    ebit_after_tax = np.array(ebit_after_tax)
    FCFF = ebit_after_tax - reinvestment_amt
    
    #Cost of Capital Calculations
    ####################################
    Capital_Cost = []
    Discount_Factor = []

    #linear interpolation between the different waccs
    Capital_Cost_subtract = (Capital_Cost_start-Capital_Cost_end)/5

    #get CoC per year
    for i in range(10):
        if i < 5:
            Capital_Cost.append(Capital_Cost_start)
        else:
            Capital_Cost.append(Capital_Cost_start-(i-4)*Capital_Cost_subtract)
    
    # get Discount Factor
    for i in range(10):
        x = 1 / (Capital_Cost[i]+1)**(i+1)
        Discount_Factor.append(x)


    #Use DCF to get NPV
    PV_FCFF = FCFF * Discount_Factor
    NPV_10 = sum(PV_FCFF)
    
    
    ###############################################################
    
    #Model TV Period
    
    ###############################################################
    
    growth_TV = revenue_growth_end
    revenue_TV = revenue[-1] * (1 + revenue_growth_end)
    ebit_TV = revenue_TV * op_margin[-1]
    ebit_after_tax_TV = ebit_TV * (1 - current_company["TaxRate"])


    #calculate reinvestment according to damodaran, using growth and ROIC in TV as inputs
    reinvestment_TV = growth_TV / TV_ROIC * ebit_after_tax_TV 

    #From FCFF TV to NPV of TV
    FCFF_TV = ebit_after_tax_TV - reinvestment_TV
    CoC_TV = Capital_Cost[-1]
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

    #Use invested cap, tax rate and average invested cap to get the implicit ROIC assumption
    invested_cap = []
    avg_invested_cap = []
    tax_rate = []
    
    invested_cap.append(current_company["Invested Capital"])
    
    for i in range(10):
        x = invested_cap[i] + reinvestment_amt[i]
        invested_cap.append(x)
    
    for i in range(len(invested_cap)-1):
        x = (invested_cap[i] + invested_cap[i+1])/ 2
        avg_invested_cap.append(x)
    
    roic = ebit_after_tax / avg_invested_cap
    
    for i in range(10):
        tax_rate.append(current_company["TaxRate"])

    #############################################################
    
    # Bulding Out-Put DataFrames
    
    ############################################################# 

    #First DF for the first 10 years
    FCCF_model = [revenue_growth, revenue, op_margin, ebit, tax_rate, ebit_after_tax, sales_to_capital, reinvestment_amt, FCFF, Capital_Cost, Discount_Factor, PV_FCFF]
    return_df = pd.DataFrame(FCCF_model, 
                             columns = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5", "Year 6", "Year 7", "Year 8", "Year 9", "Year 10"], 
                             index = [
        "Revenue Growth",          # revenue_growth
        "Revenue",                 # revenue
        "Operative Margin",         # op_margin
        "EBIT",# ebit
        "Tax-rate t",           # tax_rate
        "EBIT*(1-t)",               # ebit_after_tax
        "Sales To Capital",        # Sales to Capital
        "- Reinvestment",          # reinvestment_amt
        "FCFF",                    # FCFF
        "Cost of Capital (WACC)",           # Capital_Cost
        "Discount Factor",
        "PV of FCFF"         # PV_FCFF
    ])

    # Define which rows should be formatted as percentages
    percent_rows = [
        "Revenue Growth",
        "Operative Margin",
        "Tax-rate t",
        "Cost of Capital (WACC)",
        "Discount Factor"
    ]

    decimal_rows = ["Sales To Capital"]
    # Apply formatting to percentage rows
    return_df.loc[percent_rows] = return_df.loc[percent_rows].applymap(lambda x: f"{x:.1%}")
    
    # round other rows to percentages
    for row in return_df.index:
        if row not in percent_rows:
            if row not in decimal_rows:
                return_df.loc[row] = return_df.loc[row].apply(lambda x: f"{round(x):,}")
    
    # Second Data Frame with the Values for Terminal Value period and the final share price etc.    
    TV_period_Data = [
        ("I. Sum of the PV of FCFF in the next 10 years", NPV_10),
        ("Revenue TV Period",revenue_TV),
        ("EBIT TV", ebit_TV),
        ("Reinvestment TV", reinvestment_TV),
        ("FCFF in TV-Periode (TV-FCFF)", FCFF_TV),
        ("Cost of Capital in TV-Periode (k)", CoC_TV),
        ("Growth Rate in TV-Periode (g)", growth_TV),
        ("Terminal Value = TV-FCFF / (k-g)", TV),
        ("II. PV of the Terminal Value", NPV_TV),
        ("I + II: Sum of PV (=Value of operating assets)", NPV_total),
        ("- Liabilities", current_company["Debt"]),
        ("+ Cash", current_company["Cash"]),
        ("Value of Equity", Equity_Value),
        ("Number of outstanding shares", current_company["SharesOut"]),
        ("Fair, fundamental value per stock", PPS)
        
    ]

    return_df2 = pd.DataFrame(TV_period_Data, columns=["Label", "Value"]).set_index("Label")



        # Define which rows are percentages
    percent_rows2 = [
        "Cost of Capital in TV-Periode (k)",
        "Growth Rate in TV-Periode (g)"
    ]
    
    # Add 2 decimals for fair share price
    float_2dec_rows = [
        "Fair, fundamental value per stock"
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

    #print PPS
    st.markdown(f"### :green[Fair Value per Share:] **{PPS:.2f} $**")

    #get company ticker and call Yahoo Finance API to get current stock price for a comparison
    ticker = current_company["Ticker"]
    try:
        stock_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        if PPS < stock_price:
            filler = "higher"
        else: 
            filler = "lower"
        st.markdown(f"### :green[Current Share Price:] **{stock_price:.2f} $**")
        st.markdown(f"#### The current stock price is {abs(stock_price/PPS-1)*100:.2f}% {filler} then its fair value")
        st.caption(f"The real-time stock data is provided by Yahoo Finance\n(https://finance.yahoo.com/quote/{ticker})", unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f"### :green[Unable to connect to Yahoo Finance]")
        st.caption(f"The real-time stock data can be found at Yahoo Finance\n(https://finance.yahoo.com/quote/{ticker})", unsafe_allow_html=True)

    #return the dfs and do some formatting in streamlit
    st.write("")
    st.subheader("DCF for the next 10 years")
    st.dataframe(return_df)
    st.write("")
    st.subheader("Terminal Value Estimation and Fundamental Value")
    st.dataframe(return_df2, width = 500)
    st.write("")
    st.subheader("Implicit assumptions regarding return on invested capital for the first 10 years")
    st.dataframe(return_df3, height = 154)
    st.write("")
    st.subheader("Graphics")
    
    #call plot function with correct values to plot Revenue estimation and ROIC estemation
    revenue_h = pd.Series(current_company["Historic Revenue"])
    revenue_new = pd.concat([revenue_h, pd.Series(revenue)])
    plot_saeulendiagramm(years[10:], revenue_new, "Revenue (in mln USD)")

    roic_new = pd.concat([pd.Series(current_company["Historic ROIC"])/100, pd.Series(roic[1:])])
    plot_saeulendiagramm(years[10:], roic_new, "ROIC")
    st.caption(f"The historical stock data is provided by Bloomberg", unsafe_allow_html=False)
###############################################################

# Build site with streamlit 

###############################################################
st.header("Model Input")

#select company via streamlit 
select_company = st.radio("Select a company", list(companies.keys())) 

#set current company variable to selected
current_company = companies[select_company]

st.divider()
st.subheader("Revenue Growth Rate")
#get user input for revenue_growth via slider
st.write("Estimate the initial revenue growth (constant for the first 5 years, then it converges towards the terminal value growth rate.\nHigher rates will expectedly lead to a higher company value.\nThe terminal value (stable) growth rate determines how fast cashflows grow in perpetuity.\nUsually this is set to a value that is at max the GDP growth rate. Company cannot indefinately grow faster than a country (Damodaran, 2025)")

revenue_growth_start_percent = st.slider(
    "Initial Revenue Growth Rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=15.0,
    step=0.1,
    format="%.1f%%"
)

revenue_growth_end_percent = st.slider(
    "Terminal Value Stable Growth Rate (%)",
    min_value=0.0,
    max_value=revenue_growth_start_percent,
    value=4.3,
    step=0.1,
    format="%.1f%%"
)

st.divider()
st.subheader("Target Operating Margin")
st.write("The Target Operating Margin is a company's goal for operating profit as a percentage of revenue, reflecting its desired efficiency and profitability from core operations. A higher target margin typically indicates better scalability and cost control, which can significantly increase the company's valuation by boosting projected future earnings.")
#get user input for op margin via slider 
op_margin_input_percent = st.slider(
    "Operating Margin (%)",
    min_value=0.0,
    max_value=100.0,
    value=45.0,
    step=0.1,
    format="%.1f%%"
)

st.divider()
st.subheader("Capital Costs")
st.write("The Capital Costs are the costs at which the cash flows are discounted, to get an equivalent to todays value. Initial cost of capital is calculated via the WACC formula. The terminal value is typically lower, because as companies mature, they tend to use more debt, which is cheaper than equity, bringing down the overall cost of capital.\n(In this model we assume the initial cost to be constant for 5 years, then it converges towards the terminal value cost of capital)")
#get user input for CoC via slider 
Capital_Cost_start_percent = st.slider(
    "Initial Cost of Capital (%)",
    min_value=0.0,
    max_value=100.0,
    value=15.0,
    step=0.1,
    format="%.1f%%"
)

Capital_Cost_end_percent = st.slider(
    "TV Cost of Capital (%)",
    min_value=0.0,
    max_value=Capital_Cost_start_percent,
    value=8.40,
    step=0.1,
    format="%.1f%%"
)

st.divider()
st.subheader("ROIC (for terminal value period)")
st.write("Return of invested capital is the return of a company over its invested assets.To calculate the reinvestment, ROIC is required. The reinvestment rate in stable growth is equal to the stable growth rate of revenue over the return of invested capital. In this model it is used to link sales to capital reinvestment. (Damodaran, 2025)")
TV_ROIC_percent = st.slider(
    "Terminal Value Return on Capital (%)",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=0.1,
    format="%.1f%%"
)
st.divider()
st.header("Model Output")

# set inputs to the decimal value as we get them in natural numbers
op_margin_input = op_margin_input_percent / 100
revenue_growth_start = revenue_growth_start_percent / 100
revenue_growth_end = revenue_growth_end_percent / 100
Capital_Cost_start = Capital_Cost_start_percent / 100
Capital_Cost_end = Capital_Cost_end_percent / 100
TV_ROIC = TV_ROIC_percent / 100
#call functions to rerun script on change
FCFF_func()










