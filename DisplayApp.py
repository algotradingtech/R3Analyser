import streamlit as st 
from ClassFunctions import OptimR3Classifier , R3Classifier,DataSymbol,ResultVisualizer
import pandas as pd 

class R3Widget:
    def __init__(self):
        self.pages = ["Load Data", "Run OptimR3Classifier", "Visualize Results"]

    def load_data_page(self):
        st.header("Load Data")
        dataSource = st.radio("select source data:",options=['CSV',"yfinance"])
        if dataSource is  "yfinance" :
            symbol = st.text_input("Symbol", value="^FCHI")
            start_date = st.date_input("Start Date", value=pd.to_datetime((pd.Timestamp.now() - pd.tseries.offsets.BDay(22)).strftime('%Y-%m-%d')))
            end_date = st.date_input("End Date", value = pd.to_datetime((pd.Timestamp.now() - pd.tseries.offsets.BDay(1)).strftime('%Y-%m-%d')))
            time_frame = st.selectbox("Time Frame", ['1m',' 2m',' 5m', '15m', '30m', '60m', '90m',' 1h', '1d', '5d', '1wk', '1mo', '3mo'])
            if st.button("Load Data"):
                data_symbol = DataSymbol(symbol, start_date, end_date,time_frame)
                df = data_symbol.get_priceData()
                st.write("Data loaded successfully!")
                st.write(df)
                df.to_csv("loaded_data.csv", index=False)
                st.success("Data correctly insert")
        else:
            filePath =st.file_uploader("Choisissez un fichier CSV", type="csv")
            if st.button("load CSV"):
                df = pd.read_csv(filePath)
                check_col_list= [ True if col in df.columns else False for col in ["Open","High","Low","Close"] ]
                if all(check_col_list): 
                    
                    df.to_csv("loaded_data.csv", index=False)
                    st.success("Data correctly insert")  
                else:
                    st.error("Noms de colonne non comformes => [Open,High,Low,Close]")
                st.write(df)



    def run_optim_r3_classifier_page(self):
        r3_range = st.slider("R3 Range", min_value=1, max_value=20, value=(1, 10))
        r3_step = st.number_input("R3 Step", min_value=1, value=1)
        target_risk_range = st.slider("Target Risk Range Bps ( 100 Bps = 1%) :", min_value=1, max_value=1000, value=(100, 500))
        target_risk_step = st.number_input("Target Risk Step ( Bps ) :", min_value= 10, value = 10,step=10)
        if st.button("Run OptimR3Classifier"):
            r3_dict = {"range": [r3_range[0], r3_range[1]], "step": r3_step, "defaultValueParam2": 100}
            target_risk_dict = {"range": [target_risk_range[0], target_risk_range[1]] , "step": target_risk_step, "defaultValueParam2": 1}
            optim_r3_classifier = OptimR3Classifier(df=pd.read_csv("loaded_data.csv") ,r3Dict=r3_dict, targetRiskDict=target_risk_dict)
            optim_r3_classifier.get_R3_classifier_loop(isr3Loop=True, isTargetRiskLoop=True)
            result_df = optim_r3_classifier.display_result_analysis()
            st.write("Optimization results:")
            st.write(result_df)
            result_df.to_csv("optimization_results.csv", index=False)

    def visualize_results_page(self):
        st.header("Visualize Results")
        result_df = pd.read_csv("optimization_results.csv")
        visualizer = ResultVisualizer(result_df)
        plot_type = st.selectbox("Select Plot Type", ["2D Scatter", "3D Scatter", "Bar Plot"])

        if plot_type == "2D Scatter":
            x_col = st.selectbox("X-axis", result_df.columns)
            y_col = st.selectbox("Y-axis", result_df.columns)
            color_col = st.selectbox("Color (Optional)", ["None"] + list(result_df.columns))
            color_col = None if color_col == "None" else color_col
            if st.button("Generate 2D Scatter Plot"):
                fig = visualizer.plot_2d(x_col, y_col, color_col, title="2D Scatter Plot")
                st.plotly_chart(fig)

        elif plot_type == "3D Scatter":
            x_col = st.selectbox("X-axis", result_df.columns)
            y_col = st.selectbox("Y-axis", result_df.columns)
            z_col = st.selectbox("Z-axis", result_df.columns)
            color_col = st.selectbox("Color (Optional)", ["None"] + list(result_df.columns))
            color_col = None if color_col == "None" else color_col
            if st.button("Generate 3D Scatter Plot"):
                fig = visualizer.plot_3d(x_col, y_col, z_col, color_col, title="3D Scatter Plot")
                st.plotly_chart(fig)

        elif plot_type == "Bar Plot":
            x_col = st.selectbox("X-axis", result_df.columns)
            y_col = st.selectbox("Y-axis", result_df.columns)
            if st.button("Generate Bar Plot"):
                fig = visualizer.plot_bar(x_col, y_col, title="Bar Plot")
                st.plotly_chart(fig)


    def run(self):
        st.title("R3 Analysis")
        page = st.sidebar.selectbox("Choose a page", self.pages)
        if page == "Load Data":
            self.load_data_page()
        elif page == "Run OptimR3Classifier":
            self.run_optim_r3_classifier_page()
        elif page == "Visualize Results":
            self.visualize_results_page()
