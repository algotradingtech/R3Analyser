import yfinance as yf
import pandas as pd
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import Counter
class R3Classifier:
    def __init__(self, ohlcDf, classificationValues : dict , typeEntry: str = "Close",numberOfEntries:int = 0):
        '''

        :param ohlcDf: dataframe Open High Low Close
        :param classficationValues: {'R3': 1 ,'targetRiskValue':0.01}
        :param typeEntry:
        '''
        self.dataDf = ohlcDf.dropna()
        self.targetR3 = classificationValues["R3"]
        self.targetRisk = classificationValues["targetRiskValue"]
        self.typeEntry = typeEntry
        self.posTypeDict= {"buy": True , "sell": False}
        self.scenarioTypeDict = {"buyTp": True, "sellTp": False, "buySl": False, 'sellSl': True}
        self.lastBestPositionToDo = 0
        self.numberOfentries = numberOfEntries


    def get_tp_sl(self, entry, targetRisk, RRR):
        slTpDict = {
            "buy": [entry * (1 + targetRisk * RRR), entry * (1 - targetRisk)],
            "sell": [entry * (1 - targetRisk * RRR), entry * (1 + targetRisk)]
        }
        return slTpDict

    def set_tp_sl(self):
        def apply_tp_sl(row):
            tp_sl_values = self.get_tp_sl(row[self.typeEntry], self.targetRisk, self.targetR3)
            return pd.Series({
                'buyTp': tp_sl_values['buy'][0],
                'sellTp': tp_sl_values['sell'][0],
                'buySl': tp_sl_values['buy'][1],
                'sellSl': tp_sl_values['sell'][1]
            })
        tp_sl_df = self.dataDf.apply(apply_tp_sl, axis=1)
        self.dataDf = pd.concat([self.dataDf, tp_sl_df], axis=1)

    def get_closest_cross_value_index(self, startIndex, valueToCheck, isUp: bool = True):
        if isUp:
            df = self.dataDf["High"].iloc[startIndex + 1:]
            df = df[df >= valueToCheck]
        else:
            df = self.dataDf["Low"].iloc[startIndex + 1:]
            df = df[df <= valueToCheck]

        if not df.empty:
            return int(df.index[0])
        else:
            return None

    def set_closest_cross_value_index(self):
        for scenarioToCheck, isUp in self.scenarioTypeDict.items():
            self.dataDf[f'closest{scenarioToCheck[0].upper()+ scenarioToCheck[1:] }Index'] = self.dataDf.apply(
                lambda row: self.get_closest_cross_value_index(
                    startIndex=row.name, valueToCheck=row[scenarioToCheck], isUp=isUp), axis=1)
        self.dataDf.fillna(0, inplace=True)

    def get_position_result(self,closestTpIndex: int,closestSlIndex:int):
        if closestTpIndex == closestSlIndex:
            return None
        if closestSlIndex != 0  and closestTpIndex != 0 :
            if closestTpIndex < closestSlIndex:
                return True
            elif closestTpIndex > closestSlIndex:
                return False
        elif closestTpIndex > 0:
            return True
        else:
            return False

    def set_position_result(self):
        for posType,posTypeValue in self.posTypeDict.items() :
            self.dataDf[f'{posType}Result'] = self.dataDf.apply(lambda row : self.get_position_result(
            closestTpIndex=row[f'closest{posType.capitalize()}TpIndex'],closestSlIndex = row[f'closest{posType.capitalize()}SlIndex']),axis=1)

    def get_best_position_type(self,buyResult : bool,sellResult : bool) :
        if buyResult :
            if self.lastBestPositionToDo >= self.numberOfentries:
                return 1
            else:
                self.lastBestPositionToDo += 1
                return 0
        elif sellResult:
            if self.lastBestPositionToDo <= -self.numberOfentries:
                return -1
            else:
                self.lastBestPositionToDo -= 1
                return 0
        else:
            self.lastBestPositionToDo = 0
            return 0

    def set_best_position_type(self):
        self.dataDf["bestPositionToDo"] = self.dataDf.apply(lambda row: self.get_best_position_type(
            buyResult=row["buyResult"], sellResult=row["sellResult"]), axis=1)
        #self.dataDf["bestPositionToDo"] = self.dataDf.apply(lambda x : x["bestPositionToDo"] if x["volBreak"] is not 0 else 0,axis=1)


    def get_missed_position(self, startedIndex, closestBuySl, closestSellSl):
        countMissedSell = (self.dataDf["bestPositionToDo"].iloc[startedIndex:closestSellSl] == -1).sum()
        countMissedBuy = (self.dataDf["bestPositionToDo"].iloc[startedIndex:closestBuySl] == 1).sum()
        return {"missedeBuyaccount": countMissedBuy, "missedeSellaccount": countMissedSell}

    def set_missed_position(self):
        def apply_missed_position(row):
            if row["bestPositionToDo"] == 0:
                missed_positions = self.get_missed_position(row.name, int(row["closestBuySlIndex"]), int(row["closestSellSlIndex"]))
                return pd.Series({
                    "missedeBuyaccount": missed_positions["missedeBuyaccount"],
                    "missedeSellaccount": missed_positions["missedeSellaccount"]
                })
            else:
                return pd.Series({"missedeBuyaccount": 0, "missedeSellaccount": 0})

        missed_position_df = self.dataDf.apply(apply_missed_position, axis=1)
        self.dataDf = pd.concat([self.dataDf, missed_position_df], axis=1)

    def run(self):
        self.set_tp_sl()
        self.set_closest_cross_value_index()
        self.set_position_result()
        self.set_best_position_type()
        self.set_missed_position()
        self.bestPosToDo = self.dataDf[["bestPositionToDo","missedeBuyaccount", "missedeSellaccount"]]


class OptimR3Classifier:
    def __init__(self,df,r3Dict:dict =  False,targetRiskDict:dict = False):
        """

        :param df: ohlc df
        :param r3Dict: {"Range":[100,400],"step":1 ,"defaultValueParam2":100,}
        :param targetRiskDict: { "Range":[1,10],"step":1,"defaultValueParam2": 1}
        """
        self.dataDf = df
        self.r3Dict = r3Dict
        self.targetRiskDict= targetRiskDict
        self.optimResult = dict()
        self.currentParamDict = {"R3": None,
                                 "targetRiskValue":None
                                 }
        self.optimResult=list()


    def call_R3Classifier(self):

        classificationResult = R3Classifier(
            ohlcDf=self.dataDf,
            classificationValues=self.currentParamDict
        )
        classificationResult.run()


        return classificationResult.bestPosToDo


    def get_R3_classifier_loop(self,isr3Loop : bool = False , isTargetRiskLoop : bool = False):
        is3dLoop = True if isr3Loop and isTargetRiskLoop else False

        if not is3dLoop:
            loopDict = self.r3Dict if isr3Loop else self.targetRiskDict
            start = loopDict["range"][0]
            end =loopDict["range"][1]
            step =loopDict["step"]
            defaultValueParam2 = loopDict["defaultValueParam2"]

            for classificationValue in range(start,end+1,step):
                self.currentParamDict  = {'R3': classificationValue/10000 if isr3Loop is not False else defaultValueParam2/10000 ,
                             "targetRiskValue": classificationValue/10000 if isTargetRiskLoop is not False else defaultValueParam2/10000
                             }
                result = self.call_R3Classifier()
                self.optimResult.append({"param": self.currentParamDict,
                                    "values": result
                                    })
        else:
            startR3 = self.r3Dict["range"][0]
            endR3 = self.r3Dict["range"][1]
            stepR3= self.r3Dict["step"]
            startTargetRisk = self.targetRiskDict["range"][0]
            endTargetRisk = self.targetRiskDict["range"][1]
            stepTargetRisk= self.targetRiskDict["step"]

            for R3ClassificationValue in range(startR3,endR3+stepR3,stepR3) :
                for targetRiskClassificationValue in range(int(startTargetRisk),int(endTargetRisk+1) ,int(stepTargetRisk)):
                    self.currentParamDict={"R3":R3ClassificationValue,
                                           "targetRiskValue":targetRiskClassificationValue/10000
                                           }
                    result = self.call_R3Classifier()
                    self.optimResult.append({"param": self.currentParamDict,
                                        "values": result
                                        })
            return self.optimResult


    def analyze_list(self,lst):
        counter = Counter(lst)
        total_count = len(lst)
        proportions = {k: v / total_count for k, v in counter.items()}
        def max_consecutive(lst, value):
            max_count = count = 0
            for i in lst:
                if i == value:
                    count += 1
                    if count > max_count:
                        max_count = count
                else:
                    count = 0
            return max_count

        max_consecutive_1 = max_consecutive(lst, 1)
        max_consecutive_neg1 = max_consecutive(lst, -1)
        max_consecutive_0 = max_consecutive(lst, 0)

        return {
                'proportions': proportions,
                'max_consecutive_1': max_consecutive_1,
                'max_consecutive_neg1': max_consecutive_neg1,
                'max_consecutive_0': max_consecutive_0
        }


    def display_result_analysis(self):
        results = []
        for result in self.optimResult:
            bestPositionToDo = result["values"]["bestPositionToDo"]
            analysis = self.analyze_list(bestPositionToDo.tolist())
            current_result = {
                'R3': result["param"]["R3"],
                'Target Risk': result["param"]["targetRiskValue"],
                'proportions_buy_signal': analysis['proportions'].get(1, 0),
                'proportions_sell_signal': analysis['proportions'].get(-1, 0),
                'proportions_no_signal': analysis['proportions'].get(0, 0),
                'max_consecutive_buy_signal': analysis['max_consecutive_1'],
                'max_consecutive_sell_signal': analysis['max_consecutive_neg1'],
                'max_consecutive_no_signal': analysis['max_consecutive_0'],
                'Profit Factor': (1 - analysis['proportions'].get(0, 0)) * result["param"]["R3"],
                "missedeBuyaccountAverage": result["values"]["missedeBuyaccount"].mean(),
                "missedeSellaccountAverage": result["values"]["missedeSellaccount"].mean()
            }
            results.append(current_result)

        dfResult = pd.DataFrame(results)
        return dfResult
class DataSymbol:
    def __init__(self, symbol, startDate, endDate,interval = "1d"):
        self.symbol = symbol
        self.startDate = startDate
        self.endDate = endDate
        self.interval = interval

    def get_priceData(self, startDate=None, endDate=None):
        df = yf.download(self.symbol, start=startDate if startDate else self.startDate, end=endDate if endDate else self.endDate,interval=self.interval).reset_index(drop=False).rename(columns={'index': 'Date'})
        return df

    def set_priceData(self):
        self.priceData = self.get_priceData()

    def set_startDate(self, newStartDate):
        self.startDate = newStartDate

    def set_endDate(self, newEndDate):
        self.endDate = newEndDate
class ResultVisualizer:
    def __init__(self, df_result):
        self.df_result = df_result

    def plot_2d(self, x_col, y_col, color_col=None, title="2D Plot", xlabel=None, ylabel=None):
        xlabel = xlabel if xlabel is not None else x_col
        ylabel = ylabel if ylabel is not None else y_col

        if color_col:
            fig = px.scatter(self.df_result, x=x_col, y=y_col, color=color_col, title=title)
        else:
            fig = px.scatter(self.df_result, x=x_col, y=y_col, title=title)
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
        return fig

    def plot_3d(self, x_col, y_col, z_col, color_col=None, title="3D Plot", xlabel=None, ylabel=None, zlabel=None):
        xlabel = xlabel if xlabel is not None else x_col
        ylabel = ylabel if ylabel is not None else y_col
        zlabel = zlabel if zlabel is not None else z_col

        if color_col:
            fig = px.scatter_3d(self.df_result, x=x_col, y=y_col, z=z_col, color=color_col, title=title)
        else:
            fig = px.scatter_3d(self.df_result, x=x_col, y=y_col, z=z_col, title=title)
        fig.update_layout(scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel
        ))
        return fig

    def plot_surface(self, x_col, y_col, z_col, title="Surface Plot", xlabel=None, ylabel=None, zlabel=None):
        xlabel = xlabel if xlabel is not None else x_col
        ylabel = ylabel if ylabel is not None else y_col
        zlabel = zlabel if zlabel is not None else z_col

        fig = px.surface(self.df_result, x=x_col, y=y_col, z=z_col, title=title)
        fig.update_layout(scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel
        ))
        return fig

    def plot_bar(self, x_col, y_col, title="Bar Plot", xlabel=None, ylabel=None):
        xlabel = xlabel if xlabel is not None else x_col
        ylabel = ylabel if ylabel is not None else y_col

        fig = px.bar(self.df_result, x=x_col, y=y_col, title=title)
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
        return figimport yfinance as yf
import pandas as pd
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import Counter
class R3Classifier:
    def __init__(self, ohlcDf, classificationValues : dict , typeEntry: str = "Close",numberOfEntries:int = 0):
        '''

        :param ohlcDf: dataframe Open High Low Close
        :param classficationValues: {'R3': 1 ,'targetRiskValue':0.01}
        :param typeEntry:
        '''
        self.dataDf = ohlcDf.dropna()
        self.targetR3 = classificationValues["R3"]
        self.targetRisk = classificationValues["targetRiskValue"]
        self.typeEntry = typeEntry
        self.posTypeDict= {"buy": True , "sell": False}
        self.scenarioTypeDict = {"buyTp": True, "sellTp": False, "buySl": False, 'sellSl': True}
        self.lastBestPositionToDo = 0
        self.numberOfentries = numberOfEntries


    def get_tp_sl(self, entry, targetRisk, RRR):
        slTpDict = {
            "buy": [entry * (1 + targetRisk * RRR), entry * (1 - targetRisk)],
            "sell": [entry * (1 - targetRisk * RRR), entry * (1 + targetRisk)]
        }
        return slTpDict

    def set_tp_sl(self):
        def apply_tp_sl(row):
            tp_sl_values = self.get_tp_sl(row[self.typeEntry], self.targetRisk, self.targetR3)
            return pd.Series({
                'buyTp': tp_sl_values['buy'][0],
                'sellTp': tp_sl_values['sell'][0],
                'buySl': tp_sl_values['buy'][1],
                'sellSl': tp_sl_values['sell'][1]
            })
        tp_sl_df = self.dataDf.apply(apply_tp_sl, axis=1)
        self.dataDf = pd.concat([self.dataDf, tp_sl_df], axis=1)

    def get_closest_cross_value_index(self, startIndex, valueToCheck, isUp: bool = True):
        if isUp:
            df = self.dataDf["High"].iloc[startIndex + 1:]
            df = df[df >= valueToCheck]
        else:
            df = self.dataDf["Low"].iloc[startIndex + 1:]
            df = df[df <= valueToCheck]

        if not df.empty:
            return int(df.index[0])
        else:
            return None

    def set_closest_cross_value_index(self):
        for scenarioToCheck, isUp in self.scenarioTypeDict.items():
            self.dataDf[f'closest{scenarioToCheck[0].upper()+ scenarioToCheck[1:] }Index'] = self.dataDf.apply(
                lambda row: self.get_closest_cross_value_index(
                    startIndex=row.name, valueToCheck=row[scenarioToCheck], isUp=isUp), axis=1)
        self.dataDf.fillna(0, inplace=True)

    def get_position_result(self,closestTpIndex: int,closestSlIndex:int):
        if closestTpIndex == closestSlIndex:
            return None
        if closestSlIndex != 0  and closestTpIndex != 0 :
            if closestTpIndex < closestSlIndex:
                return True
            elif closestTpIndex > closestSlIndex:
                return False
        elif closestTpIndex > 0:
            return True
        else:
            return False

    def set_position_result(self):
        for posType,posTypeValue in self.posTypeDict.items() :
            self.dataDf[f'{posType}Result'] = self.dataDf.apply(lambda row : self.get_position_result(
            closestTpIndex=row[f'closest{posType.capitalize()}TpIndex'],closestSlIndex = row[f'closest{posType.capitalize()}SlIndex']),axis=1)

    def get_best_position_type(self,buyResult : bool,sellResult : bool) :
        if buyResult :
            if self.lastBestPositionToDo >= self.numberOfentries:
                return 1
            else:
                self.lastBestPositionToDo += 1
                return 0
        elif sellResult:
            if self.lastBestPositionToDo <= -self.numberOfentries:
                return -1
            else:
                self.lastBestPositionToDo -= 1
                return 0
        else:
            self.lastBestPositionToDo = 0
            return 0

    def set_best_position_type(self):
        self.dataDf["bestPositionToDo"] = self.dataDf.apply(lambda row: self.get_best_position_type(
            buyResult=row["buyResult"], sellResult=row["sellResult"]), axis=1)
        #self.dataDf["bestPositionToDo"] = self.dataDf.apply(lambda x : x["bestPositionToDo"] if x["volBreak"] is not 0 else 0,axis=1)


    def get_missed_position(self, startedIndex, closestBuySl, closestSellSl):
        countMissedSell = (self.dataDf["bestPositionToDo"].iloc[startedIndex:closestSellSl] == -1).sum()
        countMissedBuy = (self.dataDf["bestPositionToDo"].iloc[startedIndex:closestBuySl] == 1).sum()
        return {"missedeBuyaccount": countMissedBuy, "missedeSellaccount": countMissedSell}

    def set_missed_position(self):
        def apply_missed_position(row):
            if row["bestPositionToDo"] == 0:
                missed_positions = self.get_missed_position(row.name, int(row["closestBuySlIndex"]), int(row["closestSellSlIndex"]))
                return pd.Series({
                    "missedeBuyaccount": missed_positions["missedeBuyaccount"],
                    "missedeSellaccount": missed_positions["missedeSellaccount"]
                })
            else:
                return pd.Series({"missedeBuyaccount": 0, "missedeSellaccount": 0})

        missed_position_df = self.dataDf.apply(apply_missed_position, axis=1)
        self.dataDf = pd.concat([self.dataDf, missed_position_df], axis=1)

    def run(self):
        self.set_tp_sl()
        self.set_closest_cross_value_index()
        self.set_position_result()
        self.set_best_position_type()
        self.set_missed_position()
        self.bestPosToDo = self.dataDf[["bestPositionToDo","missedeBuyaccount", "missedeSellaccount"]]


class OptimR3Classifier:
    def __init__(self,df,r3Dict:dict =  False,targetRiskDict:dict = False):
        """

        :param df: ohlc df
        :param r3Dict: {"Range":[100,400],"step":1 ,"defaultValueParam2":100,}
        :param targetRiskDict: { "Range":[1,10],"step":1,"defaultValueParam2": 1}
        """
        self.dataDf = df
        self.r3Dict = r3Dict
        self.targetRiskDict= targetRiskDict
        self.optimResult = dict()
        self.currentParamDict = {"R3": None,
                                 "targetRiskValue":None
                                 }
        self.optimResult=list()


    def call_R3Classifier(self):

        classificationResult = R3Classifier(
            ohlcDf=self.dataDf,
            classificationValues=self.currentParamDict
        )
        classificationResult.run()


        return classificationResult.bestPosToDo


    def get_R3_classifier_loop(self,isr3Loop : bool = False , isTargetRiskLoop : bool = False):
        is3dLoop = True if isr3Loop and isTargetRiskLoop else False

        if not is3dLoop:
            loopDict = self.r3Dict if isr3Loop else self.targetRiskDict
            start = loopDict["range"][0]
            end =loopDict["range"][1]
            step =loopDict["step"]
            defaultValueParam2 = loopDict["defaultValueParam2"]

            for classificationValue in range(start,end+1,step):
                self.currentParamDict  = {'R3': classificationValue/10000 if isr3Loop is not False else defaultValueParam2/10000 ,
                             "targetRiskValue": classificationValue/10000 if isTargetRiskLoop is not False else defaultValueParam2/10000
                             }
                result = self.call_R3Classifier()
                self.optimResult.append({"param": self.currentParamDict,
                                    "values": result
                                    })
        else:
            startR3 = self.r3Dict["range"][0]
            endR3 = self.r3Dict["range"][1]
            stepR3= self.r3Dict["step"]
            startTargetRisk = self.targetRiskDict["range"][0]
            endTargetRisk = self.targetRiskDict["range"][1]
            stepTargetRisk= self.targetRiskDict["step"]

            for R3ClassificationValue in range(startR3,endR3+stepR3,stepR3) :
                for targetRiskClassificationValue in range(int(startTargetRisk),int(endTargetRisk+1) ,int(stepTargetRisk)):
                    self.currentParamDict={"R3":R3ClassificationValue,
                                           "targetRiskValue":targetRiskClassificationValue/10000
                                           }
                    result = self.call_R3Classifier()
                    self.optimResult.append({"param": self.currentParamDict,
                                        "values": result
                                        })
            return self.optimResult


    def analyze_list(self,lst):
        counter = Counter(lst)
        total_count = len(lst)
        proportions = {k: v / total_count for k, v in counter.items()}
        def max_consecutive(lst, value):
            max_count = count = 0
            for i in lst:
                if i == value:
                    count += 1
                    if count > max_count:
                        max_count = count
                else:
                    count = 0
            return max_count

        max_consecutive_1 = max_consecutive(lst, 1)
        max_consecutive_neg1 = max_consecutive(lst, -1)
        max_consecutive_0 = max_consecutive(lst, 0)

        return {
                'proportions': proportions,
                'max_consecutive_1': max_consecutive_1,
                'max_consecutive_neg1': max_consecutive_neg1,
                'max_consecutive_0': max_consecutive_0
        }


    def display_result_analysis(self):
        results = []
        for result in self.optimResult:
            bestPositionToDo = result["values"]["bestPositionToDo"]
            analysis = self.analyze_list(bestPositionToDo.tolist())
            current_result = {
                'R3': result["param"]["R3"],
                'Target Risk': result["param"]["targetRiskValue"],
                'proportions_buy_signal': analysis['proportions'].get(1, 0),
                'proportions_sell_signal': analysis['proportions'].get(-1, 0),
                'proportions_no_signal': analysis['proportions'].get(0, 0),
                'max_consecutive_buy_signal': analysis['max_consecutive_1'],
                'max_consecutive_sell_signal': analysis['max_consecutive_neg1'],
                'max_consecutive_no_signal': analysis['max_consecutive_0'],
                'Profit Factor': (1 - analysis['proportions'].get(0, 0)) * result["param"]["R3"],
                "missedeBuyaccountAverage": result["values"]["missedeBuyaccount"].mean(),
                "missedeSellaccountAverage": result["values"]["missedeSellaccount"].mean()
            }
            results.append(current_result)

        dfResult = pd.DataFrame(results)
        return dfResult
class DataSymbol:
    def __init__(self, symbol, startDate, endDate,interval = "1d"):
        self.symbol = symbol
        self.startDate = startDate
        self.endDate = endDate
        self.interval = interval

    def get_priceData(self, startDate=None, endDate=None):
        df = yf.download(self.symbol, start=startDate if startDate else self.startDate, end=endDate if endDate else self.endDate,interval=self.interval).reset_index(drop=False).rename(columns={'index': 'Date'})
        return df

    def set_priceData(self):
        self.priceData = self.get_priceData()

    def set_startDate(self, newStartDate):
        self.startDate = newStartDate

    def set_endDate(self, newEndDate):
        self.endDate = newEndDate
class ResultVisualizer:
    def __init__(self, df_result):
        self.df_result = df_result

    def plot_2d(self, x_col, y_col, color_col=None, title="2D Plot", xlabel=None, ylabel=None):
        xlabel = xlabel if xlabel is not None else x_col
        ylabel = ylabel if ylabel is not None else y_col

        if color_col:
            fig = px.scatter(self.df_result, x=x_col, y=y_col, color=color_col, title=title)
        else:
            fig = px.scatter(self.df_result, x=x_col, y=y_col, title=title)
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
        return fig

    def plot_3d(self, x_col, y_col, z_col, color_col=None, title="3D Plot", xlabel=None, ylabel=None, zlabel=None):
        xlabel = xlabel if xlabel is not None else x_col
        ylabel = ylabel if ylabel is not None else y_col
        zlabel = zlabel if zlabel is not None else z_col

        if color_col:
            fig = px.scatter_3d(self.df_result, x=x_col, y=y_col, z=z_col, color=color_col, title=title)
        else:
            fig = px.scatter_3d(self.df_result, x=x_col, y=y_col, z=z_col, title=title)
        fig.update_layout(scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel
        ))
        return fig

    def plot_surface(self, x_col, y_col, z_col, title="Surface Plot", xlabel=None, ylabel=None, zlabel=None):
        xlabel = xlabel if xlabel is not None else x_col
        ylabel = ylabel if ylabel is not None else y_col
        zlabel = zlabel if zlabel is not None else z_col

        fig = px.surface(self.df_result, x=x_col, y=y_col, z=z_col, title=title)
        fig.update_layout(scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel
        ))
        return fig

    def plot_bar(self, x_col, y_col, title="Bar Plot", xlabel=None, ylabel=None):
        xlabel = xlabel if xlabel is not None else x_col
        ylabel = ylabel if ylabel is not None else y_col

        fig = px.bar(self.df_result, x=x_col, y=y_col, title=title)
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
        return fig
