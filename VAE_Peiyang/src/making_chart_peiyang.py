import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import calendar
import pickle as pk
import torchvision.transforms as transforms

# Peiyang REvision
IMAGE_WIDTH = {5: 15, 20: 60, 25: 75, 30: 90, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 25: 64, 30: 64, 60: 96}

class Scale:
    def __init__(self, height):
        self.height = height
        
    def set_scale(self, rangeV): #set the scale for mapping the price to pixels
        self.range = rangeV
        self.vertical_scale_linear = (self.height-1)/(max(rangeV) - min(rangeV))
         ##range is from 1 to height

    def linear(self, value):
        return self.height - (value-min(self.range)) * self.vertical_scale_linear


class Chart:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.barWidth = 3
        # self.img = np.zeros([width, height], dtype=np.uint8)
    
    def generate_with_public_holidays(self, data, workdays, scale): #generating a chart in workdays, 
        img = np.zeros([self.height, self.width], dtype=np.uint8)
        minV = data.min()[["Open", "Close", "High", "Low"]].min()
        maxV = data.max()[["Open", "Close", "High", "Low"]].max()
        scale.set_scale([minV, maxV])
        # print("workdays", workdays)
        for i in range(len(workdays)):
            try:
                o = round(scale.linear(data.loc[workdays[i]]["Open"])-1)
                c = round(scale.linear(data.loc[workdays[i]]["Close"])-1)
                h = round(scale.linear(data.loc[workdays[i]]["High"])-1)
                l = round(scale.linear(data.loc[workdays[i]]["Low"])-1)
                # print(o, c, h, l)
                img[o, i*self.barWidth] = 255
                img[h:l+1, i*self.barWidth+1] = 255
                img[c, i*self.barWidth+2] = 255
            except KeyError:
                continue
        return img
    
    def generate(self, data, scale):
        img = np.zeros([self.height, self.width], dtype=np.uint8)
        minV = data.min()[["Open", "Close", "High", "Low"]].min()
        maxV = data.max()[["Open", "Close", "High", "Low"]].max()
        scale.set_scale([minV, maxV])
        for i in range(len(data)):
            o = round(scale.linear(data.iloc[i]["Open"])-1)
            c = round(scale.linear(data.iloc[i]["Close"])-1)
            h = round(scale.linear(data.iloc[i]["High"])-1)
            l = round(scale.linear(data.iloc[i]["Low"])-1)
            # print(o, c, h, l)
            img[o, i*self.barWidth] = 255
            img[h:l+1, i*self.barWidth+1] = 255
            img[c, i*self.barWidth+2] = 255
        return img


class IndexGenerator:
    def __init__(self, dataset):
        self.dataset = dataset

    def n_working_days_before(self, current_date, n):
        while n>0:
            current_date -= datetime.timedelta(days=1)
            weekday = current_date.weekday() #Saturday 5, Sunday 6 
            if weekday >= 5:
                continue
            n -= 1
        return current_date

    def n_working_days_after(self, current_date, n):
        while n>0:
            current_date += datetime.timedelta(days=1)
            weekday = current_date.weekday()
            if weekday >= 5:
                continue
            n -= 1
        return current_date

    def get_periodical_indices(self, year, period):
        if period=="monthly":
            time_gap = 19# the whole period is 20 days but the current day is included, so time_gap is 19.
            start = datetime.date(year, 1, 1)
            end = datetime.date(year, 12, 31)
            date = pd.bdate_range(start, end, freq="BM") #find the last workday in each month
            date = date.to_pydatetime()
            data_indices = []
            label_indices = []
            for i in range(len(date)):
                data_indices.append([self.n_working_days_before(date[i], time_gap).strftime('%Y-%m-%d'), date[i].strftime('%Y-%m-%d')])
                label_indices.append([self.n_working_days_after(date[i], 1).strftime('%Y-%m-%d'), self.n_working_days_after(date[i], time_gap).strftime('%Y-%m-%d')])
            return data_indices, label_indices
    
    def get_indices(self, year, time_gap_data, time_gap_label, stride=1):
        start = datetime.date(year, 1, 1)
        end = datetime.date(year, 12, 31)
        date = self.dataset.loc[start: end]["Date"] ##pandas.loc is end-inclusive, so, year/12/31 is included.
        data_indices = []
        label_indices = []
        for i in range(0, len(date), stride):
            if i+time_gap_data+time_gap_label-1<len(date):
                
                s_data = date[i].strftime('%Y-%m-%d')
                e_data = date[i+time_gap_data-1].strftime('%Y-%m-%d')
                data_indices.append([s_data, e_data])
                s_label = date[i+time_gap_data].strftime('%Y-%m-%d')
                e_label = (date[i+time_gap_data+time_gap_label-1]).strftime('%Y-%m-%d')
                label_indices.append([s_data, e_label])
        return data_indices, label_indices

# take path of csv file then returns the indices of the data and label with .pk format
# Peiyang revision
def generate_pk(path, start_year = 2002, end_year = 2022, time_gap_data = 20, time_gap_label = 5, stride=5):
    # print("hi there")
    df = pd.read_csv(path, encoding="GBK") 
    df.rename(columns={
                "代码": "Symbol", 
                "简称": "Name",
                "日期": "Date", 
                "开盘价(元)": "Open",
                "收盘价(元)": "Close",  
                "最高价(元)": "High",
                "最低价(元)": "Low", 
                "成交量(股)": "Volume",
                "成交金额(元)": "Amount",
                "涨跌(元)": "Change amount",
                "涨跌幅(%)": "Change rate",
                "均价(元)": "Average"
            }, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index(df.Date, inplace=True)
    idx = IndexGenerator(df)
    time_gap_data = 20
    time_gap_label = 5
    stride = 5
    scale_data = Scale(IMAGE_HEIGHT[time_gap_data])
    chart_data = Chart(IMAGE_WIDTH[time_gap_data], IMAGE_HEIGHT[time_gap_data])
    scale_label = Scale(IMAGE_HEIGHT[time_gap_data+time_gap_label])
    chart_label = Chart(IMAGE_WIDTH[time_gap_data+time_gap_label], IMAGE_HEIGHT[time_gap_data+time_gap_label])
    years = range(2002, 2022)
    img_all = []
    label_all = []
    for year in years:
        data_indices, label_indices = idx.get_indices(year, time_gap_data, time_gap_label, stride)
        imgs = []
        labels = []
        # print(data_indices[-2:], label_indices[-2:])

        for i in range(len(data_indices)):
            data_data = df.loc[data_indices[i][0]: data_indices[i][1]]
            try:
                img_data = chart_data.generate(data_data, scale_data)
                data_label = df.loc[label_indices[i][0]: label_indices[i][1]]
                img_label = chart_label.generate(data_label, scale_label)
                img_label = img_label[:, time_gap_data*3:] #take the last time_gap_label*3 columns; note:barwidth=3;
                imgs.append(img_data)
                labels.append(img_label)
            except ValueError:
                # print("symbol, year: ", symbol, year)
                continue
        if imgs and labels:
            img_all.extend(imgs)
            label_all.extend(labels)
    return img_all, label_all
    