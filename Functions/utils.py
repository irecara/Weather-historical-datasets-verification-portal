import pickle
import boto3
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import datetime as dt
import dateutil.parser


def get_AWS_credentials():
    
    load_dotenv()

    AWS_SECRET_KEY=os.getenv('AWS_SECRET_KEY')
    AWS_ACCESS_KEY=os.getenv('AWS_ACCESS_KEY')
    
    return AWS_SECRET_KEY, AWS_ACCESS_KEY


def get_s3():
    
    AWS_SECRET_KEY, AWS_ACCESS_KEY=get_AWS_credentials()
    
    s3 = boto3.client('s3', 
                            aws_access_key_id = AWS_ACCESS_KEY, 
                            aws_secret_access_key = AWS_SECRET_KEY
                           )
    return s3

def save_in_s3(obj, filename, bucket, key_analysis):
    
    AWS_SECRET_KEY, AWS_ACCESS_KEY=get_AWS_credentials()
    
    session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY ,aws_secret_access_key=AWS_SECRET_KEY)
    s3_resource = session.resource('s3')
    
    pickle_byte_obj = pickle.dumps(obj)
    s3_resource.Object(bucket,key_analysis + '/' + filename + '.pkl').put(Body=pickle_byte_obj)
    

def open_from_S3(filename, bucket, key):
    
    s3=get_s3()
    
    response = s3.get_object(Bucket=bucket, Key=key + '/' + filename + '.pkl')
    body = response['Body'].read()
    obj = pickle.loads(body) 
    
    return obj

def get_CEHUB_API():
    
    from dotenv import load_dotenv
    
    load_dotenv()

    CEHUB_API_KEY=os.getenv('CEHUB_API_KEY')
    url = "http://my.meteoblue.com/dataset/query?apikey=" + CEHUB_API_KEY # CE Hub API key
    
    return url

def percentage_below_threshold(series, threshold):
    above_threshold = series[series < threshold]
    return (above_threshold.count() / series.count()) * 100


# function to get data from CE Hub station api endpoitn
def get_station_data(station_id, datefrom, dateto, variables, frequency, format):
    import requests
    import json
    
    load_dotenv()
    CEHUB_API_KEY_WS=os.getenv('CEHUB_API_KEY_WS')
    
    url = "https://t2xer83e5a.execute-api.eu-central-1.amazonaws.com/cehub-prod/stationdata/v2/historic"
    
    headers = {
    'x-api-key': CEHUB_API_KEY_WS,
    'Content-Type': 'application/json'
    }

    payload = {
        "stations": station_id,
        "datefrom": datefrom,
        "dateto": dateto, 
        "variables":variables,
        "frequencies":frequency,
        "format":format
    }
    print(payload)
    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    response = response.json()
    ###### ADD the possibility that there is error response because there are no data for that period !!!!!!!!!!!!!
    return response

# functio to create dataframe from CE Hub station data
def create_df_station_data(station_data):
    
    for _time in range(len(station_data)):
        for _station in range(len(station_data[_time]['dataset'])):
            station_data[_time]['dataset'][_station]['phenTime'] = station_data[_time]['timeSequence']
    # create dataframe
    data = pd.json_normalize(
        station_data, 
        record_path = ['dataset'], 
        )
    data = data.explode(['values', 'phenTime']) 
    data = data[data['values'] != 'NA']
    data = data.dropna(subset=['values'])
    data['values'] = data['values'].astype(float)
    data = data.reset_index(drop=True)
    # pivot on variableID and values
    data = data.pivot_table(index = ['stationID', 'X', 'Y', 'phenTime', 'UtcOffset'], columns = 'variableID', values = 'values').reset_index()
    # remove index name
    data.columns.name = None
    # rename columns
    data = data.rename(columns={'variableID':'phenTime'})
    # conver phenTime to datetime
    data['phenTime'] = pd.to_datetime(data['phenTime'])
    # sort data
    data = data.sort_values(by=['stationID', 'phenTime'])
    # split stationID by '-' and create new columns
    data[['vendor', 'id']] = data['stationID'].str.split('-', expand=True)
    # return data.set_index('phenTime')
    return data

# function to reorganize data from create_df_station_data
def reorganize_data(df):

    # stack data
    df = df.set_index([ 'phenTime', 'stationID', 'X', 'Y', 'UtcOffset', 'vendor', 'id']).stack()
    # unstack data
    df = df.unstack().reset_index()
    # sort by stationID and phenTime
    df = df.sort_values(by=['stationID', 'phenTime'])
    
    # set index to phenTime
    df = df.set_index([ 'phenTime'])

    df=df.reset_index()
    df=df.rename(columns={'phenTime':'Time','X':'Longitude','Y':'Latitude','stationID':'StationID'})
    df['TimeDate']=pd.to_datetime(df['Time'],format='%Y%m%d')

    df['Longitude']=df['Longitude'].astype(float)
    df['Latitude']=df['Latitude'].astype(float)
    df['StationID']=pd.Categorical(df['StationID'])

    df['Longitude']=round(df['Longitude'],3)
    df['Latitude']=round(df['Latitude'],3)

    del df['UtcOffset']
    del df['vendor']
    del df['id']
    del df['TimeDate']
    return df

def meteoblue_timeinterval_to_timestamps(t):
    if len(t.timestrings) > 0:
        def map_ts(time):
            if "-" in time:
                return dateutil.parser.parse(time.partition("-")[0])
            return dateutil.parser.parse(time)

        return list(map(map_ts, t.timestrings))

    timerange = range(t.start, t.end, t.stride)
    return list(map(lambda t: dt.datetime.fromtimestamp(t), timerange))

def meteoblue_result_to_dataframe(geometry):
    t = geometry.timeIntervals[0]
    timestamps = meteoblue_timeinterval_to_timestamps(t)

    n_locations = len(geometry.lats)
    n_timesteps = len(timestamps)

    df = pd.DataFrame(
        {
            "timestamp": np.tile(timestamps, n_locations),
            "lon": np.repeat(geometry.lons, n_timesteps),
            "lat": np.repeat(geometry.lats, n_timesteps),
        }
    )

    for code in geometry.codes:
        if code.code==11: variable_name='E_AIR_TEMPERATURE'
        elif code.code==61: variable_name='E_RAINFALL'
        elif code.code==204: variable_name='E_SOLAR'
        elif code.code==52: variable_name='E_RELATIVE_HUMIDITY'
        elif code.code==32: variable_name='E_WIND_SPEED'
        elif code.code==180: variable_name='E_WIND_GUST'
        elif code.code==17: variable_name='E_DEWPOINT'
        
        if code.aggregation=='mean': aggregation_name='daily_avg'
        elif code.aggregation=='sum': aggregation_name='daily_sum'
        elif code.aggregation=='min': aggregation_name='daily_min'
        elif code.aggregation=='max': aggregation_name='daily_max'
        
        name = variable_name + "_" + aggregation_name + '_dataset'
        df[name] = list(code.timeIntervals[0].data)

    return df