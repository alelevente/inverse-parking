############### GRADIENT LEAKAGE ANALYZATION ##################################
#***IMPORTS***
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from multiprocessing import Lock
from multiprocessing.pool import ThreadPool

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import requests
import neural_network

#***CONSTANTS***
#Configuration:
TIME_WINDOW = 15*60 #[s]

#GPU:
NUM_GPUS = 4#6
GLOBAL_GPU_MEM_SIZE = 512 #MiB

#Files:
RESULTS_ROOT = "../../01_simulation/04_results/"
SEEDS = ['42', '1234', '1867', '613', '1001']



#***SETUP FUNCTION***
def setup_logical_gpu():
    #prepare the GPU (for supporting possible multiprocessing)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=GLOBAL_GPU_MEM_SIZE)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
            
            
#***TRAINING FUNCTION***
def create_difference_dataset(global_predict, vehicle_predict):
    '''
        Creates a per parking list of the differences in prediction between the global and
        a vehicle's model.
        
        Parameters:
            - global_predict: prediction of the global model per parking lots
            - vehicle_predict: prediction of a local vehicle per parking lots
            
        Returns:
            - prediction differences
    '''
    p_diffs = {}

    for p in parkings:
        pred_vehicle = np.array(vehicle_predict[p])
        pred_global = global_predict[p]
        
        p_diffs[p] = (pred_vehicle-pred_global)**2
                 
    return p_diffs

def predict_eval_positions(p_diffs, true_parkings, num_parking_lots = 10):
    '''
        Maliciously infers possibly visited parking lots from
        prediction differences. Then evaluates the success rate of the attacker.
        
        Parameters:
            - p_diffs: prediction differences
            - true_parkings: list of the parking lots which were visited by a vehicle
            - num_parking_lots: how many lots try to guess
            
        Returns:
            - the successfully identified parking lots (out of the prescribed num_parking_lots)
    '''
    
    p_diff_means = {}
    for p in parkings:
        p_diff_means[p] = np.mean(p_diffs[p])

    p_diff_series = pd.Series(p_diff_means)

    #converting to sets to be able to get the prediction as an intersection
    predicted_ps = set(p_diff_series.nlargest(num_parking_lots).index)
    true_parkings = set(true_parkings)
    intersection = predicted_ps.intersection(true_parkings)
    return intersection

def search_nearest_move_bin_(pred_time, move_bins_bins, move_bins_counts):
    '''
        From a moving time histogram and a prediction time, it calculates the
        offset between a true moving time and the prediction time.
        
        Parameters:
            - pred_time: predicted moving time in [0,1) range
            - move_bins_bins: x values of the histogram (bin ranges)
            - move_bins_counts: y values of the histogram
            
        Returns:
            - an offset, if negative: the predicted value is later than the closest true moving
    '''
    
    ref_idx, = np.where(move_bins_bins == pred_time)[0]
    rel_idx = 0
    #searching backward:
    while (ref_idx+rel_idx >= 0) and (move_bins_counts[ref_idx+rel_idx]==0):
        rel_idx -= 1
    down_step = 1
    if rel_idx<=0: #found some data
        down_step = rel_idx
    rel_idx = 0
    #searching forward:
    while (ref_idx+rel_idx < len(move_bins_counts)) and (move_bins_counts[ref_idx+rel_idx]==0):
        rel_idx += 1
    if rel_idx < (24*60*60)//TIME_WINDOW: #found some data
        if down_step == 1:
            return rel_idx
        else:
            return down_step if abs(down_step)<rel_idx else rel_idx
    else:
        return None if down_step == 1 else down_step

def predict_eval_time(p_diffs, true_moving_times, time_window=900):
    '''
        Maliciously infers possible moving time from
        prediction differences. Then evalutes how many time windows
        the prediction is away from a true moving of the vehicle.
        
        Parameters:
            - p_diffs: prediction_differences
            - true_moving_times: data series describing the true moving times
            - time_window: how long is a time window in seconds
            
        Returns:
            - an offset, how many time window is away the best prediction from
              a true moving of the vehicle
    '''
    
    time_diffs = {}
    for p in parkings:
        for t in range(0, 24*60*60, time_window):
            if t in time_diffs:
                time_diffs[t] += np.mean(p_diffs[p][t:t+time_window])
            else:
                time_diffs[t] = np.mean(p_diffs[p][t:t+time_window])
    
    time_diffs_series = pd.Series(time_diffs) #to be able to run handy functions
    
    prediction_diff_rates_x = np.arange(time_window, 24*60*60, time_window) #1 step shorter because of the differentiation
    prediction_diff_rates_y = np.abs(np.diff(time_diffs_series.values)) #|d/dt(time_diff(x, t))|
    prediction_diff_rates = pd.Series(data = prediction_diff_rates_y, index = prediction_diff_rates_x)
    pred_time = prediction_diff_rates.index[prediction_diff_rates.argmax()]/(24*60*60)
    #creating the histogram:
    move_bins_counts, move_bins_bins = np.histogram(true_moving_times, bins=np.arange(0, 24*60*60, time_window)/(24*60*60))
    #calculating the offset:
    offset = search_nearest_move_bin_(pred_time, move_bins_bins, move_bins_counts)
    return offset


def train_vehicle(configuration):
    vehicle = configuration["vehicle_id"]
    p_data = configuration["p_data"]
    true_parkings = configuration["true_parkings"] #for the vehicle
    oh = configuration["one_hot_encoding"]
    
    #preparing training data:
    p_train = p_data[p_data["veh_id"] == vehicle]
    X_train = p_train.drop(columns=["veh_id", "time", "occupancy", "seed"])
    y_train = p_train["occupancy"]
    
    #sending 
    payload = {
        "one_hot_encoding": oh,
        "train_features": X_train.values.tolist(),
        "train_labels": y_train.values.tolist(),
        "model_weights": global_weights,
        "epochs": 1
    }
    r = requests.post("http://localhost:5000/compute",
                      json = payload)
    response = json.loads(r.text)
    vehicle_predictions = response["predictions"]
        
    p_diffs = create_difference_dataset(global_predictions, vehicle_predictions)
    pred_lots = predict_eval_positions(p_diffs, true_parkings)
    offset = predict_eval_time(p_diffs, p_train["time_of_day"])
    
    result_lock.acquire()
    results[vehicle] = {
        "positions": list(pred_lots),
        "time_offset": offset
    }
    result_lock.release()
            
if __name__ == "__main__":
    setup_logical_gpu()
    
    
    #loading the list selected vehicles:
    with open("../veh_list.json", "r") as f:
        saved_vehs = json.load(f)
    vehicles = saved_vehs["test_vehs"][:5]
    
    #reading the parking dataset:
    p_data = pd.DataFrame()
    #processing data files:
    for s in SEEDS:
        filename = RESULTS_ROOT+f'poccup_by_vehs_{s}.csv'
        pf = pd.read_csv(filename)
        pf["seed"] = [s]*len(pf)
        p_data = pd.concat([p_data, pf])

    parkings = p_data["parking_id"].unique()
    #preparing ground truth for true parking positions:
    true_parkings = {}
    for v in vehicles:
        true_parkings[v] = p_data[p_data["veh_id"] == v]["parking_id"].unique()

    p_data = pd.get_dummies(p_data, columns=["parking_id"])
    p_data["time"] = p_data["time"] - 4*24*60*60
    p_data["time"] = p_data["time"].astype(int)
    p_data["time_of_day"] = (p_data["time"] - (p_data["time"] // (24*60*60))*24*60*60) / (24*60*60) #converting to 0.0-1.0 and removing periodicity
    
    with open("../one_hot_encoding_dict.json") as f:
        oh_encoding_dict = json.load(f)
    
    #loading global model to GPU0 and making predictions:
    with tf.device(tf.config.list_logical_devices('GPU')[0].name):
        global_model = tf.keras.models.load_model("../saved_models/pretrained")
        n_parkings = len(parkings)
        
        global_predictions = {}
        for p in parkings:
            test_data = np.zeros((24*60*60, n_parkings+1))
            i = oh_encoding_dict[p]
            for j in range(24*60*60):
                test_data[j, i] = 1 # setting the one hot encoding
                test_data[j, -1] = j/(24*60*60) # setting the time of day
                
            global_predictions[p] = global_model.predict(test_data, batch_size=10000)
        
            
        global_weights = neural_network.encode_weights(global_model.get_weights())
    
    results = {}
    result_lock = Lock() # handles access to the results object
    
    #collecting configs for multiprocessing training:
    train_config = []
    for i, v in enumerate(vehicles):
        gpu_idx = i%(NUM_GPUS)
        config = {
            "vehicle_id": v,
            "p_data": p_data,
            "true_parkings": true_parkings[v],
            "one_hot_encoding": oh_encoding_dict
        }
        train_config.append(config)
        
    with ThreadPool(NUM_GPUS) as pool:
        pool.map(train_vehicle, train_config)
        
    with open("vehicle_results.json", "w") as f:
        json.dump(results, f)