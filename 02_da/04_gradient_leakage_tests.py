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

#***CONSTANTS***
#Configuration:
TIME_WINDOW = 15*60 #[s]

#GPU:
NUM_GPUS = 5#6
GPU_MEM_SIZE = 1024 #MiB

#Files:
RESULTS_ROOT = "../01_simulation/04_results/"
SEEDS = ['42', '1234', '1867', '613', '1001']



#***SETUP FUNCTION***
def setup_logical_gpus():
    '''
        Sets up virtual GPUs and places models onto them.
        Returns:
            - tf models placed on corresponding logical GPUs
    '''
    #prepare the GPU (for supporting possible multiprocessing)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEM_SIZE)]*NUM_GPUS)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    
    #placing the model on the gpus:
    models = [None]
    for i in range(1, NUM_GPUS):
        models.append(None)
    return models
            
            
#***TRAINING FUNCTION***
def create_test_data(p_id, p_data):
    '''
        Creates test data for parking data for a complete day.
    '''
    parking_data = p_data[p_data[f"parking_id_{p_id}"] == 1]
    t = np.arange(0, 1, 1/(24*60*60))
    one_hot = parking_data.drop(columns=["veh_id", "time", "occupancy", "seed", "time_of_day"]).iloc[0]
    one_hot = [one_hot.values]*len(t)
    pred_x = np.array(one_hot)
    pred_x = pd.DataFrame(pred_x)
    pred_x["t"] = t
    return pred_x

def create_difference_dataset(p_data, parkings, vehicle_model, vehicle_gpu, global_model, global_gpu, global_model_lock):
    '''
        Creates a per parking list of the differences in prediction between the global and
        a vehicle's model.
        
        Parameters:
            - p_data: parking lot dataset
            - parkings: list of parking lots
            - vehicle_model: tensorflow model of the vehicle
            - vehicle_gpu: logical device on which the vehicle model resides
            - global_model: tensorflow model of the global model
            - global_gpu: logical device on which the global model resides
            - global_model_lock: lock object to access the global model
            
        Returns:
            - prediction differences
    '''
    p_diffs = {}

    for p in parkings:
        test_data = create_test_data(p, p_data)
        #local, vehicle:
        with tf.device(vehicle_gpu):
            pred_vehicle = vehicle_model.predict(test_data, batch_size=10000)
            
        #global:
        global_model_lock.acquire()
        with tf.device(global_gpu):
            pred_global  = global_model.predict(test_data, batch_size=10000)
        global_model_lock.release()
        
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
    gpu = configuration["gpu"]
    vehicle_model = configuration["model"]
    true_parkings = configuration["true_parkings"] #for the vehicle
    
    #preparing training data:
    p_train = p_data[p_data["veh_id"] == vehicle]
    X_train = p_train.drop(columns=["veh_id", "time", "occupancy", "seed"])
    y_train = p_train["occupancy"]
    
    with tf.device(gpu):
        #initializing the model:
        with tf.device(tf.config.list_logical_devices('GPU')[i].name):
            '''vehicle_model = keras.Sequential([
                layers.Dense(64, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(1)
            ])
            vehicle_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001))
            vehicle_model.build(input_shape=(None,79))'''
            vehicle_model = tf.keras.models.load_model("saved_models/pretrained")
            #models.append(vehicle_model)
            
        #tf.keras.backend.clear_session()
        
        #importing weights of the global model:
        #global_model_lock.acquire()
        #vehicle_model.set_weights(global_model.get_weights())
        #global_model_lock.release()
        
        #training 1 epoch:
        vehicle_model.fit(x=X_train, y=y_train, epochs=1, batch_size=10000, verbose=0)
        
    p_diffs = create_difference_dataset(p_data, parkings,
                    vehicle_model, gpu,
                    global_model, tf.config.list_logical_devices('GPU')[0].name,
                    global_model_lock)
    pred_lots = predict_eval_positions(p_diffs, true_parkings)
    offset = predict_eval_time(p_diffs, p_train["time_of_day"])
    
    result_lock.acquire()
    results[vehicle] = {
        "positions": list(pred_lots),
        "time_offset": offset
    }
    result_lock.release()
            
if __name__ == "__main__":
    models = setup_logical_gpus()
    
    #loading global model to GPU0:
    with tf.device(tf.config.list_logical_devices('GPU')[0].name):
        global_model = tf.keras.models.load_model("saved_models/pretrained")
    global_model_lock = Lock() # handles access to the global model
    
    #loading the list selected vehicles:
    with open("veh_list.json", "r") as f:
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
    
    results = {}
    result_lock = Lock() # handles access to the results object
    
    #collecting configs for multiprocessing training:
    train_config = []
    for i, v in enumerate(vehicles):
        gpu_idx = i%(NUM_GPUS-1)+1
        config = {
            "vehicle_id": v,
            "p_data": p_data, #p_data[p_data["veh_id"] == v],
            "gpu": tf.config.list_logical_devices('GPU')[gpu_idx].name,
            "model": models[gpu_idx],
            "true_parkings": true_parkings[v]
        }
        train_config.append(config)
        
    with ThreadPool(NUM_GPUS-1) as pool:
        pool.map(train_vehicle, train_config)
        
    with open("vehicle_results.json", "w") as f:
        json.dump(results, f)