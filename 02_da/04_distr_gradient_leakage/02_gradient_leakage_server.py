from flask import Flask, request
from flask.logging import default_handler
import logging
import os, signal
import gc

import json
import argparse

import numpy as np
import tensorflow as tf

import threading

import sys
sys.path.append("..")
import neural_network


app = Flask(__name__)
log_level = logging.CRITICAL
app.logger.setLevel(log_level)
app.logger.disabled = True #no logging is needed
logging.getLogger('werkzeug').disabled = True #logging disabled

resource_semaphore = None #semaphore to limit the access to the GPU
resource_set_lock = threading.Lock() #lock to handle the GPU reservations
resource_set = set() #set of the free resources

with open("device_configuration.json") as f: #load configuration file into global var.
    gpu_config = json.load(f)
    
#Creating appropriately sized semaphore:
resource_semaphore = threading.Semaphore(gpu_config["n_logical_gpus"])

#Dividing the GPU into logical devices:
logical_gpus = [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_config["memory_size"])]*gpu_config["n_logical_gpus"]
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        logical_gpus
    )

    resource_set = set(tf.config.list_logical_devices("GPU"))
except:
    pass

def do_training(payload):
    train_features, train_labels = np.array(payload["train_features"]), np.array(payload["train_labels"])
    weights_decoded = neural_network.decode_weights(payload["model_weights"])
    one_hot_encoding = payload["one_hot_encoding"]
    epochs = payload["epochs"]
    
    ############### PROTECTED SECTION #################
    resource_semaphore.acquire() #Wait for a free GPU
    
    resource_set_lock.acquire() #Wait to access the GPU reservation list
    my_gpu = list(resource_set)[0]
    resource_set.remove(my_gpu)
    resource_set_lock.release()
    
    with tf.device(my_gpu):
        nn = neural_network.NeuralNetwork()
        nn.model.build(train_features.shape)
        nn.model.set_weights(weights_decoded)
    
        history = nn.train(train_features, train_labels, epochs = epochs)
    
        parking_predictions = {}
        for p in one_hot_encoding:
            i = one_hot_encoding[p]
            tdp = test_data[i*24*60*60: (i+1)*24*60*60] #test data for testing
            parking_predictions[p] = nn.model.predict(x=tdp, batch_size=10000).tolist()
    
        #release resources:
        del nn
    gc.collect()
    
    resource_set_lock.acquire()
    resource_set.add(my_gpu)
    resource_set_lock.release()
    
    resource_semaphore.release()
    ############ END OF PROTECTED SECTION #############

    return {"predictions": parking_predictions}
   

@app.route("/compute", methods=["POST"])
def compute():
    payload = json.loads(request.get_data(), strict=False)
    answer = do_training(payload)
    gc.collect()
    return json.dumps(answer)

@app.route('/stopServer', methods=['GET'])
def stopServer():
    os.kill(os.getpid(), signal.SIGKILL)
    return json.dumps({})

@app.route("/n_GPUs", methods=["GET"])
def answer_n_gpus():
    return json.dumps({
        "n_gpus": gpu_config["n_logical_gpus"]
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask based learning server")
    parser.add_argument("--port", help="port", default=5000)
    parser.add_argument("--n_parking_lots", help="number of parking lots", default=78)
    args = parser.parse_args()
    
    n_parkings = args.n_parking_lots
    test_data = np.zeros((n_parkings*24*60*60, n_parkings+1))
    for i in range(n_parkings):
        for j in range(24*60*60):
            test_data[i*24*60*60+j, i] = 1 # setting the one hot encoding
            test_data[i*24*60*60+j, -1] = j/(24*60*60) # setting the time of day
    
    
    app.run(host="0.0.0.0", port = args.port) 