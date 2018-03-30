from flask import Flask
from flask import request
import sys
import json
import random

sys.path.insert(0,"/home/ubuntu/Spring2018/Neural_Network") # this lets us use the neural network module even tho its in a different folder

from better_neural_network import Neural_Network

app = Flask(__name__)
app.debug = True

@app.route('/')
def hello_world():
  return 'Hello from Flask!'



@app.route('/run_nn')
def run_nn():
	try:
		pass
		id = "/home/ubuntu/flaskapp/" + request.args.get("id")
		data = {}
		with open("/home/ubuntu/flaskapp/nn_names.json", "r") as f:
			data = json.load(f)
		#return data.keys()[0]
		size = data[id]
		inputs = json.loads(request.args.get("inputs"))
		myNN = Neural_Network(size, id)
		myNN.read_data()
		#return json.dumps(size) + " " + json.dumps(id)
		return json.dumps(myNN.get_output(inputs))
	except Exception as e:
		return("ERROR: " + str(e))




@app.route('/train_nn')
def train_nn():
	try:
		training_data_link = request.args.get("link")
		layer_sizes = json.loads(request.args.get("sizes"))
		training_runs = min(json.loads(request.args.get("training_runs")), 2000)
		inputs_and_outputs = json.loads(request.args.get("io"))
		id = "/home/ubuntu/flaskapp/" + request.args.get("id")
		myNN = Neural_Network(layer_sizes, id)
		myNN.learn_from_url(training_data_link, inputs_and_outputs, training_runs)
		#with open("/home/ubuntu/flaskapp/test1.txt", "wb") as f:
		#	f.write("ayyyyy")
		myNN.save_data()
		data = {}
		with open("/home/ubuntu/flaskapp/nn_names.json", "r") as f:
			data = json.load(f)
       		if id in data.keys():
			return "error: there is already a neural network with that name, please pick another name"
		data[id] = layer_sizes
        	with open("/home/ubuntu/flaskapp/nn_names.json", "wb") as f:
                	json.dump(data, f)
		return "got here" + str(training_data_link) + json.dumps(layer_sizes[0]) + str(training_runs) + json.dumps(inputs_and_outputs[0])
	except Exception as e:
		return(str(e))


if __name__ == '__main__':
  app.run()
