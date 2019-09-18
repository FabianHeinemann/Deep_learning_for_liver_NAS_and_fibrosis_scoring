""" 
Copyright (c) 2019, Fabian Heinemann, Gerald Birk, Birgit Stierstorfer; Boehringer Ingelheim Boehringer Ingelheim Pharma GmbH & Co KG
All rights reserved.
 
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials # provided with the distribution.
 
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import sys
import argparse
import yaml
from cnn_utils import cnn_utils
import time
	
def main(args):
	# Open the configuration YAML file
	# given as command line argument
	with open(args.config, "r") as file:		
		# Measure start time
		start = time.time()
	
		# Get arguments from YAML		
		config_yaml = yaml.load(file)											
		model_path = config_yaml["model"]["model_path"]	
		model_file_name = config_yaml["model"]["model_file_name"]				
		list_of_classes = config_yaml["model"]["list_of_classes"]		
		thresholds_json = config_yaml["model"]["thresholds_json"]	
		tile_path = config_yaml["tiles"]["tile_path"]			
		score_name = config_yaml["results"]["score_name"]			
		results_path = config_yaml["results"]["results_path"]					
		experiment_name = config_yaml["results"]["experiment_name"]
		
		# Print arguments
		print("\n-----------------------------\n")		
		print("Model path: \t\t%s\n" % (model_path))		
		print("Model file name: \t%s\n" % (model_file_name))
		print("List of classes: \t%s\n" % (list_of_classes))	
		print("Thresholds json: \t%s\n" % (thresholds_json))
		print("Score name:  \t\t%s\n" % (score_name))
		print("Tile path: \t\t%s\n" % (tile_path))	
		print("Results path: \t\t%s\n" % (results_path))		
		print("Experiment name: \t%s\n" % (experiment_name))
		print("-----------------------------")
		
		# Create a cnn_utils object (handles tiled data and the CNN)					
		cnn_utils_obj = cnn_utils(model_path = model_path, model_file_name = model_file_name, tile_path = tile_path, results_path = results_path, list_of_classes = list_of_classes)

		# Initialize the CNN
		print("\n")
		print("Initializing CNN...")		
		cnn_utils_obj.initialize_model(load_pretrained_model = True)
		print("Model loaded.\n")
		
		# Classify tiles
		classification_result = cnn_utils_obj.classify_tiles()
		
		# Renormalize result and add readout column
		classification_result = cnn_utils_obj.process_results(classification_result)
		
		# Save detailled results
		file_name_detailled_results = cnn_utils_obj.results_path + experiment_name + "_" + score_name + ".csv"
		classification_result.to_csv(file_name_detailled_results, index = False, sep = ";", decimal=",")		
		print("Details saved to: %s" % (file_name_detailled_results))			
		
		# Generate summary results (per experiment, group and animal) and map to pathologist score		
		summary_result = cnn_utils_obj.generate_summary_results(classification_result, score_name, thresholds_json)
		
		# Save summary
		file_name_summary_results = cnn_utils_obj.results_path + experiment_name + "_" + score_name + "_summary.csv"
		summary_result.to_csv(file_name_summary_results, index = False, sep = ";", decimal=",")
		print("Summary saved to: %s" % (file_name_summary_results))
		
		# Print elapsed time
		end = time.time()
		print("Time elapsed: %.1f s" % (end - start))
		
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config_file', action="store", dest="config", help="Filename of config file (*.yaml)", required = True)	
	args = parser.parse_args()
	main(args)