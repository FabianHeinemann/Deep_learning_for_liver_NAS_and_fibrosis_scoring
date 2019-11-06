import sys
import os
import time as time
import argparse
import yaml
import pandas as pd
from cnn_utils import cnn_utils
	
def main(args):
	# Open the configuration YAML file
	# given as command line argument
	with open(args.config, "r") as file:
		# Measure start time
		start = time.time()
	
		# Get arguments from YAML		
		config_yaml = yaml.load(file)		

		# Scores
		score_list = ["Ballooning", "Inflammation", "Steatosis", "Fibrosis"]
		
		# List of classes
		list_of_classes = {"Ballooning" : ["0", "1", "ignore"], "Inflammation" : ["0", "1", "2", "ignore"], "Steatosis" : ["0", "1", "2", "3" , "ignore"] , "Fibrosis" : ["0", "1", "2", "3", "4", "ignore"]}
		
		# Score name (for output tables, do not change)
		score_names = {"Ballooning" : "Ballooning_sub_score", "Inflammation" : "Inflammation_sub_score", "Steatosis" : "Steatosis_sub_score", "Fibrosis" : "Fibrosis_score"}
		
		# Models
		models = {"Ballooning" : config_yaml["models"]["ballooning_model"], 
		          "Inflammation" : config_yaml["models"]["inflammation_model"],
				  "Steatosis" : config_yaml["models"]["steatosis_model"],
				  "Fibrosis" :  config_yaml["models"]["fibrosis_model"]}
		
		# Thresholds
		thresholds_files = {"Ballooning" : config_yaml["thresholds"]["balloning_thresholds_json"],
		                    "Inflammation" : config_yaml["thresholds"]["inflammation_thresholds_json"],
							"Steatosis" : config_yaml["thresholds"]["steatosis_thresholds_json"],
							"Fibrosis" : config_yaml["thresholds"]["fibrosis_thresholds_json"]}
		
		# Tiles
		tiles = {"Ballooning" : config_yaml["tiles"]["NAS_tile_path"], 
                 "Inflammation" : config_yaml["tiles"]["NAS_tile_path"],
                 "Steatosis" : config_yaml["tiles"]["NAS_tile_path"],
                 "Fibrosis" :  config_yaml["tiles"]["fibrosis_tile_path"]}				  
		
		# Results		
		results_path = config_yaml["results"]["results_path"]
		experiment_name = config_yaml["results"]["experiment_name"]				
		
		# Print arguments		
		print("\n-----------------------------\n")				
		print("scores: \t\t%s\n" % (score_list))
		
		for score, model_file_str in models.items():
			print("%s model: \t%s\n" % (score, os.path.basename(model_file_str)))
		
		for score, thresholds_file_str in thresholds_files.items():
			print("%s thresholds_file: \t%s\n" % (score, os.path.basename(thresholds_file_str)))
			
		print("results_path: \t%s\n" % (results_path))
		print("experiment_name:  \t\t%s\n" % (experiment_name))
		print("-----------------------------")
		
		summary_result = pd.DataFrame()
		
		for score in score_list:
			print("\nCurrent score: %s" % (score))
		
			# Create a cnn_utils object (handles tiled data and the CNN)					
			cnn_utils_obj = cnn_utils(model_path = "", model_file_name = models[score], tile_path = tiles[score], results_path = results_path, list_of_classes = list_of_classes[score])

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
			file_name_detailled_results = cnn_utils_obj.results_path + experiment_name + "_" + score_names[score] + ".csv"
			classification_result.to_csv(file_name_detailled_results, index = False, sep = ";", decimal=".", float_format='%.2f')		
			print("Details saved to: %s" % (file_name_detailled_results))
			
			# Generate summary results (per experiment, group and animal) and map to pathologist score		
			current_summary_result = cnn_utils_obj.generate_summary_results(classification_result, score_names[score], thresholds_files[score])			
			current_summary_result.drop(columns = ["n_tiles", "average_uncertainty"], inplace = True)
			
			if summary_result.empty:
				summary_result = current_summary_result
			else:
				summary_result = pd.merge(summary_result, current_summary_result, how = "left", left_on=["experiment", "group", "animal"], right_on=["experiment", "group", "animal"])
			
		# Compute NAS score
		summary_result["NAS_Score"] = summary_result["Ballooning_sub_score"] + summary_result["Inflammation_sub_score"] + summary_result["Steatosis_sub_score"]
			
		# Save summary			
		file_name_summary_results = results_path + experiment_name + "_summary.csv"
		summary_result.to_csv(file_name_summary_results, index = False, sep = ";", decimal=".", float_format='%.2f')
		print("Summary saved to: %s" % (file_name_summary_results))		
		
		# Print elapsed time
		end = time.time()
		print("Time elapsed: %.1f s" % (end - start))
		
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config_file',action="store", dest="config",help="Filename of config file (*.yaml)", required = True)	
	args = parser.parse_args()
	main(args)