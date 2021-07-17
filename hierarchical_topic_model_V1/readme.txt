1. The parameter “mallet_path” inside "myMenuNavigator\config_project.ini" may need to be changed so to the location of mallet is the correct one.

2. For running the program, we will execute:
	
	C:\hierarchical_topic_model\myMenuNavigator>python main.py --p C:\project --source C:\mallet\data_news_txt_500

	When running the program, it is important that we insert for both “—p” and “—source” options the full path to each of the folders:
	• “—p”: Project folder where the results are going to be saved
	• “—source”: Folder where the data that is going to be used for training the model is save

3. Activate configuration file (option 1) in order to be able to reuse the same project file for other executions.