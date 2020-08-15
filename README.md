# grabmodels
Repository with code to replicate the experiments of the upcoming paper Graph-Based Compact Textual Models for the Automatic Detection of Sarcasm.

GrabModels or Graph-Based Compact Models, is a tool to create word and pattern embeddings for text classifiers. It converts text datasets to graphs in order to learn representations for the vertices of the graph.

## Installation

1. Clone the repository with $git clone https://github.com/carlos-argueta/grabmodels.git
1. (Optional) Create a virtual environment
1. Enter the grabmodels folder and install dependencies: 
      $cd grabmodels
      $pip install -r requirements.txt

## Usage

Within the folder grabmodels run
$grabmodels -t <graph_type>

Replace graph type with pathways, minusnet, or jammin to create a classifier end-to-end with one of the three different graph creation approaches.
The code is configured to run the complete pipeline with the best reported parameters for each method. At this moment, in order to try other parameter 
combinations you will have to open the file grabmodels.py and within one of the functions run_pathways_example(), run_minusnet_example(), and run_pathways_example(), add the corresponding parameters to the function calls get_pathways_model(), get_minusnet_model(), or get_jammin_model().
