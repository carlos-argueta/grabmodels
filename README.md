# grabmodels
Repository with code to replicate the experiments of the upcoming paper Graph-Based Compact Textual Models for the Automatic Detection of Sarcasm.

GrabModels or Graph-Based Compact Models, is a tool to create word and pattern embeddings for text classifiers. It converts text datasets to graphs in order to learn representations for the vertices of the graph. GrabModels is only compatible with Python 3.

## Installation

1. Clone the repository with $ git clone https://github.com/carlos-argueta/grabmodels.git
1. (Optional) Create a virtual environment
1. Enter the grabmodels folder and install dependencies: 
      * $ cd grabmodels
      * $ pip3 install -r requirements.txt
1. Install DeepWalk
     1. $ git clone https://github.com/phanein/deepwalk.git
     1. cd deepwalk
     1. pip install -r requirements.txt (Note: DeepWalk uses Python 2)
     1. python setup.py install

## Usage

Note: The following instructions are for Ubuntu and may apply to other similar unix-based systems. For Windows, some steps may need to be modified.

1. If this is the first time to use the program, you may need to make sure that the NLTK's stopwords are available. In a new terminal do the following:
     1. $ python3
     1. &gt;&gt;&gt; import nltk
     1. >>> nltk.download('stopwords')
     1. >>> quit()
     

1. Within the folder grabmodels run
$ python3 grabmodels.py -t <graph_type>

Replace graph type with pathways, minusnet, or jammin to create a classifier end-to-end with one of the three different graph creation approaches.
The code is configured to run the complete pipeline with the best reported parameters for each method. At this moment, in order to try other parameter 
combinations you will have to open the file grabmodels.py and within one of the functions run_pathways_example(), run_minusnet_example(), and run_pathways_example(), add the corresponding parameters to the function calls get_pathways_model(), get_minusnet_model(), or get_jammin_model().
