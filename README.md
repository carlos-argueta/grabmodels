# GrabModels
Repository with code to replicate the experiments of the upcoming paper Graph-Based Compact Textual Models for the Automatic Detection of Sarcasm.

GrabModels or Graph-Based Compact Models, is a tool to create word and pattern embeddings for text classifiers. It converts text datasets to graphs in order to learn representations for the vertices of the graph. 

Note: GrabModels is only compatible with Python 3.

## Installation

1. Clone the repository with $ git clone https://github.com/carlos-argueta/grabmodels.git
1. (Optional) Create a virtual environment
1. Enter the grabmodels folder and install dependencies: 
      1. $ cd grabmodels
      1. $ pip3 install -r requirements.txt
1. Install DeepWalk
     1. $ git clone https://github.com/phanein/deepwalk.git
     1. $ cd deepwalk
     1. $ pip install -r requirements.txt (Note: DeepWalk uses Python 2)
     1. $ python setup.py install

## Usage

Note: The following instructions are for Ubuntu and may apply to other similar unix-based systems. For Windows, some steps may need to be modified.

1. If this is the first time to use the program, you may need to make sure that the NLTK's stopwords are available. In a new terminal do the following:
     1. $ python3
     1. &gt;&gt;&gt; import nltk
     1. &gt;&gt;&gt; nltk.download('stopwords')
     1. &gt;&gt;&gt; quit()
     

2. Within the folder grabmodels run $ python3 grabmodels.py -t <graph_type>


Replace <graph_type> with pathways, minusnet, or jammin to create a classifier end-to-end with one of the three different graph creation approaches.

### Hyperparameters

The code is configured to run the complete pipeline with the best reported hyperparameters for each method in the paper. At this moment, in order to try other parameter combinations, you will have to open the file grabmodels.py and within one of the functions run_pathways_example(), run_minusnet_example(), and run_pathways_example(), add the corresponding parameters to the function calls get_pathways_model(), get_minusnet_model(), or get_jammin_model().

The available hyperparameters are:

Hyperparameter | Description
------------ | -------------
graph_sarcasm_dataset | The dataset with sarcastic texts
step | The maximum separation between words when building the graph
stopwords_list | List of stopwords to remove. Only used if remove_stop_words = True
graph_news_datsaset=None | The dataset with neutral texts. Only used with Minusnet and Jammin graphs
graph_sarcasm_dataset2=None | Second optional dataset with sarcasm texts to generate the Jammin patterns from. If None, the mandatory graph_sarcasm_dataset is used instead
minus_diff_th=0.0 | The threshold used when building Minusnet graphs to adjust the edge weights of the graph
cc_th=0.1 | Clustering coefficient threshold used to determine the Topic words to build the Jammin patterns
centrality_th=0.0001 | Betweenness Centrality threshold used to determine the Connector words to build the Jammin patterns
meta_patterns=[] | The patterns templates used to build the Jammin patterns
min_freq=1 | Minimum frequency of a pattern to be retained in the final list
remove_stop_words=False | Remove stop words when building the graphs
stemming=False | Stem words when building the graphs
replace_entities=False | Replace Twitter entities when building the graphs
rep_size=128 | DeepWalk embedding representation size 
workers=-1 | Number of processes to use, -1 means use as many as there are cores in the CPU
verbose=True | Controls whether progress messages are printed or not

## Issues
You can ignore several warnings of the form: 

*".local/lib/python3.6/site-packages/joblib/externals/loky/backend/resource_tracker.py:304: UserWarning: resource_tracker: /dev/shm/joblib_memmapping_folder_3714_d154fc4344634825bc067a9c6bafb91b_ca7e0135c15e4ab7a80f8c67136d4786/3714-139675070605072-4c051f2e9967473ab88d0caf0aff7d0a.pkl: FileNotFoundError(2, 'No such file or directory')"*
