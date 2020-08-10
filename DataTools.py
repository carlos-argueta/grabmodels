import csv
import pandas as pd


def load_graph_generation_sarcasm_dataset():
    f = open("data/graph_generation_sarcasm_dataset.txt","r")
    sarcasm_data = f.readlines()
    f.close()
    return sarcasm_data


def load_graph_generation_news_datasets():
    f = open("data/graph_generation_news_dataset.txt", "r")
    news_data = f.readlines()
    f.close()
    return news_data


def load_training_validation_datasets():
    path_training = "data/training.tsv"
    path_validation = "data/validation.tsv"
    return pd.read_csv(path_training, sep='\t'), pd.read_csv(path_validation, sep='\t')


def load_testing_datasets():
    path_testing = "data/testing.tsv"
    return pd.read_csv(path_testing, sep='\t',quoting=csv.QUOTE_NONE,error_bad_lines=False)







