from DataTools import *
from HelperTools import *
from ModelTools import *


def run_pathways_example():
    # Get the graph generation sarcasm data
    gsd = load_graph_generation_sarcasm_dataset()

    # Get the model
    model = get_pathways_model(gsd, 3, load_stopwords(True), stemming=True, workers=8)

    # Load the training, validation, and testing datasets
    train, valid = load_training_validation_datasets()
    test = load_testing_datasets()

    # Train the model
    print("\n\nTraining the Model")
    model.fit(train["text"], train["label"])

    # Validate the model
    print("\n\nValidating the Model")
    predicted = model.predict(valid["text"])
    y_valid = valid["label"]
    evaluate_prediction(predicted, y_valid, "Validating")

    # Test the model
    print("\n\nTesting the Model")
    predicted = model.predict(test["text"])
    y_valid = test["label"]
    evaluate_prediction(predicted, y_valid, "Testing")


def run_minusnet_example():
    # Get the graph generation sarcasm data
    gsd = load_graph_generation_sarcasm_dataset()

    # Get the graph generation news dataset
    gnd = load_graph_generation_news_datasets()

    # Get the model
    model = get_minusnet_model(gsd, gnd, 2, load_stopwords(True), minus_diff_th=0.0, replace_entities=True, workers=8)

    # Load the training, validation, and testing datasets
    train, valid = load_training_validation_datasets()
    test = load_testing_datasets()

    # Train the model
    print("\n\nTraining the Model")
    model.fit(train["text"], train["label"])

    # Validate the model
    print("\n\nValidating the Model")
    predicted = model.predict(valid["text"])
    y_valid = valid["label"]
    evaluate_prediction(predicted, y_valid, "Validating")

    # Test the model
    print("\n\nTesting the Model")
    predicted = model.predict(test["text"])
    y_valid = test["label"]
    evaluate_prediction(predicted, y_valid, "Testing")


def run_jammin_example():
    # Get the graph generation sarcasm data
    gsd = load_graph_generation_sarcasm_dataset()

    # Get the graph generation news dataset
    gnd = load_graph_generation_news_datasets()

    # Get the model
    model = get_jammin_model(gsd, gnd, 2, load_stopwords(True), stemming=True, minus_diff_th=0.0, cc_th=0.3,
                             centrality_th=0.0001, min_freq=1, workers=8)

    # Load the training, validation, and testing datasets
    train, valid = load_training_validation_datasets()
    test = load_testing_datasets()

    # Train the model
    print("\n\nTraining the Model")
    model.fit(train["text"], train["label"])

    # Validate the model
    print("\n\nValidating the Model")
    predicted = model.predict(valid["text"])
    y_valid = valid["label"]
    evaluate_prediction(predicted, y_valid, "Validating")

    # Test the model
    print("\n\nTesting the Model")
    predicted = model.predict(test["text"])
    y_valid = test["label"]
    evaluate_prediction(predicted, y_valid, "Testing")


if __name__ == '__main__':
    # Uncomment the examples that you want to run
    #run_pathways_example()
    #run_minusnet_example()
    run_jammin_example()
    

