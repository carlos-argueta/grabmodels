from GraphTools import TextsToJamminFeaturesTransformer, TextsToPathwaysFeaturesTransformer \
    , TextsToMinusnetFeaturesTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def get_pathways_model(graph_sarcasm_dataset, step, stopwords_list, **kwargs):
    # Vectorization
    tweets_to_pathways_feats = TextsToPathwaysFeaturesTransformer(graph_sarcasm_dataset, step,
                                                                  stopwords_list,
                                                                  **kwargs)

    # Classifier
    logreg = LogisticRegression(random_state=42, n_jobs=8, max_iter=1000)

    # Cross-validation
    tuned_parameters = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}]
    clf = GridSearchCV(logreg, tuned_parameters)

    prepare_scale_and_predict_pathways_pipeline = Pipeline([
        ('preparation', tweets_to_pathways_feats),
        ('scaling', StandardScaler()),
        ('log_reg_CV', clf)
    ])

    return prepare_scale_and_predict_pathways_pipeline


def get_minusnet_model(graph_sarcasm_dataset, graph_news_datsaset, step, stopwords_list, **kwargs):
    kwargs["graph_news_datsaset"] = graph_news_datsaset

    # Vectorization
    tweets_to_minusnet_feats = TextsToMinusnetFeaturesTransformer(graph_sarcasm_dataset, step,
                                                                  stopwords_list,
                                                                  **kwargs)

    # Classifier
    logreg = LogisticRegression(random_state=42, n_jobs=8, max_iter=1000)

    # Cross-validation
    tuned_parameters = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}]
    clf = GridSearchCV(logreg, tuned_parameters)

    prepare_scale_and_predict_minusnet_pipeline = Pipeline([
        ('preparation', tweets_to_minusnet_feats),
        ('scaling', StandardScaler()),
        ('log_reg_CV', clf)
    ])

    return prepare_scale_and_predict_minusnet_pipeline


def get_jammin_model(graph_sarcasm_dataset, graph_news_datsaset, step, stopwords_list, **kwargs):
    kwargs["graph_news_datsaset"] = graph_news_datsaset

    # Vectorization
    tweets_to_jammin_feats = TextsToJamminFeaturesTransformer(graph_sarcasm_dataset, step,
                                                              stopwords_list,
                                                              **kwargs)

    # Classifier
    logreg = LogisticRegression(random_state=42, n_jobs=8, max_iter=1000)

    # Cross-validation
    tuned_parameters = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}]
    clf = GridSearchCV(logreg, tuned_parameters)

    prepare_scale_and_predict_jammin_pipeline = Pipeline([
        ('preparation', tweets_to_jammin_feats),
        ('scaling', StandardScaler()),
        ('log_reg_CV', clf)
    ])

    return prepare_scale_and_predict_jammin_pipeline
