import re
import random
import operator

import itertools
import subprocess
from multiprocessing import Pool

import numpy as np

import networkx as nx

import gensim
from gensim.models import KeyedVectors

from nltk import ngrams
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer

from sklearn.base import BaseEstimator, TransformerMixin


class GraphTools():
    def __init__(self, graph_sarcasm_dataset, step, stopwords_list, graph_news_datsaset=None,
                 graph_sarcasm_dataset2=None, minus_diff_th=0.0, cc_th=0.1, centrality_th=0.0001, meta_patterns=[],
                 min_freq=1, remove_stop_words=False, stemming=False, replace_entities=False, rep_size=128, workers=-1,
                 verbose=True):

        self.graph_sarcasm_dataset = graph_sarcasm_dataset
        self.graph_news_dataset = graph_news_datsaset
        self.step = step
        self.stopwords_list = stopwords_list
        self.remove_stop_words = remove_stop_words
        self.stemming = stemming
        self.replace_entities = replace_entities

        self.minus_diff_th = minus_diff_th

        self.cc_th = cc_th
        self.centrality_th = centrality_th
        self.min_freq = min_freq
        if not meta_patterns:
            self.meta_patterns = ["TW_TW_CW", "TW_CW_TW", "CW_TW_TW", "CW_CW_TW", "CW_TW_CW", "TW_CW_CW", "TW_CW",
                                  "CW_TW"]
        if not graph_sarcasm_dataset2:
            self.graph_sarcasm_dataset2 = self.graph_sarcasm_dataset
        else:
            self.graph_sarcasm_dataset2 = graph_sarcasm_dataset2
        self.tw_regex = "[^\s]+"  # any non whitespace character repeated one or more times
        self.ht_regex = "#[^\s]+"
        self.um_regex = "@[^\s]+"
        self.url_regex = "(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
        self.minurl_regex = "(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"

        self.workers = workers
        self.rep_size = rep_size
        self.verbose = verbose

        self.stemmer = PorterStemmer()
        self.tknzr = TweetTokenizer()

        self.id2word = {}
        self.word2id = {}

        if self.verbose:
            if remove_stop_words:
                print("Remove stopwords On")

            if replace_entities:
                print("Replace entities On")

            if stemming:
                print("Stemming On")

    def tokenize(self, text):
        if self.replace_entities:
            text = re.sub(r'@[^\s]+', '<usermention>',
                          text)  # re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)', '<usermention>', tweet)
            text = re.sub(r'#[^\s]+', '<hashtag>',
                          text)  # re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9-_]+)', '<hashtag>', tweet)

        tokens = self.tknzr.tokenize(text)
        try:
            text = tokens[0]
        except:
            print("Error")
            print(tokens)
        for token in tokens[1:]:
            text += "\t" + token

        return text

    def process_word(self, word):

        # If URL
        if "http" in word or "https" in word:
            word = "<url>"

        # If mini URL
        if len(word) > 3 and word[0:3] == "co/":
            word = "<minurl>"

        return word

    def process_texts(self, sarcasm=True):
        clean_texts = {}

        nodes = {}
        edges = {}
        # word_edges = {}

        if sarcasm:
            texts = self.graph_sarcasm_dataset
        else:
            texts = self.graph_news_dataset

        for text in texts:

            # Lowercase the tweet
            text = text.lower()

            # Remove #sarcasm
            text = text.replace("#sarcasm", "")

            # Remove leading and trailing space
            text = text.strip()

            # Tokenize
            temp_tokens = self.tokenize(text).split("\t")
            if self.remove_stop_words:
                tokens = [word for word in temp_tokens if word not in self.stopwords_list]
            else:
                tokens = temp_tokens

            clean_tokens = []
            for word in tokens:

                if self.stemming:
                    word = self.stemmer.stem(word)

                clean_tokens.append(self.process_word(word))

            if len(clean_tokens) > 0:
                clean_texts[text] = clean_tokens

        count = 0
        for text in clean_texts:
            tokens = clean_texts[text]

            for w_count, word in enumerate(tokens):
                for i in range(1, self.step + 1):
                    if w_count + i >= len(tokens):
                        break

                    word = word
                    word2 = tokens[w_count + i]

                    nodes[word] = word
                    nodes[word2] = word2

                    edge = word + " " + word2

                    if edge in edges:
                        edges[edge] += 1
                    else:
                        edges[edge] = 1

            count += 1

        if self.verbose:
            print("Processed %d texts" % count)
            print("Found %d edges and %d nodes" % (len(edges), len(nodes)))

        return nodes, edges

    def rank_edges(self, edges):
        sorted_edges = dict(sorted(edges.items(), key=operator.itemgetter(1), reverse=True))
        maximum = 0

        for c, edge in enumerate(sorted_edges):
            if c == 0:
                maximum = float(sorted_edges[edge])
            edges[edge] = float(sorted_edges[edge]) / maximum

        return edges

    def generate_pathways_graph(self):
        if self.verbose:
            print("\nPathways Graph Generation")
        f = open("temp/graph.edgelist", "w")

        nodes = {}
        edges = {}

        t_count = 0
        for t_count, text in enumerate(self.graph_sarcasm_dataset):

            text = text.lower()
            temp_tokens = self.tokenize(text).split("\t")
            if self.remove_stop_words:
                tokens = [word for word in temp_tokens if word not in self.stopwords_list]
            else:
                tokens = temp_tokens

            for w_count, word in enumerate(tokens):

                for i in range(1, self.step + 1):
                    if w_count + i >= len(tokens):
                        break

                    word = word
                    word2 = tokens[w_count + i]

                    if self.stemming:
                        word = self.stemmer.stem(word)
                        word2 = self.stemmer.stem(word2)

                    nodes[word] = word
                    nodes[word2] = word2

                    edge = word + " " + word2
                    # f.write(edge+"\n")

                    if edge in edges:
                        edges[edge] += 1
                    else:
                        edges[edge] = 1

        if self.verbose:
            print("Processed %d texts" % t_count)
            print("Found %d edges and %d nodes" % (len(edges), len(nodes)))

        for idx, node in enumerate(nodes):
            idx += 1

            nodes[node] = idx
            self.id2word[idx] = node

            word = "%s" % node
            self.word2id[word] = str(idx)

        for edge in edges:
            tokens = edge.split()
            f.write("%s %s\n" % (nodes[tokens[0]], nodes[tokens[1]]))

        f.close()

        return nodes, edges

    def generate_minusnet_graph(self):
        if self.verbose:
            print("\nMinusnet Graph Generation")
            print("Step 1: Generate Sarcasm Pathways Graph ...")

        nodes_sar, edges_sar = self.process_texts(sarcasm=True)

        edges_sar = self.rank_edges(edges_sar)

        sorted_edges = dict(sorted(edges_sar.items(), key=operator.itemgetter(1), reverse=True))
        if self.verbose:
            print("\n\nTop 10 Sarcasm edges")
            for c, edge in enumerate(sorted_edges):
                if c == 10:
                    break
                print(edge, sorted_edges[edge])

            print()

        if self.verbose:
            print("Step 2: Generate News Pathways Graph ...")
        nodes_neu, edges_neu = self.process_texts(sarcasm=False)

        edges_neu = self.rank_edges(edges_neu)

        sorted_edges = dict(sorted(edges_neu.items(), key=operator.itemgetter(1), reverse=True))
        if self.verbose:
            print("\n\nTop 10 News edges")
            for c, edge in enumerate(sorted_edges):
                if c == 10:
                    break
                print(edge, sorted_edges[edge])

        if self.verbose:
            print("\nStep 3: Generating Minusnet Graph ...")
        final_nodes = {}
        final_edges = {}

        # For every edge in the sarcasm edges
        for edge in edges_sar:
            # If the edge is also in the neutral edges
            if edge in edges_neu:
                # Compute the new value as the differenec of both values
                value = edges_sar[edge] - edges_neu[edge]

            else:
                # Keep the value as the value of the sarcastic edge
                value = edges_sar[edge]

            # If the value passes the threshold
            if value >= self.minus_diff_th:
                tokens = edge.split()
                if tokens[0] not in final_nodes:
                    final_nodes[tokens[0]] = len(final_nodes) + 1
                if tokens[1] not in final_nodes:
                    final_nodes[tokens[1]] = len(final_nodes) + 1
                final_edge = str(final_nodes[tokens[0]]) + " " + str(final_nodes[tokens[1]])

                if final_edge in final_edges:
                    final_edges[final_edge] = max(value, final_edges[final_edge])

                else:
                    final_edges[final_edge] = value

        if self.verbose:
            print("\nConstructed Minusnet Graph")
            print("Found %d edges and %d nodes" % (len(final_edges), len(final_nodes)))

        f = open("temp/graph.edgelist", "w")
        for node in final_nodes:
            self.id2word[final_nodes[node]] = node

            word = "%s" % node
            self.word2id[word] = str(final_nodes[node])

        for edge in final_edges:
            tokens = edge.split()
            f.write("%s %s\n" % (tokens[0], tokens[1]))

        f.close()

        return final_nodes, final_edges

    def select_words(self, ccs, bcs):
        sorted_ccs = dict(sorted(ccs.items(), key=operator.itemgetter(1), reverse=True))
        sorted_bcs = dict(sorted(bcs.items(), key=operator.itemgetter(1), reverse=True))

        topic_words = {}
        connector_words = {}

        for cc in sorted_ccs:

            label = self.id2word[int(cc)]
            cc_value = ccs[cc]

            if cc_value >= self.cc_th:
                if label in topic_words:
                    print("Repeated topic word " + label + " with value " + str(cc_value))
                topic_words[label] = cc_value

        for bc in sorted_bcs:

            label = self.id2word[int(bc)]
            centrality_value = bcs[bc]

            if centrality_value >= self.centrality_th:
                if label in connector_words:
                    print("Repeated connector word " + label + " with value " + str(centrality_value))
                connector_words[label] = centrality_value

        if self.verbose:
            print("Number of topic words " + str(len(topic_words)))
            print("Number of connector words " + str(len(connector_words)))

        self.tws = dict(sorted(topic_words.items(), key=operator.itemgetter(1), reverse=True))
        self.cws = dict(sorted(connector_words.items(), key=operator.itemgetter(1), reverse=True))

    def tokenize_and_process(self, text):
        tokens = self.tokenize(text).split("\t")
        if self.remove_stop_words:
            tokens = [word for word in tokens if word not in self.stopwords_list]

        clean_tokens = []
        for word in tokens:
            if self.stemming:
                word = self.stemmer.stem(word)

            clean_tokens.append(self.process_word(word))

        return clean_tokens

    def get_instances(self, meta_pattern):

        if self.verbose:
            print("Processing meta patterns", meta_pattern)
            print("Extracting instances from %d texts" % len(self.graph_sarcasm_dataset2))

        instances = {}

        # Get the components of the meta pattern and length
        meta_tokens = meta_pattern.split("_")
        W = len(meta_tokens)

        separator = " "

        texts = {}

        for text in self.graph_sarcasm_dataset2:
            tokens = self.tokenize_and_process(text)

            for i in range(0, len(tokens) - W + 1):

                pattern = ""
                for j in range(W):

                    current = i + j
                    token = tokens[current]

                    # Decide which list of words to use based on the current item of the
                    # meta pattern
                    if meta_tokens[j] == "TW":
                        ws = self.tws

                    elif meta_tokens[j] == "CW":
                        ws = self.cws
                    else:
                        print("Error, wrong meta pattern token")

                    if token in ws:
                        pattern += separator + token
                    else:
                        break

                    if j + 1 == W:
                        if pattern in instances:
                            instances[pattern] += 1
                        else:
                            instances[pattern] = 1

        self.instances[meta_pattern] = dict(sorted(instances.items(), key=operator.itemgetter(1), reverse=True))

    def do_all_get_instances(self):
        self.instances = {}
        for meta_pattern in self.meta_patterns:
            self.get_instances(meta_pattern)

        print(list(self.instances.keys()))

    def escape_reserved_tokens(self, token):
        new_token = ""
        for letter in token:
            output = re.search(r'(\^|\$|\*|\(|\)|\+|\[|\]|\{|\}|\||\.|\?|\\)', letter)
            # print(output)
            if output is not None:
                new_token += "\\" + letter
            else:
                new_token += letter

        return new_token

    def get_patterns(self, meta_pattern):

        patterns = {}
        words = {}

        # Topic words wildcard
        tw_wc = ".+"

        pos = {}
        tokens = meta_pattern.split("_")
        # Get the position to check for TWs in the instances
        for idx, token in enumerate(tokens):
            if token == "TW":
                pos[idx] = idx

        separator = " "

        instances = self.instances[meta_pattern]
        for instance in instances:
            pattern = ""
            tw_inst = ""

            inst_count = instances[instance]
            # print(instance, inst_count)
            tokens = self.escape_reserved_tokens(instance).split()

            # temp to put all the topic words instantiated by the specific pattern.
            temp = {}
            # // temp used to put all the patterns than instantiate a specific instance
            # TreeMap<String, String> temp2 = new TreeMap();

            # Go over the positions of TWs in the meta pattern to replace words at that
            # position in the instance with the wildcard
            for idx in pos:
                # Keep track of all TWs seen
                tw_inst += tokens[idx] + " "

                # Replace word with wildcard
                tokens[idx] = tw_wc

            # Topic words found in current instance
            tw_inst = tw_inst.strip()
            temp[tw_inst] = inst_count

            # Construct the pattern from the instances with TWs replaced by wildcard
            for token in tokens:
                pattern += separator + token

            # Add pattern to the global instances list
            # We use this just to get the global count
            if not tw_inst in self.global_instances:

                self.global_instances[tw_inst] = inst_count
            else:
                self.global_instances[tw_inst] += inst_count

            # Add patterns to the list
            if not pattern in patterns:
                patterns[pattern] = inst_count
            else:
                patterns[pattern] += inst_count

                # Since we already saw this pattern before. Update the list of topic words
                # this pattern instantiates
                temp = words[pattern]
                temp[tw_inst] = inst_count

            # Update the list of topic words for this pattern
            words[pattern] = temp

        self.global_patterns = {**self.global_patterns, **words}
        self.global_patterns2 = {**self.global_patterns2, **patterns}

        self.patterns[meta_pattern] = dict(sorted(patterns.items(), key=operator.itemgetter(1), reverse=True))

    def do_all_get_patterns(self):
        self.patterns = {}
        self.global_instances = {}
        self.global_patterns = {}
        self.global_patterns2 = {}

        for meta_pattern in self.meta_patterns:
            self.get_patterns(meta_pattern)

        print("Total instances " + str(len(self.global_instances)))
        print("Total patterns " + str(len(self.global_patterns)))
        print("Total patterns " + str(len(self.global_patterns2)))

    def select_final_patterns(self):
        delim_regex = " "

        self.final_patterns = {}

        for meta_pattern in self.meta_patterns:
            patterns = self.patterns[meta_pattern]
            for patt in patterns:

                frequency = patterns[patt]

                if frequency >= self.min_freq:

                    tokens = patt.split(" ")
                    pattern = delim_regex
                    for t in tokens:
                        if t == ".+":
                            t = self.tw_regex

                        if t == "<hashtag>":
                            t = self.ht_regex

                        if t == "<usermention>":
                            t = self.um_regex

                        if t == "<url>":
                            t = self.url_regex

                        if t == "<minurl>":
                            t = self.minurl_regex

                        pattern += t + delim_regex

                    self.final_patterns[pattern.strip()] = patt.strip()

        if self.verbose:
            print("Number of final patterns:", len(self.final_patterns))

    def graph_components_from_pattern_match_on_partial_data(self, text):

        nodes = []
        edges = []
        # print(text)
        clean_tokens = self.tokenize_and_process(text)
        if len(clean_tokens) < 3:
            return nodes, edges

        text = " ".join(clean_tokens)
        # print(text)
        n_grams = list(ngrams(clean_tokens, 3))
        # print(n_grams)
        # Keep track of the previous patterns
        prev = []

        # Go over all patterns and find matches
        ps = []
        ms = []
        for pattern in self.final_patterns:

            matches = re.findall(r'(?<!\w)%s(?!\w)' % pattern, text)

            # If there is a match, add the pattern to the list of current patterns
            if len(matches) > 0:
                # print(matches)
                pattern = pattern.replace(self.ht_regex, "<hashtag>")
                pattern = pattern.replace(self.um_regex, "<usermention>")
                pattern = pattern.replace(self.tw_regex, ".+")
                pattern = pattern.replace(self.url_regex, "<url>")

            # Keep track of each unique match and its pattern
            for match in matches:
                ps.append(pattern)
                ms.append(match)

        # Go over all ngrams to find the corresponding matches
        for n_gram in n_grams:
            n_gram_str = ' '.join(n_gram)

            # Keep track of current patterns
            current = []

            # Go over all matches and see if it is part of an ngram
            for idx, match in enumerate(ms):
                # Match might be shorter than n-gram
                if match in n_gram_str:
                    pattern = ps[idx]
                    current.append(pattern)

                    # If there were previous patterns, generate the edges of the graph
            if len(prev) > 0:
                for p in prev:
                    for c in current:
                        if p != c:
                            nodes.append(p)
                            nodes.append(c)

                            edge = p + "\t" + c
                            # print(edge)
                            edges.append(edge)

            # Now the previous become the current
            prev = current

        return nodes, edges

    def graph_from_pattern_match(self):

        print("Generating graph from %d texts." % len(self.graph_sarcasm_dataset))

        p = Pool(processes=self.workers)
        data = p.map(self.graph_components_from_pattern_match_on_partial_data, self.graph_sarcasm_dataset)
        p.close()
        # nodes, edges = data
        print("Done")

        nodes = {}
        edges = {}

        for ns, es in data:
            for n in ns:
                nodes[n] = n
            for e in es:
                if e in edges:
                    edges[e] += 1
                else:
                    edges[e] = 1
        return nodes, edges

    def chunks(self, l, n):
        """Divide a list of nodes `l` in `n` chunks"""
        l_c = iter(l)
        while 1:
            x = tuple(itertools.islice(l_c, n))
            if not x:
                return
            yield x

    def _betmap(self, G_normalized_weight_sources_tuple):
        """Pool for multiprocess only accepts functions with one argument.
        This function uses a tuple as its only argument. We use a named tuple for
        python 3 compatibility, and then unpack it when we send it to
        `betweenness_centrality_source`
        """
        return nx.betweenness_centrality_source(*G_normalized_weight_sources_tuple)

    def betweenness_centrality_parallel(self, G, processes=None):
        if self.verbose:
            print("Computing Centrality with", processes, "processes.")

        """Parallel betweenness centrality  function"""
        p = Pool(processes=processes)
        node_divisor = len(p._pool) * 4
        node_chunks = list(self.chunks(G.nodes(), int(G.order() / node_divisor)))
        num_chunks = len(node_chunks)
        bt_sc = p.map(self._betmap,
                      zip([G] * num_chunks,
                          [True] * num_chunks,
                          [None] * num_chunks,
                          node_chunks))

        # Reduce the partial solutions
        bt_c = bt_sc[0]
        for bt in bt_sc[1:]:
            for n in bt:
                bt_c[n] += bt[n]
        return bt_c

    def generate_jammin_graph(self):
        if self.verbose:
            print("\nJammin Graph Generation")

        # Step 1: Graph Analysis
        G = nx.read_adjlist("temp/graph.edgelist")
        # Obtain the reverse nodes
        # nodes, reverse_nodes, edges = net_file_to_graph_data(("%s.net"%graph_path_prefix))

        # Compute Clustering Coeffs
        if self.verbose:
            print("Computing CC")

        ccs = nx.clustering(G)

        # Compute Betweenness Centrality
        if self.verbose:
            print("Computing BC")

        bcs = self.betweenness_centrality_parallel(G, processes=self.workers)

        # Step 2: Create list of topic and connector words
        if self.verbose:
            print("Generating word lists")
        self.select_words(ccs, bcs)

        # Step 3: Meta Patterns extraction and ranking
        if self.verbose:
            print("Getting and ranking Meta Patterns")

        # Get instances

        self.do_all_get_instances()

        # Step 4: Get patterns
        if self.verbose:
            print("Generating patterns")
        self.do_all_get_patterns()

        # Step 5: Final Patterns Selection
        if self.verbose:
            print("Selecting final patterns")

        self.select_final_patterns()
        print("Done")

        # Step 6: Graph construction from pattern's match
        if self.verbose:
            print("Constructing Graph")

        nodes, edges = self.graph_from_pattern_match()

        if self.verbose:
            print("Processed %d texts" % len(self.graph_sarcasm_dataset))
            print("Found %d edges and %d nodes" % (len(edges), len(nodes)))

        f = open("temp/graph.edgelist", "w")

        for idx, node in enumerate(nodes):
            idx += 1

            nodes[node] = idx
            self.id2word[idx] = node

            word = "%s" % node
            self.word2id[word] = str(idx)

        for edge in edges:
            tokens = edge.split("\t")
            f.write("%s %s\n" % (nodes[tokens[0]], nodes[tokens[1]]))

        f.close()

        return nodes, edges

    def word_averaging(self, words):
        mean = []

        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in self.embeddings.vocab:

                mean.append(self.embeddings.syn0norm[self.embeddings.vocab[word].index])

        if not mean:
            # print("cannot compute similarity with no input %s", words)

            # All words are unseen!
            return [0.0] * self.embeddings.syn0norm[self.embeddings.vocab["1"].index].shape[0]

        mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

    def sentence_to_vector(self, sentence):

        sentence = sentence.lower()

        temp_tokens = self.tokenize(sentence).split("\t")

        # Remove #sarcasm
        while "#sarcasm" in temp_tokens:
            idx = temp_tokens.index("#sarcasm")
            temp_tokens.remove("#sarcasm")

        if self.remove_stop_words:
            words = [word for word in temp_tokens if word not in self.stopwords_list]
        else:
            words = temp_tokens

        word_ids = []

        for word in words:
            if self.stemming:
                word = self.stemmer.stem(word)

            # processed_words.append(word)
            if word in self.word2id:
                word_id = self.word2id[word]
                word_ids.append(word_id)
        # print(word_ids)
        avg = self.word_averaging(word_ids)
        return avg

    def get_embeddings(self):
        if self.verbose:
            print("\nGenerating Embeddings")
        process = subprocess.run(
            ["deepwalk", "--workers", str(self.workers), "--format", "edgelist", "--input", "temp/graph.edgelist",
             "--output", "temp/graph.embeddings", "--representation-size", str(self.rep_size)])
        print(process)

        self.embeddings = KeyedVectors.load_word2vec_format("temp/graph.embeddings", binary=False)

        # These lines need to be run to load the vocab, or later code would fail
        similar_to = random.choice(list(self.word2id.keys()))
        result = self.embeddings.similar_by_word(self.word2id[similar_to])

        if self.verbose:
            print("Done generating Embeddings")


class TextsToGraphFeaturesTransformer(BaseEstimator, TransformerMixin, GraphTools):
    def transform(self, X, y=None):
        if self.verbose:
            print("Changing texts to feats")

        data = []
        for i, line in enumerate(X):
            tokens = self.sentence_to_vector(line)

            data.append(tokens)

        return data


class TextsToPathwaysFeaturesTransformer(TextsToGraphFeaturesTransformer, GraphTools):

    def fit(self, X, y=None):
        # Get pathways graph
        nodes, edges = self.generate_pathways_graph()

        # Get embeddings
        self.get_embeddings()

        return self


class TextsToMinusnetFeaturesTransformer(TextsToGraphFeaturesTransformer, GraphTools):

    def fit(self, X, y=None):
        # Get graph
        nodes, edges = self.generate_minusnet_graph()

        # Get embeddings
        self.get_embeddings()

        return self


class TextsToJamminFeaturesTransformer(TextsToGraphFeaturesTransformer, GraphTools):

    def fit(self, X, y=None):
        # Create the minusnet graph
        nodes, edges = self.generate_minusnet_graph()

        # Create the Jammin graph
        nodes, edges = self.generate_jammin_graph()

        # Get embeddings
        self.get_embeddings()

        return self
