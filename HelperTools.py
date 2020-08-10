import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, \
    classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix


def union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list


def load_stopwords(extend=True):
    stopwords_str = "#sarcasm, ’,a, a’s, able, about, above, according, accordingly, across, actually, after, " \
                    "afterwards, again, against, ain’t, all, allow, allows, almost, alone, along, already, also, " \
                    "although, always, am, among, amongst, an, and, another, any, anybody, anyhow, anyone, anything, " \
                    "anyway, anyways, anywhere, apart, appear, appreciate, appropriate, are, aren’t, around, as, " \
                    "aside, ask, asking, associated, at, available, away, awfully, be, became, because, become, " \
                    "becomes, becoming, been, before, beforehand, behind, being, believe, below, beside, besides, " \
                    "best, better, between, beyond, both, brief, but, by, c’mon, c’s, came, can, can’t, cannot, cant, " \
                    "cause, causes, certain, certainly, changes, clearly, co, com, come, comes, concerning, " \
                    "consequently, consider, considering, contain, containing, contains, corresponding, could, " \
                    "couldn’t, course, currently, definitely, described, despite, did, didn’t, different, do, does, " \
                    "doesn’t, doing, don’t, done, down, downwards, during, each, edu, eg, eight, either, else, " \
                    "elsewhere, enough, entirely, especially, et, etc, even, ever, every, everybody, everyone, " \
                    "everything, everywhere, ex, exactly, example, except, far, few, fifth, first, five, followed, " \
                    "following, follows, for, former, formerly, forth, four, from, further, furthermore, get, gets, " \
                    "getting, given, gives, go, goes, going, gone, got, gotten, greetings, had, hadn’t, happens, " \
                    "hardly, has, hasn’t, have, haven’t, having, he, he’s, hello, help, hence, her, here, here’s, " \
                    "hereafter, hereby, herein, hereupon, hers, herself, hi, him, himself, his, hither, hopefully, " \
                    "how, howbeit, however, , i, i’d, i’ll, i’m, i’ve, ie, if, ignored, immediate, in, inasmuch, inc, " \
                    "indeed, indicate, indicated, indicates, inner, insofar, instead, into, inward, is, isn’t, it, " \
                    "it’d, it’ll, it’s, its, itself, just, keep, keeps, kept, know, knows, known, last, lately, " \
                    "later, latter, latterly, least, less, lest, let, let’s, like, liked, likely, little, look, " \
                    "looking, looks, ltd, mainly, many, may, maybe, me, mean, meanwhile, merely, might, mine, more, " \
                    "moreover, most, mostly, much, must, my, myself, name, namely, nd, near, nearly, necessary, need, " \
                    "needs, neither, never, nevertheless, new, next, nine, no, nobody, non, none, noone, nor, " \
                    "normally, not, nothing, novel, now, nowhere, obviously, of, off, often, oh, ok, okay, old, on, " \
                    "once, one, ones, only, onto, or, other, others, otherwise, ought, our, ours, ourselves, out, " \
                    "outside, over, overall, own, particular, particularly, per, perhaps, placed, please, plus, " \
                    "possible, presumably, probably, provides, que, quite, qv, rather, rd, re, really, reasonably, " \
                    "regarding, regardless, regards, relatively, respectively, right, said, same, saw, say, saying, " \
                    "says, second, secondly, see, seeing, seem, seemed, seeming, seems, seen, self, selves, sensible, " \
                    "sent, serious, seriously, seven, several, shall, she, should, shouldn’t, since, six, so, some, " \
                    "somebody, somehow, someone, something, sometime, sometimes, somewhat, somewhere, soon, sorry, " \
                    "specified, specify, specifying, still, s, sub, such, sup, sure, t’s, take, taken, tell, tends, " \
                    "th, than, thank, thanks, thanx, that, that’s, thats, the, their, theirs, them, themselves, then, " \
                    "thence, there, there’s, thereafter, thereby, therefore, therein, theres, thereupon, these, they, " \
                    "they’d, they’ll, they’re, they’ve, think, third, this, thorough, thoroughly, those, though, " \
                    "three, through, throughout, thru, thus, to, together, too, took, toward, towards, tried, tries, " \
                    "truly, try, trying, twice, two, un, under, unfortunately, unless, unlikely, until, unto, up, " \
                    "upon, us, use, used, useful, uses, using, usually, value, various, very, via, viz, vs, want, " \
                    "wants, was, wasn’t, way, we, we’d, we’ll, we’re, we’ve, welcome, well, went, were, weren’t, " \
                    "what, what’s, whatever, when, whence, whenever, where, where’s, whereafter, whereas, whereby, " \
                    "wherein, whereupon, wherever, whether, which, while, whither, who, who’s, whoever, whole, whom, " \
                    "whose, why, will, willing, wish, with, within, without, won’t, wonder, would, would, wouldn’t, " \
                    "yes, yet, you, you’d, you’ll, you’re, you’ve, your, yours, yourself, yourselves, zero "

    stopwords_list1 = []
    stopwords_list2 = stopwords.words('english')

    if extend:
        for stop in stopwords_str.split(","):
            stopwords_list1.append(stop.strip())
        # print(len(stopwords_list1), len(stopwords_list2))

        return union(stopwords_list1, stopwords_list2)
    else:
        stopwords_list2.append("#sarcasm")
        # print(len(stopwords_list2))
        return stopwords_list2


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    my_labels = ["sarcasm", "neutral", "emotion"]
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(my_labels))
    plt.xticks(tick_marks, my_labels, rotation=45)
    plt.yticks(tick_marks, my_labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # plt.savefig('images/experiments/%s.png'%title)


def evaluate_prediction(predictions, target, title="Confusion matrix"):
    my_labels = ["sarcasm", "neutral", "emotion"]

    p_w = precision_score(target, predictions, average='weighted')
    p_mi = precision_score(target, predictions, average='micro')
    p_ma = precision_score(target, predictions, average='macro')
    r_w = recall_score(target, predictions, average='weighted')
    r_mi = recall_score(target, predictions, average='micro')
    r_ma = recall_score(target, predictions, average='macro')
    f1_w = f1_score(target, predictions, average='weighted')
    f1_mi = f1_score(target, predictions, average='micro')
    f1_ma = f1_score(target, predictions, average='macro')

    metrics = precision_recall_fscore_support(target, predictions)

    accuracy = accuracy_score(target, predictions)

    cm = confusion_matrix(target, predictions, labels=my_labels)

    print()
    print("Classification Report")

    print(classification_report(target, predictions))
    print()
    print(metrics)
    print()
    print('accuracy %s' % accuracy)
    print()
    print("macro results are")
    print("average precision is %f" % (p_ma))
    print("average recall is %f" % (r_ma))
    print("average f1 is %f" % (f1_ma))
    print()
    print("micro results are")
    print("average precision is %f" % (p_mi))
    print("average recall is %f" % (r_mi))
    print("average f1 is %f" % (f1_mi))
    print()
    print("weighted results are")
    print("average precision is %f" % (p_w))
    print("average recall is %f" % (r_w))
    print("average f1 is %f" % (f1_w))
    print()

    print('confusion matrix\n %s' % cm)
    print('(row=expected, col=predicted)')

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, title + ' Normalized')

    return accuracy

