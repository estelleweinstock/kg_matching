import torch
from sentence_transformers import SentenceTransformer, util
import random
import numpy as np
import pickle
import pandas as pd

def calc_prec_rec(lefties, righties, gold_standard, threshold):
    # definition of variables
    matches_all = []
    matches_all_complete = []
    match_count = 0
    false_positives = 0

    # give back matches (in both forms to make it easier to track for me)
    output = util.semantic_search(lefties, righties, top_k=1)
    for i, out in zip(range(0, len(output)), output):
        if (out[0]['score'] > threshold):
            matches_all_complete.append([i, out[0]['corpus_id'], out[0]['score']])
            matches_all.append([i, out[0]['corpus_id']])

    # if match contains member of gold standard, we count it in our performance
    for match in matches_all:
        if any(match[0] in sublist for sublist in gold_standard) or any(match[1] in sublist for sublist in gold_standard):
            if match in gold_standard:
                match_count = match_count + 1 # einf wenn es passt
            else:
                false_positives = false_positives + 1 # wenn er was macht, was aber nicht stimmt

    false_negatives = len(gold_standard) - match_count # alle matches, die er nicht gefunden hat

    try:
        prec =  match_count / (match_count + false_positives) # wenn das gering ist, wählt er zu viele aus
    except:
        prec = 0
    recall = match_count / (match_count + false_negatives) # wenn das gering ist, wählt er nicht genügend aus?, wenn hoch, dann wählt er viele von der tatsächlichen Menge aus
    return prec, recall, matches_all_complete


@torch.no_grad()
def test_harder(mode, model, source_data):
    model.eval()
    pred = model(source_data)
    if mode == "train":
        gold_standard = list(map(list, zip(source_data.train_set_left.tolist(), source_data.train_set_right.tolist())))
    else:
        gold_standard = list(map(list, zip(source_data.test_set_left.tolist(), source_data.test_set_right.tolist())))
    lefties = pred[0]
    righties = pred[1]
    prec, rec, matches = calc_prec_rec(lefties, righties, gold_standard, 0.9)
    return prec, rec, matches

def filter_out_wrong_matches(matches, data):

    # get the respective counterpart of each match
    left_match = [item[0] for item in matches]
    right_match = [item[1] for item in matches]

    # filter out all wrong matches
    try:
        model_matches = list(map(list, zip(left_match, right_match)))
        gold_standard = list(map(list, zip(data.train_set_left.tolist(), data.train_set_right.tolist())))
        for i in model_matches[:]:
            if i in gold_standard:
                model_matches.remove(i)
    except:
        model_matches = [] # wenn es noch keine Matches gibt, bleibt das eine leere Liste

    # get the respective counterpart of each wrong match
    wrong_set_left = [item[0] for item in model_matches]
    wrong_set_right = [item[1] for item in model_matches]

    # concatenates the embeddings of the wrong matches and changes data format
    lefts = torch.cat((data.train_set_left, torch.tensor(wrong_set_left)), 0)
    rights = torch.cat((data.train_set_right, torch.tensor(wrong_set_right)), 0)
    lefts = lefts.type(torch.int64)
    rights = rights.type(torch.int64)

    return lefts, rights, wrong_set_left, wrong_set_right

def create_fake_data(tens1, tens2):
    right_inds_list, left_inds_list = tens1.tolist(), tens2.tolist()
    double_left_list = left_inds_list + left_inds_list
    m = max(right_inds_list)

    complete_list = list(np.arange(0, m))

    non_commoners = list(set(complete_list) ^ set(right_inds_list))
    wrong_right_indices = [random.choice(non_commoners) for x in range(len(left_inds_list))]
    double_right_list = right_inds_list + wrong_right_indices
    double_right_tensor = torch.tensor(double_right_list, dtype=torch.long)
    double_left_tensor = torch.tensor(double_left_list)
    return double_left_tensor, double_right_tensor


def save_wrong_matches_left(version, file, case):
    with open("/content/gdrive/My Drive/1_Studium/2_Master/Master Thesis/2_Projekt/Scratch/all_in/MAGNET/results/trainings/GCN/wrong_sets_left_" + version + "_" + case , "wb") as fp:   #Pickling
        pickle.dump(file, fp)


def save_wrong_matches_right(version, file, case):
    with open("/content/gdrive/My Drive/1_Studium/2_Master/Master Thesis/2_Projekt/Scratch/all_in/MAGNET/results/trainings/GCN/wrong_sets_right_" + version + "_" + case,"wb") as fp:   #Pickling
        pickle.dump(file, fp)


def save_matches(version, file, case):
    with open("/content/gdrive/My Drive/1_Studium/2_Master/Master Thesis/2_Projekt/Scratch/all_in/MAGNET/results/trainings/GCN/matches" + version + "_" + case, "wb") as fp:   #Pickling
        pickle.dump(file, fp)


def save_training_verlauf(losses, precisions_train, recall_train,
                          precision_tests, recall_tests,
                          track, case, model_name, version):
    df = pd.DataFrame(list(zip(losses, precisions_train, recall_train, precision_tests, recall_tests)),
                      columns =['losses', 'precisions_train', 'recalls_train', 'precisions_test', 'recalls_test'])
    path = "/content/gdrive/My Drive/1_Studium/2_Master/Master Thesis/2_Projekt/Scratch/all_in/MAGNET/results/trainings/GCN/" + track + "_" + case + "_" + model_name + "_" + version
    df.to_csv(path + ".csv")


def save_model(model_name, version, model, track, case):
    path = "/content/gdrive/My Drive/1_Studium/2_Master/Master Thesis/2_Projekt/Scratch/all_in/MAGNET/results/trained_models/" + track + "_" + case + "_" + model_name + "_" + version
    torch.save(model.state_dict(), path)
