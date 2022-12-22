import torch
from .helper_funcs import filter_out_wrong_matches, create_fake_data
import math
import random
import numpy as np


class CosineSimilarityLoss0(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fct = torch.nn.MSELoss()

    def forward(self, data, matches):
        x_one, x_two = self.model(data)
        left_embeddings = torch.index_select(x_one, 0, data.train_set_left)
        right_embeddings = torch.index_select(x_two, 0, data.train_set_right)
        output = torch.cosine_similarity(left_embeddings, right_embeddings)
        return self.loss_fct(output, torch.ones(int(len(left_embeddings))))


class CosineSimilarityLoss1(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fct = torch.nn.MSELoss()

    def forward(self, data, matches, double_left_tensor, double_right_tensor):
        x_one, x_two = self.model(data)
        left_embeddings_double = torch.index_select(x_one, 0, double_left_tensor)
        right_embeddings_double = torch.index_select(x_two, 0, double_right_tensor)
        output = torch.cosine_similarity(left_embeddings_double, right_embeddings_double)
        return self.loss_fct(output, torch.cat(((torch.ones(int(len(left_embeddings_double) / 2)),
                                                 torch.zeros(int(len(left_embeddings_double) / 2))))))  # negatives


class CosineSimilarityLoss2(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fct = torch.nn.MSELoss()

    def forward(self, data, matches):
        x_one, x_two = self.model(data)
        lefts, rights, wrong_set_left, wrong_set_right = filter_out_wrong_matches(matches, data)
        left_embeddings = torch.index_select(x_one, 0, lefts)
        right_embeddings = torch.index_select(x_two, 0, rights)

        output = torch.cosine_similarity(left_embeddings, right_embeddings)

        return self.loss_fct(output, torch.cat(((torch.ones(int(len(data.train_set_left))),
                                                 torch.zeros(int(len(wrong_set_left)))))))  # negatives


class CosineSimilarityLoss3(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fct = torch.nn.MSELoss()

    def forward(self, data, matches, double_left_tensor, double_right_tensor):
        x_one, x_two = self.model(data)
        left_embeddings_double = torch.index_select(x_one, 0, double_left_tensor)
        right_embeddings_double = torch.index_select(x_two, 0, double_right_tensor)
        output = torch.cosine_similarity(left_embeddings_double, right_embeddings_double)
        return self.loss_fct(output, torch.cat(((torch.ones(int(len(left_embeddings_double) / 2)),
                                                 torch.zeros(int(len(left_embeddings_double) / 2))))))  # negatives


class CosineSimilarityLoss4(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fct = torch.nn.MSELoss()

    def forward(self, data, lefts, rights):
        x_one, x_two = self.model(data)

        left_embeddings = torch.index_select(x_one, 0, lefts)
        right_embeddings = torch.index_select(x_two, 0, rights)
        length_of_anti_alignments = len(lefts) - (len(data.train_set_left))

        output = torch.cosine_similarity(left_embeddings, right_embeddings)

        if length_of_anti_alignments == 0:
            return self.loss_fct(output, torch.ones(int(len((data.train_set_left)))))
        else:
            return self.loss_fct(output, torch.cat(((torch.ones(int(len(data.train_set_left))),
                                                     torch.zeros(int(length_of_anti_alignments))))))  # negatives


def create_anti_alignment_v41(epoch, matches, data):
    # Check, if wrong matches exist to train the model with
    lefts, rights, wrong_set_left, wrong_set_right = filter_out_wrong_matches(matches, data)
    max_length = len(data.train_set_right)

    if len(wrong_set_left) > 0:

        # Create mixed version of anti-alignments

        # 1. Create the proportions
        prop_model = epoch / 1000
        proportion_of_model_faults = math.floor(prop_model * 100) / 100.0
        proportion_of_model_random = 1 - proportion_of_model_faults

        # 2. Choose wrong examples from model proportionately
        count_of_model_faults = math.floor(
            proportion_of_model_faults * max_length)  # calculation, how many examples are considered

        amount_that_can_be_chosen = min(count_of_model_faults, len(wrong_set_right))
        chosen_indices = random.sample(range(len(wrong_set_right)), amount_that_can_be_chosen)
        # print("chosen_indices", chosen_indices)
        picked_left = list(np.array(wrong_set_left)[chosen_indices])
        picked_right = list(np.array(wrong_set_right)[chosen_indices])

        # 3. Create random anti-alignments

        random_length = round(proportion_of_model_random * max_length)
        # print('random length', random_length)
        m = max(data.train_set_right.tolist())  # max value of mapped_to
        complete_list = list(np.arange(0, m))
        non_commoners = list(set(complete_list) ^ set(data.train_set_right.tolist()))
        wrong_right_indices = [random.choice(non_commoners) for x in range(0, (random_length))]
        # print('len of random anti alignments: ', len(wrong_right_indices))

        # 4. Pack it all together

        # First part is the true alignment, then the random alignment and then the model-generated faults
        lefts = torch.cat(
            (data.train_set_left, data.train_set_left[:len(wrong_right_indices)], torch.tensor(picked_left)), 0)
        rights = torch.cat((data.train_set_right, torch.tensor(wrong_right_indices), torch.tensor(picked_right)), 0)

        lefts = lefts.type(torch.int64)
        rights = rights.type(torch.int64)
        return lefts, rights

    # In the case that no model output exists yet, only the random version happens
    else:
        lefts, rights = create_fake_data(data.train_set_right, data.train_set_left)
        return lefts, rights


def create_anti_alignment_v42(epoch, matches, data):
    # Check, if wrong matches exist to train the model with
    lefts, rights, wrong_set_left, wrong_set_right = filter_out_wrong_matches(matches, data)
    max_length = len(data.train_set_right)

    if len(wrong_set_left) > 0:

        # Create mixed version of anti-alignments

        # 1. Create the proportions
        prop_model = epoch / 2000
        proportion_of_model_faults = math.floor(prop_model * 100) / 100.0
        proportion_of_model_random = 1 - proportion_of_model_faults
        # print('prop model: ' , proportion_of_model_faults)
        # print('prop random: ' , proportion_of_model_random)

        # 2. Choose wrong examples from model proportionately
        count_of_model_faults = math.floor(
            proportion_of_model_faults * max_length)  # calculation, how many examples are considered
        # print("count_of_model_faults", count_of_model_faults)
        # chosen_indices = [random.randint(0, len(wrong_set_left)) for x in range(count_of_model_faults)]
        amount_that_can_be_chosen = min(count_of_model_faults, len(wrong_set_right))
        chosen_indices = random.sample(range(len(wrong_set_right)), amount_that_can_be_chosen)
        # print("chosen_indices", chosen_indices)
        picked_left = list(np.array(wrong_set_left)[chosen_indices])
        picked_right = list(np.array(wrong_set_right)[chosen_indices])
        # print('picked_left: ', picked_left)
        # print('picked_left: ', picked_right)

        # 3. Create random anti-alignments

        random_length = round(proportion_of_model_random * max_length)
        # print('random length', random_length)
        m = max(data.train_set_right.tolist())  # max value of mapped_to
        complete_list = list(np.arange(0, m))
        non_commoners = list(set(complete_list) ^ set(data.train_set_right.tolist()))
        wrong_right_indices = [random.choice(non_commoners) for x in range(0, (random_length))]
        # print('len of random anti alignments: ', len(wrong_right_indices))

        # 4. Pack it all together

        # First part is the true alignment, then the random alignment and then the model-generated faults
        lefts = torch.cat(
            (data.train_set_left, data.train_set_left[:len(wrong_right_indices)], torch.tensor(picked_left)), 0)
        rights = torch.cat((data.train_set_right, torch.tensor(wrong_right_indices), torch.tensor(picked_right)), 0)

        lefts = lefts.type(torch.int64)
        rights = rights.type(torch.int64)
        return lefts, rights

    # In the case that no model output exists yet, only the random version happens
    else:
        lefts, rights = create_fake_data(data.train_set_right, data.train_set_left)
        return lefts, rights


def create_anti_alignment_v43(epoch, matches, data):
    # Check, if wrong matches exist to train the model with
    lefts, rights, wrong_set_left, wrong_set_right = filter_out_wrong_matches(matches, data)
    max_length = len(data.train_set_right)

    if len(wrong_set_left) > 0:

        # Create mixed version of anti-alignments

        # 1. Create the proportions
        prop_random = epoch / 1000
        proportion_of_model_random = math.floor(prop_random * 100) / 100.0
        proportion_of_model_faults = 1 - proportion_of_model_random

        # 2. Create random anti-alignments

        random_length = round(proportion_of_model_random * max_length)
        # print('random length', random_length)
        m = max(data.train_set_right.tolist())  # max value of mapped_to
        complete_list = list(np.arange(0, m))
        non_commoners = list(set(complete_list) ^ set(data.train_set_right.tolist()))
        wrong_right_indices = [random.choice(non_commoners) for x in range(0, (random_length))]
        # print('len of random anti alignments: ', len(wrong_right_indices))

        # 3. Choose wrong examples from model proportionately
        count_of_model_faults = math.floor(
            proportion_of_model_faults * max_length)  # calculation, how many examples are considered
        # print("count_of_model_faults", count_of_model_faults)
        # chosen_indices = [random.randint(0, len(wrong_set_left)) for x in range(count_of_model_faults)]
        amount_that_can_be_chosen = min(count_of_model_faults, len(wrong_set_right))
        chosen_indices = random.sample(range(len(wrong_set_right)), amount_that_can_be_chosen)
        # print("chosen_indices", chosen_indices)
        picked_left = list(np.array(wrong_set_left)[chosen_indices])
        picked_right = list(np.array(wrong_set_right)[chosen_indices])
        # print('picked_left: ', picked_left)
        # print('picked_left: ', picked_right)

        # 4. Pack it all together

        # First part is the true alignment, then the random alignment and then the model-generated faults
        lefts = torch.cat(
            (data.train_set_left, data.train_set_left[:len(wrong_right_indices)], torch.tensor(picked_left)), 0)
        rights = torch.cat((data.train_set_right, torch.tensor(wrong_right_indices), torch.tensor(picked_right)), 0)

        lefts = lefts.type(torch.int64)
        rights = rights.type(torch.int64)
        return lefts, rights

    # In the case that no model output exists yet, no additional loss happens
    else:
        return data.train_set_left, data.train_set_right


def create_anti_alignment_v44(epoch, matches, data):
    # Check, if wrong matches exist to train the model with
    lefts, rights, wrong_set_left, wrong_set_right = filter_out_wrong_matches(matches, data)
    max_length = len(data.train_set_right)

    if len(wrong_set_left) > 0:

        # Create mixed version of anti-alignments

        # 1. Create the proportions
        prop_random = epoch / 2000
        proportion_of_model_random = math.floor(prop_random * 100) / 100.0
        proportion_of_model_faults = 1 - proportion_of_model_random

        # 2. Create random anti-alignments

        random_length = round(proportion_of_model_random * max_length)
        # print('random length', random_length)
        m = max(data.train_set_right.tolist())  # max value of mapped_to
        complete_list = list(np.arange(0, m))
        non_commoners = list(set(complete_list) ^ set(data.train_set_right.tolist()))
        wrong_right_indices = [random.choice(non_commoners) for x in range(0, (random_length))]
        # print('len of random anti alignments: ', len(wrong_right_indices))

        # 3. Choose wrong examples from model proportionately
        count_of_model_faults = math.floor(
            proportion_of_model_faults * max_length)  # calculation, how many examples are considered
        # print("count_of_model_faults", count_of_model_faults)
        # chosen_indices = [random.randint(0, len(wrong_set_left)) for x in range(count_of_model_faults)]
        amount_that_can_be_chosen = min(count_of_model_faults, len(wrong_set_right))
        chosen_indices = random.sample(range(len(wrong_set_right)), amount_that_can_be_chosen)
        # print("chosen_indices", chosen_indices)
        picked_left = list(np.array(wrong_set_left)[chosen_indices])
        picked_right = list(np.array(wrong_set_right)[chosen_indices])
        # print('picked_left: ', picked_left)
        # print('picked_left: ', picked_right)

        # 4. Pack it all together

        # First part is the true alignment, then the random alignment and then the model-generated faults
        lefts = torch.cat(
            (data.train_set_left, data.train_set_left[:len(wrong_right_indices)], torch.tensor(picked_left)), 0)
        rights = torch.cat((data.train_set_right, torch.tensor(wrong_right_indices), torch.tensor(picked_right)), 0)

        lefts = lefts.type(torch.int64)
        rights = rights.type(torch.int64)
        return lefts, rights

    # In the case that no model output exists yet, no additional loss happens
    else:
        return data.train_set_left, data.train_set_right
