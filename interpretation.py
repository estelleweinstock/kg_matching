from collections import Counter


def count_occurences(matches_train, matches_test, top_k):
    train = Counter(x for xs in matches_train for x in set(xs)).most_common(top_k)
    test = Counter(x for xs in matches_test for x in set(xs)).most_common(top_k)
    return train, test


def check_real_val_of_both(data, resource_position_map_one, resource_position_map_two, ind):
    """Um an die richtigen URIs zu kommen:

   1) Such nach dem Index in den left und right indices
   2) Gib den Wert von dem Tensor in resource_position_map_one > left und resource_position_map_two > right an. """
    left_item = resource_position_map_one[data.left_indices[ind].item()]
    right_item = resource_position_map_two[data.right_indices[ind].item()]
    return left_item, right_item


def check_single_left_vals(data, resource_position_map_one, ind):
    return resource_position_map_one[data.left_indices[ind].item()]


def check_single_right_vals(data, resource_position_map_two, ind):
    return resource_position_map_two[data.right_indices[ind].item()]


def save_output(matches_test, source_data, resource_position_map_one, resource_position_map_two):
    resource_position_map_one_rev = {y: x for x, y in resource_position_map_one.items()}
    resource_position_map_two_rev = {y: x for x, y in resource_position_map_two.items()}
    uris_left = []
    for ind in matches_test:
        uris_left.append(check_single_left_vals(source_data, resource_position_map_one_rev, ind[0]))
    uris_right = []
    for ind in matches_test:
        uris_right.append(check_single_right_vals(source_data, resource_position_map_two_rev, ind[1]))
    conf_scores = [el[2] for el in matches_test]
    relation = ['='] * len(conf_scores)
    output_test = list(zip(uris_left, uris_right, relation, conf_scores))
    return output_test

