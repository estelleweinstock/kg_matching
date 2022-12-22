from .parse_graph import parse_kg_file
from .AlignmentFormat import parse_mapping_from_file
from rdflib import Graph, Literal, URIRef
import torch
from torch_geometric.data import Data


def create_pyg_data(kg_file_one, kg_file_two, alignment_file):

    # create the graphs
    graph_one = _create_graph(kg_file_one)
    graph_two = _create_graph(kg_file_two)

    x_case_a_one, x_case_b_one, case_a_edge_index_one, case_b_edge_index_one, resource_position_map_one = parse_kg_file(graph_one)
    x_case_a_two, x_case_b_two, case_a_edge_index_two, case_b_edge_index_two, resource_position_map_two = parse_kg_file(graph_two)

    alignment, onto1, onto2, extension = parse_mapping_from_file(alignment_file)

    left_indices, left_indices_list, right_indices, right_indices_list = _get_alignment(alignment,
                                                                                        resource_position_map_one,
                                                                                        resource_position_map_two)

    test_set_left, test_set_right, train_set_left, train_set_right, val_set_left, val_set_right = _train_test_val_split(
        left_indices_list, right_indices_list)

    return Data(x_one=x_case_a_one,
                edge_index_one=case_a_edge_index_one,
                res_map_one=resource_position_map_one,
                x_two=x_case_a_two,
                edge_index_two=case_a_edge_index_two,
                res_map_two=resource_position_map_two,
                left_indices=left_indices, right_indices=right_indices,
                train_set_left=train_set_left, test_set_left=test_set_left, val_set_left=val_set_left,
                train_set_right=train_set_right, test_set_right=test_set_right, val_set_right=val_set_right
                ), Data(x_one=x_case_b_one,
                 edge_index_one=case_b_edge_index_one,
                 res_map_one=resource_position_map_one,
                 x_two=x_case_b_two,
                 edge_index_two=case_b_edge_index_two,
                 res_map_two=resource_position_map_two,
                 left_indices=left_indices, right_indices=right_indices,
                 train_set_left=train_set_left, test_set_left=test_set_left, val_set_left=val_set_left,
                 train_set_right=train_set_right, test_set_right=test_set_right, val_set_right=val_set_right
                )

def _create_graph(file):
    graph = Graph()
    return graph.parse(file)


def _get_alignment(alignment, resource_position_map_one, resource_position_map_two):
    left_indices_list = []
    right_indices_list = []
    for src, tgt, rel, conf in alignment:
        try:
            left_indices_list.append(resource_position_map_one[src])
            right_indices_list.append(resource_position_map_two[tgt])
        except:
            left_indices_list.append(0)
            right_indices_list.append(0)
    left_indices = torch.tensor(left_indices_list)
    right_indices = torch.tensor(right_indices_list)
    return left_indices, left_indices_list, right_indices, right_indices_list


def _train_test_val_split(left_indices_list, right_indices_list):
    train_indices_left = left_indices_list[: int(len(left_indices_list) * 0.7)]
    train_indices_right = right_indices_list[: int(len(right_indices_list) * 0.7)]
    test_indices_left = left_indices_list[int(len(left_indices_list) * 0.7):int(len(left_indices_list) * 0.9)]
    test_indices_right = right_indices_list[int(len(right_indices_list) * 0.7):int(len(right_indices_list) * 0.9)]
    val_indices_left = left_indices_list[int(len(left_indices_list) * 0.9):]
    val_indices_right = right_indices_list[int(len(right_indices_list) * 0.9):]
    train_set_left = torch.tensor(train_indices_left)
    test_set_left = torch.tensor(test_indices_left)
    val_set_left = torch.tensor(val_indices_left)
    train_set_right = torch.tensor(train_indices_right)
    test_set_right = torch.tensor(test_indices_right)
    val_set_right = torch.tensor(val_indices_right)
    return test_set_left, test_set_right, train_set_left, train_set_right, val_set_left, val_set_right
