import torch

from rdflib import Graph, Literal, URIRef

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
import numpy as np

def parse_kg_file(rdflib_graph):

    resource_position_map, attribute_position_map = _get_resources_and_properties(rdflib_graph)
    all_edges, res_edges = _create_edges(rdflib_graph, resource_position_map, attribute_position_map)

    index_case_a, index_case_b = _change_datatype(res_edges, all_edges)

    resource_attr_map = _update_embeddings_with_literals(rdflib_graph, attribute_position_map, resource_position_map)

    x_case_a = _create_embeddings_case_a(resource_attr_map)
    x_case_b = _create_embeddings_case_b(resource_attr_map, attribute_position_map)

    return x_case_a, x_case_b, index_case_a, index_case_b, resource_position_map


def _get_resources_and_properties(rdflib_graph):
    print('_get_ressources_and_properties')
    resources = set()
    attributes = set()

    for subj, pred, obj in rdflib_graph:
        if (type(subj) is URIRef) and ('genid' not in subj.toPython()) and ('genid' not in obj.toPython()):
            resources.add(subj.toPython())
        if (type(obj) is URIRef) and ('genid' not in obj.toPython()) and ('genid' not in subj.toPython()):
            resources.add(obj.toPython())
        if type(obj) is Literal:
            attributes.add(str(obj.toPython()).lower().replace('_', ' '))

    resource_position_map = dict((element, index) for (index, element) in enumerate(resources))
    attribute_position_map = dict((element, index) for (index, element) in enumerate(attributes, len(resource_position_map)))
    print(f"Resource position map: {resource_position_map}")
    print(f"Attribute position map: {attribute_position_map}")
    return resource_position_map, attribute_position_map


def _create_edges(graph, resource_position_map, attribute_position_map):
    print('_create_edges')
    all_edges_list = []
    ressources_edges_list = []

    for subj, pred, obj in graph:
        if type(subj) is URIRef and type(obj) is URIRef and ('genid' not in subj.toPython()) and ('genid' not in obj.toPython()):
            #all_edges_list.append([resource_position_map[subj.toPython()], resource_position_map[obj.toPython()]])
            ressources_edges_list.append([resource_position_map[subj.toPython()], resource_position_map[obj.toPython()]])
    print(f"Number of all edges: {len(all_edges_list)}")
    print(f"Number of resource edges: {len(ressources_edges_list)}")
    return all_edges_list, ressources_edges_list


def _change_datatype(res_edges, all_edges):
    print('_change_datatype')
    res_edge_index_tensor = torch.tensor(res_edges, dtype=torch.long)
    res_edge_index = res_edge_index_tensor.t().contiguous()
    all_edge_index_tensor = torch.tensor(all_edges, dtype=torch.long)
    all_edge_index = all_edge_index_tensor.t().contiguous()
    return res_edge_index, all_edge_index


def _update_embeddings_with_literals(graph, attribute_position_map, resource_position_map):
    print('_update_embeddings_with_literals')
    attribute_position_map_rev = {y: x for x, y in attribute_position_map.items()}
    resource_attr_map = {y: x for x, y in resource_position_map.items()}

    for a, i in resource_position_map.items():
        triples_of_curr = graph.triples((URIRef(a), None, None))
        new_des = ' '
        for s, p, o in triples_of_curr:
            if type(o) == Literal:
                new_des = new_des + o + ' '
      ### TODO: ACTIVATE ROW
        new_des = new_des + resource_attr_map[i].split('/')[-1]
        resource_attr_map.update({i: new_des})
    return resource_attr_map


def _create_embeddings_case_a(resource_attr_map):
    print('_create_embeddings_for_prepared_nodes')
    attributes = resource_attr_map.values()
    embeddings = model.encode([i for i in list(attributes)])
    return torch.tensor(embeddings)

def _create_embeddings_case_b(resource_position_map, attribute_position_map):
    print("_create_embeddings_for_nodes_plus_literals")
    try:
        attribute_position_map['tooth']
        emb_res = torch.randn(len(resource_position_map), 384)
    except:
        res_embs = _prepare_simple_embs_for_nodes(resource_position_map)
        emb_res = _create_embs_from_simplified_nodes(res_embs)
    emb_lit = _attribute_emb(attribute_position_map)
    return torch.cat((emb_res, emb_lit), 0)

def _prepare_simple_embs_for_nodes(resource_position_map):
    print('_prepare_simple_embs_for_nodes')
    res_embs = []
    for key in resource_position_map.keys():
        res_embs.append(resource_position_map[key].split('/')[-1])
    return res_embs

def _create_embs_from_simplified_nodes(simplified):
    print('_create_embs_from_simplified_nodes')
    embeddings_res = [model.encode(i) for i in simplified]
    return torch.tensor(np.array(embeddings_res))

def _attribute_emb(attribute_position_map):
    print('_attribute_emb')
    attributes = attribute_position_map.keys() # Richtige Embeddings für alle Literale
    embeddings_lits = model.encode([i for i in list(attributes)])
    return torch.tensor(embeddings_lits)
