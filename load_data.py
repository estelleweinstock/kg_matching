import os.path as osp
import ssl
import numpy as np
import urllib.request
from rdflib import Graph, Literal, URIRef, RDF
from AlignmentFormat import parse_mapping_from_file
import torch
from torch_geometric.data import Data
import xml.etree.ElementTree as etree
import random
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


# Main
def create_pyg_data(kg_file_one, kg_file_two, alignment_file):
    pred_dict, graph_one, graph_two = get_pred_dict(kg_file_one, kg_file_two)
    x_prepared_embs_one, x_embs_and_literals_one, res_edge_index_one, res_edge_type_one, all_edge_index_one, all_edge_type_one, edge_attr_embs_res_tensor_one, edge_attr_embs_all_tensor_one, resource_position_map_one = parse_kg_file(
        graph_one, pred_dict)
    x_prepared_embs_two, x_embs_and_literals_two, res_edge_index_two, res_edge_type_two, all_edge_index_two, all_edge_type_two, edge_attr_embs_res_tensor_two, edge_attr_embs_all_tensor_two, resource_position_map_two = parse_kg_file(
        graph_two, pred_dict)

    alignment, onto1, onto2, extension = parse_mapping_from_file(alignment_file)

    left_indices, left_indices_list, right_indices, right_indices_list = _get_alignment(alignment,
                                                                                        resource_position_map_one,
                                                                                        resource_position_map_two)

    test_set_left, test_set_right, train_set_left, train_set_right, val_set_left, val_set_right = _train_test_val_split(
        left_indices_list, right_indices_list)

    return Data(x_one=x_prepared_embs_one, edge_index_one=res_edge_index_one,
                edge_type_one=res_edge_type_one, edge_attr_emb_one=edge_attr_embs_res_tensor_one,
                map_one=edge_attr_embs_res_tensor_one, res_map_one=resource_position_map_one,
                x_two=x_prepared_embs_two, edge_index_two=res_edge_index_two,
                edge_type_two=res_edge_type_two, edge_attr_emb_two=edge_attr_embs_res_tensor_two,
                map_two=edge_attr_embs_res_tensor_two, res_map_two=resource_position_map_two,
                left_indices=left_indices, right_indices=right_indices, train_set_left=train_set_left,
                test_set_left=test_set_left, val_set_left=val_set_left,
                train_set_right=train_set_right, test_set_right=test_set_right, val_set_right=val_set_right
                ), Data(x_one=x_embs_and_literals_one, edge_index_one=all_edge_index_one,
                        edge_type_one=all_edge_type_one, edge_attr_emb_one=edge_attr_embs_all_tensor_one,
                        map_one=edge_attr_embs_res_tensor_one, res_map_one=resource_position_map_one,
                        x_two=x_embs_and_literals_two, edge_index_two=all_edge_index_two,
                        edge_type_two=all_edge_type_two, edge_attr_emb_two=edge_attr_embs_all_tensor_two,
                        map_two=edge_attr_embs_res_tensor_two, res_map_two=resource_position_map_two,
                        left_indices=left_indices, right_indices=right_indices, train_set_left=train_set_left,
                        test_set_left=test_set_left, val_set_left=val_set_left,
                        train_set_right=train_set_right, test_set_right=test_set_right, val_set_right=val_set_right
                        )


def get_pred_dict(file_one, file_two):
    graph_one = create_graph(file_one)
    graph_two = create_graph(file_two)
    return create_dict(graph_one, graph_two), graph_one, graph_two


def create_graph(file):
    graph = Graph()
    return graph.parse(file)


def create_dict(graph_one, graph_two):
    pred_one = get_preds(graph_one)
    pred_two = get_preds(graph_two)
    print(f"Count of first graph ind. predicates: {len(pred_one)}")
    print(f"Count of sec. graph ind. predicates: {len(pred_two)}")
    return dict((element, index) for (index, element) in enumerate(pred_one.union(pred_two)))


def get_preds(graph):
    set_of_diff_preds = set()
    for subj, pred, obj in graph:
        set_of_diff_preds.add(pred)
    return set_of_diff_preds


def _get_alignment(alignment, resource_position_map_one, resource_position_map_two):
    left_indices_list = []
    right_indices_list = []
    for src, tgt, rel, conf in alignment:
        left_indices_list.append(resource_position_map_one[src])
        right_indices_list.append(resource_position_map_two[tgt])
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


def parse_kg_file(rdflib_graph, pred_dict):
    resource_position_map, attribute_position_map = _get_ressources_and_properties(rdflib_graph)
    all_edges, res_edges, all_edge_type, res_edge_type = _create_edges(rdflib_graph,
                                                                       resource_position_map,
                                                                       attribute_position_map,
                                                                       pred_dict)
    res_edge_index, res_edge_type, all_edge_index, all_edge_type = _change_datatype(res_edges, res_edge_type, all_edges,
                                                                                    all_edge_type)

    resource_attr_map = _update_embeddings_with_literals(rdflib_graph, resource_position_map)
    x_prepared_embs = _create_embeddings_for_prepared_nodes(resource_attr_map)

    x_embs_and_literals = _create_embeddings_for_nodes_plus_literals(resource_attr_map, attribute_position_map)

    edge_attr_embs_res_tensor, edge_attr_embs_all_tensor = _create_embeddings_for_edges(pred_dict, res_edge_type,
                                                                                        all_edge_type)

    return x_prepared_embs, x_embs_and_literals, res_edge_index, res_edge_type, all_edge_index, all_edge_type, edge_attr_embs_res_tensor, edge_attr_embs_all_tensor, resource_position_map


def _get_ressources_and_properties(rdflib_graph):
    print('_get_ressources_and_properties')
    resources = set()
    attributes = set()

    for subj, pred, obj in rdflib_graph:
        if type(subj) is URIRef:
            resources.add(subj.toPython())
        if type(obj) is URIRef:
            resources.add(obj.toPython())
        if type(obj) is Literal:
            attributes.add(str(obj.toPython()).lower().replace('_', ' '))

    resource_position_map = dict((element, index) for (index, element) in enumerate(resources))
    attribute_position_map = dict(
        (element, index) for (index, element) in enumerate(attributes, len(resource_position_map)))
    return resource_position_map, attribute_position_map


def _create_edges(graph, resource_position_map, attribute_position_map, pred_dict):
    print('_create_edges')
    all_edges_list = []
    ressources_edges_list = []
    all_edge_type_list = []
    ressources_edge_type_list = []

    for subj, pred, obj in graph:
        if type(subj) is URIRef and type(obj) is URIRef:
            all_edges_list.append([resource_position_map[subj.toPython()], resource_position_map[obj.toPython()]])
            ressources_edges_list.append(
                [resource_position_map[subj.toPython()], resource_position_map[obj.toPython()]])
            all_edge_type_list.append(pred_dict[pred])
            ressources_edge_type_list.append(pred_dict[pred])
        elif type(subj) is URIRef and type(obj) is Literal:
            all_edges_list.append([resource_position_map[subj.toPython()],
                                   attribute_position_map[str(obj.toPython()).lower().replace('_', ' ')]])
            all_edge_type_list.append(pred_dict[pred])
    return all_edges_list, ressources_edges_list, all_edge_type_list, ressources_edge_type_list


def _change_datatype(res_edges, res_edge_type, all_edges, all_edge_type):
    print('_change_datatype')
    res_edge_index_tensor = torch.tensor(res_edges, dtype=torch.long)
    res_edge_index = res_edge_index_tensor.t().contiguous()
    res_edge_type = torch.tensor(res_edge_type, dtype=torch.long)

    all_edge_index_tensor = torch.tensor(all_edges, dtype=torch.long)
    all_edge_index = all_edge_index_tensor.t().contiguous()
    all_edge_type = torch.tensor(all_edge_type, dtype=torch.long)

    return res_edge_index, res_edge_type, all_edge_index, all_edge_type


def _update_embeddings_with_literals(graph, resource_position_map):
    print('_update_embeddings_with_literals')
    resource_attr_map = {y: x for x, y in resource_position_map.items()}

    for a, i in resource_position_map.items():
        triples_of_curr = graph.triples((URIRef(a), None, None))
        new_des = ' '
        for s, p, o in triples_of_curr:
            if type(o) == Literal:
                new_des = new_des + o + ' '
        # wir können ja sowas machen wie z.B. : nimm die letzten paar Buchstaben ohne Sonderzeichen wie z.B. "#, /" und checke ob es ein echtes Wort ist
        ### TODO: ACTIVATE ROW
        new_des = new_des + resource_attr_map[i].split('/')[
            -1]  # Das bringt bei der Maus nichts - würde es nur schlechter machen, Abfrage überlegen!!!
        resource_attr_map.update({i: new_des})
    return resource_attr_map


def _create_embeddings_for_prepared_nodes(resource_attr_map):
    print('_create_embeddings_for_prepared_nodes')
    attributes = resource_attr_map.values()
    embeddings = model.encode([i for i in list(attributes)])
    return torch.tensor(embeddings)


def _create_embeddings_for_nodes_plus_literals(resource_position_map, attribute_position_map):
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
    attributes = attribute_position_map.keys()  # Richtige Embeddings für alle Literale
    embeddings_lits = model.encode([i for i in list(attributes)])
    return torch.tensor(embeddings_lits)


def _create_embeddings_for_edges(pred_dict, res_edge_type, all_edge_type):
    ### TODO: ACTIVATE ROW

    print('_create_embeddings_for_edges')
    properties_embedding_map = pred_dict.copy()
    for key in pred_dict.keys():
        properties_embedding_map.update(
            {key: model.encode(key.split('/')[-1])})  # das ist bei jedem anders, ist jetzt für starwars klein
        # properties_embedding_map.update({key: model.encode(key.split('#')[-1])}) # für maus benutze ich jetzt einfach das, wird schon nicht schlechter als random sein

    edge_attr_embs_all = []
    edge_attr_embs_res = []

    positions_rev = {y: x for x, y in pred_dict.items()}

    for t in res_edge_type:
        edge_attr_embs_res.append(
            properties_embedding_map[positions_rev[t.item()]])  # Herausfinden, warum es hier anders ist!!!!
    for t in all_edge_type:
        edge_attr_embs_all.append(properties_embedding_map[positions_rev[t.item()]])

    edge_attr_embs_all = np.array(edge_attr_embs_all)
    edge_attr_embs_res = np.array(edge_attr_embs_res)

    edge_attr_embs_res_tensor = torch.tensor(edge_attr_embs_res, dtype=torch.float)
    edge_attr_embs_all_tensor = torch.tensor(edge_attr_embs_all, dtype=torch.float)
    return edge_attr_embs_res_tensor, edge_attr_embs_all_tensor


# Helper funcs


class AlignmentHandler(object):
    def __init__(self):
        self.base = "{http://knowledgeweb.semanticweb.org/heterogeneity/alignment}"
        self.rdf = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
        self.text = ""
        self.alignment = []
        self.one_cell = ["", "", "", "", {}]
        self.extension = {}
        self.onto1 = ""
        self.onto2 = ""
        self.onto_temp = ["", ""]
        self.in_cell = False
        self.used_tags = set(
            [
                self.base + name
                for name in [
                "entity1",
                "entity2",
                "relation",
                "measure",
                "Cell",
                "map",
                "Alignment",
                "xml",
                "level",
                "type",
                "onto1",
                "onto2",
                "Ontology",
                "location",
                "formalism",
                "Formalism",
            ]
            ]
        )
        self.used_tags.add(self.rdf + "RDF")

    def start(self, name, attrs):
        if name == self.base + "entity1":
            self.one_cell[0] = attrs[self.rdf + "resource"]  # .encode('utf-8')
        elif name == self.base + "entity2":
            self.one_cell[1] = attrs[self.rdf + "resource"]  # .encode('utf-8')
        elif name == self.base + "Ontology":
            self.onto_temp[0] = attrs[self.rdf + "about"]  # .encode('utf-8')
        elif name == self.base + "Cell":
            self.in_cell = True
        self.text = ""

    def end(self, name):
        if name == self.base + "relation":
            self.one_cell[2] = self.text.strip()
        elif name == self.base + "measure":
            self.one_cell[3] = self.text.strip()
        elif name == self.base + "Cell":
            self.alignment.append(self.one_cell)
            self.one_cell = ["", "", "", "", {}]
            self.in_cell = False
        elif name == self.base + "location":
            self.onto_temp[1] = self.text.strip()
        elif name == self.base + "onto1":
            if self.onto_temp[0] == "" and self.onto_temp[1] == "":
                self.onto_temp[0] = self.text.strip()
            self.onto1 = list(self.onto_temp)
        elif name == self.base + "onto2":
            if self.onto_temp[0] == "" and self.onto_temp[1] == "":
                self.onto_temp[0] = self.text.strip()
            self.onto2 = list(self.onto_temp)
        elif name not in self.used_tags:
            key = name.replace("{", "", 1).replace("}", "", 1)  # name[name.index("}") + 1 :]
            if self.in_cell:
                self.one_cell[4][key] = self.text
            else:
                self.extension[key] = self.text

    def data(self, chars):
        self.text += chars

    def close(self):
        pass


def remove_cell_extensions(alignment):
    for c in alignment:
        c.pop()


def parse_mapping_from_file(source, parse_cell_extensions=False):
    """
    Parses a alignment from a filename or file object.
    :param source: is a filename or file object containing a alignment in alignment format
    :param parse_cell_extensions: if true, also parses the cell extensions
    :return: (alignment: list of (source, target, relation, confidence, extensions - which are parsed only when parse_cell_extensions is True),
    onto1 as (id, url, formalismName, formalismURI),
    onto2 similar to onto1,
    extension (iterable of key, values)
    )
    """
    handler = AlignmentHandler()
    etree.parse(source, etree.XMLParser(target=handler))
    if parse_cell_extensions == False:
        remove_cell_extensions(handler.alignment)
    return handler.alignment, handler.onto1, handler.onto2, handler.extension


def download_url(url, path):
    if osp.exists(path):
        return path

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def create_fake_data(tens1, tens2):
    right_inds_list, left_inds_list = tens1.tolist(), tens2.tolist()
    double_left_list = left_inds_list + left_inds_list

    m = max(right_inds_list)

    wrong_right_indices = [(lambda x: random.choice([x for x in range(m) if x not in right_inds_list]))(x)
                           for x in range(len(left_inds_list))]
    double_right_list = right_inds_list + wrong_right_indices
    double_right_tensor = torch.tensor(double_right_list, dtype=torch.long)
    double_left_tensor = torch.tensor(double_left_list)
    return double_left_tensor, double_right_tensor
