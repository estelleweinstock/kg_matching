import logging
import os
import os.path as osp
import ssl
import urllib.request
from rdflib import Graph, Literal, URIRef, RDF
from AlignmentFormat import parse_mapping_from_file
import torch
from torch_geometric.data import Data
import xml.etree.ElementTree as etree
import random
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

list_of_important_properties = [
    'http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym',
    'http://www.w3.org/2000/01/rdf-schema#label',
    'http://www.w3.org/2002/07/owl#someValuesFrom',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
    'http://www.w3.org/2000/01/rdf-schema#subClassOf',
    'http://www.w3.org/2002/07/owl#onProperty'
]


# Main
def create_pyg_data(kg_file_one, kg_file_two, alignment_file, query_labels):
    x_one, edge_index_one, resource_position_map_one, edge_type_one, resource_position_map_clone_one, edge_attr_one = parse_kg_file(
        kg_file_one, query_labels)
    x_two, edge_index_two, resource_position_map_two, edge_type_two, resource_position_map_clone_two, edge_attr_two = parse_kg_file(
        kg_file_two, query_labels)

    alignment, onto1, onto2, extension = parse_mapping_from_file(alignment_file)

    left_indices, left_indices_list, right_indices, right_indices_list = _get_alignment(alignment,
                                                                                        resource_position_map_one,
                                                                                        resource_position_map_two)

    test_set_left, test_set_right, train_set_left, train_set_right, val_set_left, val_set_right = _train_test_val_split(
        left_indices_list, right_indices_list)

    return Data(x_one=x_one, edge_index_one=edge_index_one, edge_type_one=edge_type_one, edge_attr_one=edge_attr_one,
                x_two=x_two, edge_index_two=edge_index_two, edge_type_two=edge_type_two, edge_attr_two=edge_attr_two,
                left_indices=left_indices, right_indices=right_indices,
                train_set_left=train_set_left, test_set_left=test_set_left, val_set_left=val_set_left,
                train_set_right=train_set_right, test_set_right=test_set_right,
                val_set_right=val_set_right), resource_position_map_one, resource_position_map_two,\
           resource_position_map_clone_one, resource_position_map_clone_two


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


def parse_kg_file(kg_file, query_labels):
    rdflib_graph = Graph()
    rdflib_graph.parse(kg_file)
    properties, resources = _get_ressources_and_properties(rdflib_graph)
    labels_dict = _map_classes_to_labels(query_labels, rdflib_graph)
    resource_position_map = dict((element, index) for (index, element) in enumerate(resources))
    resource_position_map_clone = resource_position_map.copy()
    resource_position_map_clone.update(labels_dict)

    properties_position_map = dict.fromkeys(el for el in list_of_important_properties)
    properties_position_map.update((k, i) for i, k in enumerate(properties_position_map))

    edge_emb_tensor, edge_index, edge_type = _get_edge_information(properties_position_map, rdflib_graph,
                                                                   resource_position_map)

    # Sentences are encoded by calling model.encode()
    x = _create_embedddings(resource_position_map_clone)

    return x, edge_index, resource_position_map, edge_type, resource_position_map_clone, edge_emb_tensor


def _get_edge_information(properties_position_map, rdflib_graph, resource_position_map):
    edge_index_list = []
    edge_type_list = []
    edge_emb_list = []
    for subj, pred, obj in rdflib_graph:
        if type(subj) is URIRef and type(obj) is URIRef:
            edge_index_list.append([resource_position_map[subj.toPython()], resource_position_map[obj.toPython()]])
            # wir müssen hier echt aufpassen, dass alle kanten gemappt werden!!!!
            # muss also eingerückt sein, weil genau dann haben wir auch einen link - in dem aufbau müssen wir das so machen <3
            if pred.toPython() in list_of_important_properties:
                edge_type_list.append(properties_position_map[pred.toPython()])
            else:
                # sonst nimm einfach das gefüllteste, damit es auch filled ist
                edge_type_list.append(properties_position_map['http://www.w3.org/2000/01/rdf-schema#subClassOf'])
            edge_emb_list.append(model.encode(pred.toPython().split('#')[1]))
    edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long)
    edge_type = torch.tensor(edge_type_list, dtype=torch.long)
    edge_emb_tensor = torch.tensor(edge_emb_list, dtype=torch.long)
    edge_index = edge_index_tensor.t().contiguous()
    return edge_emb_tensor, edge_index, edge_type


def _map_classes_to_labels(query_labels, rdflib_graph):
    keys = []
    vals = []
    for r in rdflib_graph.query(query_labels):
        keys.append(r["p"].toPython())
        vals.append(r["label"].toPython())
    labels_dict = dict(zip(keys, vals))
    return labels_dict


def _get_ressources_and_properties(rdflib_graph):
    resources = set()
    properties = set()
    for subj, pred, obj in rdflib_graph:
        if type(subj) is URIRef:
            resources.add(subj.toPython())
        if type(pred) is URIRef:
            properties.add(pred.toPython())
        if type(obj) is URIRef:
            resources.add(obj.toPython())
    return properties, resources


def _create_embedddings(resource_position_map_clone):
    real_labels = resource_position_map_clone.values()
    embeddings = model.encode([str(i).lower().replace('_', ' ') for i in list(real_labels)])
    x = torch.tensor(embeddings)
    return x


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
