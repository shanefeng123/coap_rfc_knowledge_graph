from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch
from prepare_pretrain_data import prepare_pretrain_data
import re
from tqdm import tqdm
import pickle
import networkx as nx
import numpy as np
import random
from z3 import *
import pdfplumber
import nltk

random.seed(4)

PROPERTIES = ["value", "error", "idempotent"]


def is_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def extract_properties(behaviour):
    behaviour = behaviour.strip()
    if len(behaviour.split("=")) < 2:
        return None
    behaviour_property = behaviour.split("=")[0].strip()
    sentiment = behaviour.split("=")[1].strip()
    # Searching for property "value" in the behaviour
    if re.search(r"\b" + "set" + r"\b", behaviour_property) and re.search(r"\b" + "to" + r"\b", behaviour_property):
        property_name = "value"
        value = behaviour_property[re.search(r"\b" + "to" + r"\b", behaviour_property).end():].strip()
        if sentiment == "True":
            operator = "="
        else:
            operator = "!="
        return property_name, operator, value
    elif re.search(r"\b" + "unknown" + r"\b", behaviour_property):
        property_name = "value"
        value = "unknown"
        if sentiment == "True":
            operator = "="
        else:
            operator = "!="
        return property_name, operator, value
    # elif re.search(r"\b\d+\b", behaviour_property) and (
    #         re.search(r"\b" + "and" + r"\b", behaviour_property) or re.search(r"\b" + "to" + r"\b",
    #                                                                           behaviour_property)):
    #     property_name = "value"
    #     values = re.findall(r"\b\d+\b", behaviour_property)
    #     if len(values) == 1:
    #         return property_name, "=", values[0]
    #     else:
    #         return [(property_name, ">", values[0]), (property_name, "<", values[1])]
    elif re.search(r"\b" + "Empty" + r"\b", behaviour_property) or re.search(r"\b" + "empty" + r"\b",
                                                                             behaviour_property):
        property_name = "value"
        value = "empty"
        if sentiment == "True":
            operator = "="
        else:
            operator = "!="
        return property_name, operator, value
    elif re.search(r"\b" + "decreased" + r"\b", behaviour_property) and re.search(r"\b" + "below" + r"\b",
                                                                                  behaviour_property):
        property_name = "value"
        value = behaviour_property[re.search(r"\b" + "below" + r"\b", behaviour_property).end():].strip()
        if sentiment == "True":
            operator = "<"
        else:
            operator = ">="
        return property_name, operator, value
    elif re.search(r"\b" + "class elective" + r"\b", behaviour_property):
        property_name = "unrecognized option"
        value = "elective"
        if sentiment == "True":
            operator = "="
        else:
            operator = "!="
        return property_name, operator, value
    elif re.search(r"\b" + "class critical" + r"\b", behaviour_property):
        property_name = "unrecognized option"
        value = "critical"
        if sentiment == "True":
            operator = "="
        else:
            operator = "!="
        return property_name, operator, value
    # elif (re.search(r"\b" + "return" + r"\b", behaviour_property) or re.search(r"\b" + "returned" + r"\b",
    #                                                                            behaviour_property)) and re.search(
    #     r"\b" + "response" + r"\b",
    #     behaviour_property):
    #     property_name = "value"
    #     value = ".".join(re.findall(r"\b\d+\b", behaviour_property))
    #     if sentiment == "True":
    #         operator = "="
    #     else:
    #         operator = "!="
    #     return property_name, operator, value
    elif re.search(r"\b" + "Response Code" + r"\b", behaviour_property):
        property_name = "value"
        value = ".".join(re.findall(r"\b\d+\b", behaviour_property))
        if value == "":
            value = "error"
        if sentiment == "True":
            operator = "="
        else:
            operator = "!="
        return property_name, operator, value
    elif re.search(r"\b" + "error code" + r"\b", behaviour_property):
        property_name = "value"
        value = ".".join(re.findall(r"\b\d+\b", behaviour_property))
        if value == "":
            value = "error"
        if sentiment == "True":
            operator = "="
        else:
            operator = "!="
        return property_name, operator, value
    elif re.search(r"\b" + "default value" + r"\b", behaviour_property):
        property_name = "value"
        value = "default"
        if sentiment == "True":
            operator = "="
        else:
            operator = "!="
        return property_name, operator, value
    elif re.search(r"\b" + "value" + r"\b", behaviour_property) and re.search(r"\b" + "be" + r"\b", behaviour_property):
        property_name = "value"
        value = behaviour_property[re.search(r"\b" + "be" + r"\b", behaviour_property).end():].strip()
        if sentiment == "True":
            operator = "="
        else:
            operator = "!="
        return property_name, operator, value
    # Searching for property "error" in the behaviour
    elif re.search(r"\b" + "ignored" + r"\b", behaviour_property):
        property_name = "error"
        value = "ignore"
        if sentiment == "True":
            operator = "="
        else:
            operator = "!="
        return property_name, operator, value
    elif re.search(r"\b" + "ignore" + r"\b", behaviour_property):
        property_name = "error"
        value = "ignore"
        if sentiment == "True":
            operator = "="
        else:
            operator = "!="
        return property_name, operator, value
    elif re.search(r"\b" + "message format error" + r"\b", behaviour_property):
        property_name = "error"
        value = "message format error"
        if sentiment == "True":
            operator = "="
        else:
            operator = "!="
        return property_name, operator, value
    elif re.search(r"\b" + "reject" + r"\b", behaviour_property) and re.search(r"\b" + "message" + r"\b",
                                                                               behaviour_property):
        property_name = "error"
        value = "reject message"
        if sentiment == "True":
            operator = "="
        else:
            operator = "!="
        return property_name, operator, value
    elif re.search(r"\b" + "rejected" + r"\b", behaviour_property):
        property_name = "error"
        value = "reject"
        if sentiment == "True":
            operator = "="
        else:
            operator = "!="
        return property_name, operator, value
    elif re.search(r"\b" + "Rejecting" + r"\b", behaviour_property) and re.search(r"\b" + "message" + r"\b",
                                                                                  behaviour_property):
        property_name = "error"
        value = "reject message"
        if sentiment == "True":
            operator = "="
        else:
            operator = "!="
        return property_name, operator, value
    elif re.search(r"\b" + "ICMP errors" + r"\b", behaviour_property):
        property_name = "error"
        value = "ICMP errors"
        if sentiment == "True":
            operator = "="
        else:
            operator = "!="
        return property_name, operator, value
    # Searching for property "idempotent" in the behaviour
    elif re.search(r"\b" + "idempotent" + r"\b", behaviour_property):
        property_name = "idempotent"
        value = sentiment
        return property_name, "=", value
    else:
        return behaviour_property, "=", sentiment


class Entity:
    def __init__(self, name):
        self.name = name


class AtomicRule:
    def __init__(self, var, op, value, rule_sentence_num, entity):
        self.var = var
        self.op = op
        self.value = value
        self.rule_sentence_num = rule_sentence_num
        self.entity = entity


class Rule:
    def __init__(self, atomic_rules, logical_connective, requirement_level, condition, rule_sentence):
        self.atomic_rules = atomic_rules
        self.logical_connective = logical_connective
        self.requirement_level = requirement_level
        self.condition = condition
        self.rule_sentence = rule_sentence


MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL", "SHOULD", "RECOMMENDED", "MAY", "OPTIONAL"]
STRONG_MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL"]
CONDITIONAL_KEYWORDS = ["if", "when", "unless", "instead", "except", "as", "thus", "therefore", "in case"]
LABELS = ["B-entity", "I-entity", "Other", "PAD"]

amqp_spec = []
with pdfplumber.open("../data/amqp_specification.pdf") as pdf:
    pages = pdf.pages[16:119]
    for page in pages:
        text = page.extract_text(layout=False, x_tolerance=1)
        text = text.split("\n")
        for line in text:
            line = line.strip()

            alpha = any(c.isalpha() for c in line)
            if not alpha:
                line = ""

            if line.startswith("amqp-core"):
                line = ""

            if line.startswith("PART"):
                line = ""

            if line.startswith("0x"):
                line = ""

            if line.startswith("<type"):
                line = ""

            if line.startswith("label="):
                line = ""

            if line.startswith("<encoding"):
                line = ""

            if line.startswith("<descriptor"):
                line = ""

            if line.startswith("Standards Track Work Product"):
                line = ""

            if line == "":
                continue

            separate = line.split(" ", 1)
            if separate[0].isdigit():
                amqp_spec.append(separate[1])
            else:
                amqp_spec.append(line)

amqp_spec = "\n".join(amqp_spec)
amqp_spec_sentences = nltk.sent_tokenize(amqp_spec, "english")
for i in range(len(amqp_spec_sentences)):
    amqp_spec_sentences[i] = amqp_spec_sentences[i].strip()
    amqp_spec_sentences[i] = amqp_spec_sentences[i].replace("\n", " ")
    amqp_spec_sentences[i] = re.sub(' +', ' ', amqp_spec_sentences[i])

    alpha = any(c.isalpha() for c in amqp_spec_sentences[i])
    if not alpha:
        amqp_spec_sentences[i] = ""

    if "Figure" in amqp_spec_sentences[i]:
        amqp_spec_sentences[i] = ""

    if amqp_spec_sentences[i].startswith("</type>"):
        amqp_spec_sentences[i] = ""

    if amqp_spec_sentences[i].startswith("<field"):
        amqp_spec_sentences[i] = ""

    if "-->" in amqp_spec_sentences[i]:
        amqp_spec_sentences[i] = ""

    if "--+" in amqp_spec_sentences[i]:
        amqp_spec_sentences[i] = ""

    if "||" in amqp_spec_sentences[i]:
        amqp_spec_sentences[i] = ""

amqp_spec_sentences = [sentence for sentence in amqp_spec_sentences if sentence != ""]
amqp_rule_sentences = []
for sentence in amqp_spec_sentences:
    for keyword in MODAL_KEYWORDS:
        if keyword in sentence:
            amqp_rule_sentences.append(sentence)
            break

entities = pickle.load(open("../data/all_AMQP_entities", "rb"))
AMQP_context_condition_split = pickle.load(open('../data/amqp_contexts_condition_split', 'rb'))
AMQP_split_properties = pickle.load(open('../data/all_AMQP_properties', 'rb'))

# Normalise entities
for i in range(len(entities)):
    entities[i] = entities[i].lower()
entities = list(set(entities))
# Compute entity vectors with PhraseBERT
phraseBERT = SentenceTransformer("whaleloops/phrase-bert")
compute_cosine_similarity = torch.nn.CosineSimilarity(dim=0)
entity_vectors = torch.tensor(phraseBERT.encode(entities))

for i in range(len(entity_vectors)):
    entity_vector_1 = entity_vectors[i]
    for j in range(i, len(entity_vectors)):
        entity_vector_2 = entity_vectors[j]
        similarity = compute_cosine_similarity(entity_vector_1, entity_vector_2)
        if similarity > 0.9:
            entities[j] = entities[i]
entities = list(set(entities))

entity_vectors = torch.tensor(phraseBERT.encode(entities))

graph = nx.Graph()
for entity in entities:
    entity_obj = Entity(entity)
    graph.add_node(entity_obj.name, data=entity_obj, label=entity_obj.name)

# Extract atomic rules
all_transformed_rules_list = []
for i in range(len(AMQP_split_properties)):
    atomic_rules_list = AMQP_split_properties[i].split(";")[:-1]
    transformed_atomic_rules_list = []
    for atomic_rule in atomic_rules_list:
        atomic_rule = atomic_rule.strip()
        if len(atomic_rule.split("@")) < 2:
            continue
        # Normalise the entity
        entity = atomic_rule.split("@")[0].strip().lower()
        entity_vector = \
            torch.tensor(phraseBERT.encode(entity))
        top_similarity = 0
        top_entity = ""
        for j in range(len(entity_vectors)):
            similarity = compute_cosine_similarity(entity_vector, entity_vectors[j])
            if similarity > top_similarity:
                top_similarity = similarity
                top_entity = entities[j]
        entity = top_entity
        # Extract the properties
        var_sentiment = atomic_rule.split("@")[1].strip()
        var_op_val = extract_properties(var_sentiment)
        if var_op_val is None:
            continue
        if type(var_op_val) == tuple:
            var = var_op_val[0].strip()
            op = var_op_val[1].strip()
            value = var_op_val[2].strip()
            transformed_atomic_rules_list.append(
                AtomicRule(var, op, value, AMQP_context_condition_split[i][-1], entity))
        else:
            for j in range(len(var_op_val)):
                var_op_val_pair = var_op_val[j]
                var = var_op_val_pair[0].strip()
                op = var_op_val_pair[1].strip()
                value = var_op_val_pair[2].strip()
                transformed_atomic_rules_list.append(
                    AtomicRule(var, op, value, AMQP_context_condition_split[i][-1], entity))
    all_transformed_rules_list.append(transformed_atomic_rules_list)

# Construct rule objects and add them to the graph
entity_rule_number = {}
condition_rules = []
for i in range(len(all_transformed_rules_list)):
    entity_atomic_rules = {}
    condition = AMQP_context_condition_split[i][2]
    rule_sentence = amqp_rule_sentences[AMQP_context_condition_split[i][-1]]

    requirement_level = "WEAK"
    for keyword in STRONG_MODAL_KEYWORDS:
        if keyword in rule_sentence:
            requirement_level = "STRONG"
            break

    logical_connective = "AND"
    if re.search(r"\b" + "or" + r"\b", rule_sentence):
        logical_connective = "OR"

    atomic_rules_list = all_transformed_rules_list[i]
    for atomic_rule in atomic_rules_list:
        if atomic_rule.entity in entity_atomic_rules:
            entity_atomic_rules[atomic_rule.entity].append(atomic_rule)
        else:
            entity_atomic_rules[atomic_rule.entity] = [atomic_rule]

    for entity in entity_atomic_rules:
        atomic_rules = entity_atomic_rules[entity]
        rule_obj = Rule(atomic_rules, logical_connective, requirement_level, condition, rule_sentence)

        rule_number = 1
        if entity in entity_rule_number:
            entity_rule_number[entity] += 1
            rule_number = entity_rule_number[entity]
        else:
            entity_rule_number[entity] = rule_number
        graph.add_node(entity + " rule " + str(rule_number), data=rule_obj, label=entity + " rule " + str(rule_number))
        graph.add_edge(entity, entity + " rule " + str(rule_number), relation="rule")

        # Create condition relation accordingly
        if condition == "Antecedent rule":
            condition_rules.append(entity + " rule " + str(rule_number))
        elif condition == "Consequent rule" and condition_rules:
            for rule in condition_rules:
                graph.add_edge(rule, entity + " rule " + str(rule_number), relation="condition")
    if condition == "Consequent rule":
        condition_rules = []

# Normalise the entity variables
for node in graph.nodes:
    if type(graph.nodes[node]["data"]) == Entity:
        rules = []
        for neighbor in graph.neighbors(node):
            if type(graph.nodes[neighbor]["data"]) == Rule:
                rules.append(neighbor)

        atomic_rules_per_rule = []
        for rule in rules:
            atomic_rules_per_rule.append(graph.nodes[rule]["data"].atomic_rules)

        variables_per_rule = []
        for atomic_rules in atomic_rules_per_rule:
            variables = []
            for atomic_rule in atomic_rules:
                variables.append(atomic_rule.var)
            variables_per_rule.append(variables)

        #
        unique_entity_variables = []
        for variables in variables_per_rule:
            unique_entity_variables += variables
        unique_entity_variables = list(set(unique_entity_variables))
        #
        unique_entity_variable_vectors = torch.tensor(phraseBERT.encode(unique_entity_variables))
        #
        grouped_entity_variables = []
        for i in range(len(unique_entity_variable_vectors)):
            variable_vector_a = unique_entity_variable_vectors[i]
            for j in range(i + 1, len(unique_entity_variable_vectors)):
                variable_vector_b = unique_entity_variable_vectors[j]
                similarity = compute_cosine_similarity(variable_vector_a, variable_vector_b)
                # Group variables together if they are similar
                if similarity > 0.95:
                    added = False
                    for variable_group in grouped_entity_variables:
                        if unique_entity_variables[i] in variable_group and unique_entity_variables[
                            j] not in variable_group:
                            variable_group.append(unique_entity_variables[j])
                            added = True
                        elif unique_entity_variables[j] in variable_group and unique_entity_variables[
                            i] not in variable_group:
                            variable_group.append(unique_entity_variables[i])
                            added = True
                        elif unique_entity_variables[i] in variable_group and unique_entity_variables[
                            j] in variable_group:
                            added = True
                    if not added:
                        grouped_entity_variables.append([unique_entity_variables[i], unique_entity_variables[j]])
        #
        transformed_entity_variables = {}
        for variable_group in grouped_entity_variables:
            selected_variable = random.choice(variable_group)
            transformed_entity_variables[selected_variable] = variable_group

        # Replace the variables in the atomic rules
        for atomic_rules in atomic_rules_per_rule:
            for atomic_rule in atomic_rules:
                for key, values in transformed_entity_variables.items():
                    if atomic_rule.var in values:
                        atomic_rule.var = key

# Transformed the values from string to according type

values = []
for node in graph.nodes:
    if type(graph.nodes[node]["data"]) == Rule:
        atomic_rules = graph.nodes[node]["data"].atomic_rules
        rule_values = []
        for atomic_rule in atomic_rules:
            rule_values.append(atomic_rule.value)
        values.extend(rule_values)
values = list(set(values))

transformed_values = {}
value_seed = 999999

for value in values:
    if is_float(value):
        transformed_values[value] = float(value)
    else:
        transformed_values[value] = float(value_seed)
        value_seed += 1

for node in graph.nodes:
    if type(graph.nodes[node]["data"]) == Rule:
        atomic_rules = graph.nodes[node]["data"].atomic_rules
        for atomic_rule in atomic_rules:
            atomic_rule.value = transformed_values[atomic_rule.value]


def check_entity_contradiction(graph):
    contradiction_entities = []
    for node in graph.nodes:
        if type(graph.nodes[node]["data"]) == Entity:
            solver = Solver()
            rules = []
            literals = {}
            entity_clauses = []
            literal_num = 1
            for neighbor in graph.neighbors(node):
                if type(graph.nodes[neighbor]["data"]) == Rule:
                    rules.append(neighbor)
            for rule in rules:
                clauses = []
                atomic_rules = graph.nodes[rule]["data"].atomic_rules
                for atomic_rule in atomic_rules:
                    var = atomic_rule.var
                    op = atomic_rule.op
                    value = atomic_rule.value

                    if var not in literals:
                        literals[var] = Real("l" + str(literal_num))
                        literal_num += 1

                    if op == "=":
                        clauses.append(literals[var] == value)
                    elif op == "!=":
                        clauses.append(literals[var] != value)
                    elif op == ">":
                        clauses.append(literals[var] > value)
                    elif op == ">=":
                        clauses.append(literals[var] >= value)
                    elif op == "<":
                        clauses.append(literals[var] < value)
                    elif op == "<=":
                        clauses.append(literals[var] <= value)

                if graph.nodes[rule]["data"].logical_connective == "AND":
                    entity_clauses.append(And(clauses))
                else:
                    entity_clauses.append(Or(clauses))
            solver.add(And(entity_clauses))
            if solver.check() == unsat:
                contradiction_entities.append(node)
                print(f"Contradiction found in entity {node}")
    return contradiction_entities


contradiction_entities = check_entity_contradiction(graph)
print("------------------------------------------------------")

# From tne contradiction entities, check if the contradiction exists in a single rule
def check_single_rule_contradiction(graph, entity):
    contradiction_rules = []
    for neighbor in graph.neighbors(entity):
        if type(graph.nodes[neighbor]["data"]) == Rule:
            solver = Solver()
            rule = graph.nodes[neighbor]["data"]
            atomic_rules = rule.atomic_rules
            clauses = []
            literals = {}
            literal_num = 1
            for atomic_rule in atomic_rules:
                var = atomic_rule.var
                op = atomic_rule.op
                value = atomic_rule.value

                if var not in literals:
                    literals[var] = Real("l" + str(literal_num))
                    literal_num += 1

                if op == "=":
                    clauses.append(literals[var] == value)
                elif op == "!=":
                    clauses.append(literals[var] != value)
                elif op == ">":
                    clauses.append(literals[var] > value)
                elif op == ">=":
                    clauses.append(literals[var] >= value)
                elif op == "<":
                    clauses.append(literals[var] < value)
                elif op == "<=":
                    clauses.append(literals[var] <= value)

            if rule.logical_connective == "AND":
                solver.add(And(clauses))
            else:
                solver.add(Or(clauses))
            if solver.check() == unsat:
                print(f"Contradiction found in rule {neighbor}")
                contradiction_rules.append(neighbor)
    print(f"No single rule contradiction found  in entity {entity}")
    return contradiction_rules


single_rule_contradiction_rules = []
for entity in contradiction_entities:
    single_rule_contradiction_rules += check_single_rule_contradiction(graph, entity)

print("------------------------------------------------------")


# From the contradiction entities, check if it is from a direct contradiction
# Type 1 direct contradiction means is there a contradiction between tne entity rules
def check_entity_direct_contradiction_type_1(graph, entity):
    entity_rules = []
    entity_clauses = []
    literals = {}
    literal_num = 1
    solver = Solver()
    for neighbor in graph.neighbors(entity):
        if type(graph.nodes[neighbor][
                    "data"]) == Rule and neighbor not in single_rule_contradiction_rules:
            rule = graph.nodes[neighbor]["data"]
            if rule.condition == "Entity rule":
                entity_rules.append(neighbor)

    for i in range(len(entity_rules)):
        entity_rule_a = entity_rules[i]
        entity_rule_a_obj = graph.nodes[entity_rule_a]["data"]
        atomic_rules = entity_rule_a_obj.atomic_rules
        entity_rule_a_clauses = []
        for atomic_rule in atomic_rules:
            var = atomic_rule.var
            op = atomic_rule.op
            value = atomic_rule.value
            if var not in literals:
                literals[var] = Real("l" + str(literal_num))
                literal_num += 1

            if op == "=":
                entity_rule_a_clauses.append(literals[var] == value)
            elif op == "!=":
                entity_rule_a_clauses.append(literals[var] != value)
            elif op == ">":
                entity_rule_a_clauses.append(literals[var] > value)
            elif op == ">=":
                entity_rule_a_clauses.append(literals[var] >= value)
            elif op == "<":
                entity_rule_a_clauses.append(literals[var] < value)
            elif op == "<=":
                entity_rule_a_clauses.append(literals[var] <= value)

        if entity_rule_a_obj.logical_connective == "AND":
            entity_clauses.append(And(entity_rule_a_clauses))
        else:
            entity_clauses.append(Or(entity_rule_a_clauses))

        for j in range(i + 1, len(entity_rules)):
            entity_rule_b = entity_rules[j]
            entity_rule_b_obj = graph.nodes[entity_rule_b]["data"]
            atomic_rules = entity_rule_b_obj.atomic_rules
            entity_rule_b_clauses = []
            for atomic_rule in atomic_rules:
                var = atomic_rule.var
                op = atomic_rule.op
                value = atomic_rule.value
                if var not in literals:
                    literals[var] = Real("l" + str(literal_num))
                    literal_num += 1

                if op == "=":
                    entity_rule_b_clauses.append(literals[var] == value)
                elif op == "!=":
                    entity_rule_b_clauses.append(literals[var] != value)
                elif op == ">":
                    entity_rule_b_clauses.append(literals[var] > value)
                elif op == ">=":
                    entity_rule_b_clauses.append(literals[var] >= value)
                elif op == "<":
                    entity_rule_b_clauses.append(literals[var] < value)
                elif op == "<=":
                    entity_rule_b_clauses.append(literals[var] <= value)

            if entity_rule_b_obj.logical_connective == "AND":
                entity_clauses.append(And(entity_rule_b_clauses))
            else:
                entity_clauses.append(Or(entity_rule_b_clauses))

            solver.add(And(entity_clauses))
            if solver.check() == unsat:
                print(f"Direct contradiction type 1 found in entity {entity}")
                print(f"Entity rule a {entity_rule_a}, Entity rule b {entity_rule_b}")
                print(entity_clauses)
            solver.reset()
            entity_clauses = entity_rule_a_clauses.copy()
        entity_clauses = []
    print(f"No direct contradiction type 1 found in entity {entity}")


for entity in contradiction_entities:
    check_entity_direct_contradiction_type_1(graph, entity)

print("------------------------------------------------------")


def locate_contradiction_rules_with_literals(graph, evaluated_rules, current_rule):
    rule_literals = {}
    literals = {}
    literal_num = 1
    for rule in evaluated_rules:
        rule_obj = graph.nodes[rule]["data"]
        atomic_rules = rule_obj.atomic_rules
        for atomic_rule in atomic_rules:
            var = atomic_rule.var
            if var not in literals:
                literals[var] = Real("l" + str(literal_num))
                literal_num += 1
            if rule not in rule_literals:
                rule_literals[rule] = []
                rule_literals[rule].append(literals[var])
            else:
                if literals[var] not in rule_literals[rule]:
                    rule_literals[rule].append(literals[var])

    current_rule_obj = graph.nodes[current_rule]["data"]
    atomic_rules = current_rule_obj.atomic_rules
    for atomic_rule in atomic_rules:
        var = atomic_rule.var
        if var not in literals:
            literals[var] = Real("l" + str(literal_num))
            literal_num += 1
        if current_rule not in rule_literals:
            rule_literals[current_rule] = []
            rule_literals[current_rule].append(literals[var])
        else:
            if literals[var] not in rule_literals[current_rule]:
                rule_literals[current_rule].append(literals[var])

    current_rule_literals = rule_literals[current_rule]
    for literal in current_rule_literals:
        for rule in evaluated_rules:
            if literal in rule_literals[rule]:
                print(f"literal {literal} in {current_rule} and {rule}")


def check_entity_direct_contradiction_type_2(graph, entity):
    entity_rules = []
    consequent_rules = []
    entity_clauses = []
    literals = {}
    literal_num = 1
    for neighbor in graph.neighbors(entity):
        if type(graph.nodes[neighbor]["data"]) == Rule and neighbor not in single_rule_contradiction_rules:
            rule = graph.nodes[neighbor]["data"]
            if rule.condition == "Entity rule":
                entity_rules.append(neighbor)
            elif rule.condition == "Consequent rule":
                consequent_rules.append(neighbor)

    for entity_rule in entity_rules:
        solver = Solver()
        entity_rule_obj = graph.nodes[entity_rule]["data"]
        atomic_rules = entity_rule_obj.atomic_rules
        entity_rule_clauses = []
        for atomic_rule in atomic_rules:
            var = atomic_rule.var
            op = atomic_rule.op
            value = atomic_rule.value
            if var not in literals:
                literals[var] = Real("l" + str(literal_num))
                literal_num += 1

            if op == "=":
                entity_rule_clauses.append(literals[var] == value)
            elif op == "!=":
                entity_rule_clauses.append(literals[var] != value)
            elif op == ">":
                entity_rule_clauses.append(literals[var] > value)
            elif op == ">=":
                entity_rule_clauses.append(literals[var] >= value)
            elif op == "<":
                entity_rule_clauses.append(literals[var] < value)
            elif op == "<=":
                entity_rule_clauses.append(literals[var] <= value)
        if entity_rule_obj.logical_connective == "AND":
            entity_clauses.append(And(entity_rule_clauses))
        else:
            entity_clauses.append(Or(entity_rule_clauses))

        for consequent_rule in consequent_rules:
            consequent_rule_obj = graph.nodes[consequent_rule]["data"]
            atomic_rules = consequent_rule_obj.atomic_rules
            consequent_rule_clauses = []
            for atomic_rule in atomic_rules:
                var = atomic_rule.var
                op = atomic_rule.op
                value = atomic_rule.value
                if var not in literals:
                    literals[var] = Real("l" + str(literal_num))
                    literal_num += 1

                if op == "=":
                    consequent_rule_clauses.append(literals[var] == value)
                elif op == "!=":
                    consequent_rule_clauses.append(literals[var] != value)
                elif op == ">":
                    consequent_rule_clauses.append(literals[var] > value)
                elif op == ">=":
                    consequent_rule_clauses.append(literals[var] >= value)
                elif op == "<":
                    consequent_rule_clauses.append(literals[var] < value)
                elif op == "<=":
                    consequent_rule_clauses.append(literals[var] <= value)
            if consequent_rule_obj.logical_connective == "AND":
                entity_clauses.append(And(consequent_rule_clauses))
            else:
                entity_clauses.append(Or(consequent_rule_clauses))

            solver.add(And(entity_clauses))
            if solver.check() == unsat:
                print(f"Direct contradiction type 2 found in entity {entity}")
                print(f"Entity rule {entity_rule}, Consequent rule {consequent_rule}")
            solver.reset()
            entity_clauses = entity_rule_clauses.copy()
        entity_clauses = []
    print(f"No direct contradiction type 2 found in entity {entity}")


for entity in contradiction_entities:
    check_entity_direct_contradiction_type_2(graph, entity)

print("------------------------------------------------------")


def check_entity_direct_contradiction_type_3(graph, entity, ignore_rule):
    antecedent_consequent_rules = {}
    solver = Solver()
    entity_clauses = []
    literals = {}
    literal_num = 1
    for neighbor in graph.neighbors(entity):
        if type(graph.nodes[neighbor][
                    "data"]) == Rule and neighbor not in single_rule_contradiction_rules and neighbor != ignore_rule:
            rule = graph.nodes[neighbor]["data"]
            if rule.condition == "Antecedent rule":
                antecedent_consequent_rules[neighbor] = []

    for antecedent_rule in antecedent_consequent_rules:
        for neighbor in graph.neighbors(antecedent_rule):
            if type(graph.nodes[neighbor][
                        "data"]) == Rule and neighbor not in single_rule_contradiction_rules and neighbor != ignore_rule:
                rule = graph.nodes[neighbor]["data"]
                if rule.condition == "Consequent rule" and rule.atomic_rules[0].entity == entity:
                    antecedent_consequent_rules[antecedent_rule].append(neighbor)

    # print(antecedent_consequent_rules)

    for antecedent_rule in antecedent_consequent_rules:
        consequent_rules = antecedent_consequent_rules[antecedent_rule]
        for i in range(len(consequent_rules)):
            consequent_rule_a = consequent_rules[i]
            consequent_rule_a_clauses = []
            consequent_rule_a = graph.nodes[consequent_rule_a]["data"]
            atomic_rules = consequent_rule_a.atomic_rules
            for atomic_rule in atomic_rules:
                var = atomic_rule.var
                op = atomic_rule.op
                value = atomic_rule.value
                if var not in literals:
                    literals[var] = Real("l" + str(literal_num))
                    literal_num += 1

                if op == "=":
                    consequent_rule_a_clauses.append(literals[var] == value)
                elif op == "!=":
                    consequent_rule_a_clauses.append(literals[var] != value)
                elif op == ">":
                    consequent_rule_a_clauses.append(literals[var] > value)
                elif op == ">=":
                    consequent_rule_a_clauses.append(literals[var] >= value)
                elif op == "<":
                    consequent_rule_a_clauses.append(literals[var] < value)
                elif op == "<=":
                    consequent_rule_a_clauses.append(literals[var] <= value)
            if consequent_rule_a.logical_connective == "AND":
                entity_clauses.append(And(consequent_rule_a_clauses))
            else:
                entity_clauses.append(Or(consequent_rule_a_clauses))

            for j in range(i + 1, len(consequent_rules)):
                consequent_rule_b = consequent_rules[j]
                consequent_rule_b_clauses = []
                consequent_rule_b = graph.nodes[consequent_rule_b]["data"]
                atomic_rules = consequent_rule_b.atomic_rules
                for atomic_rule in atomic_rules:
                    var = atomic_rule.var
                    op = atomic_rule.op
                    value = atomic_rule.value
                    if var not in literals:
                        literals[var] = Real("l" + str(literal_num))
                        literal_num += 1

                    if op == "=":
                        consequent_rule_b_clauses.append(literals[var] == value)
                    elif op == "!=":
                        consequent_rule_b_clauses.append(literals[var] != value)
                    elif op == ">":
                        consequent_rule_b_clauses.append(literals[var] > value)
                    elif op == ">=":
                        consequent_rule_b_clauses.append(literals[var] >= value)
                    elif op == "<":
                        consequent_rule_b_clauses.append(literals[var] < value)
                    elif op == "<=":
                        consequent_rule_b_clauses.append(literals[var] <= value)
                if consequent_rule_b.logical_connective == "AND":
                    entity_clauses.append(And(consequent_rule_b_clauses))
                else:
                    entity_clauses.append(Or(consequent_rule_b_clauses))

                solver.add(And(entity_clauses))
                if solver.check() == unsat:
                    print(f"Direct contradiction type 3 found in entity {entity}")
                    print(
                        f"Antecedent rule {antecedent_rule}, Consequent rule {consequent_rule_a}, Consequent rule {consequent_rule_b}")
                solver.reset()
                entity_clauses = consequent_rule_a_clauses.copy()
            entity_clauses = []
    print(f"No direct contradiction type 3 found in entity {entity}")


for entity in contradiction_entities:
    check_entity_direct_contradiction_type_3(graph, entity, None)

print("------------------------------------------------------")


def check_conditional_contradiction(graph, entity):
    entity_rules = []
    antecedent_rules = []
    entity_clauses = []
    literals = {}
    literal_num = 1
    for neighbor in graph.neighbors(entity):
        if type(graph.nodes[neighbor]["data"]) == Rule and neighbor not in single_rule_contradiction_rules:
            rule = graph.nodes[neighbor]["data"]
            if rule.condition == "Entity rule":
                entity_rules.append(neighbor)
            elif rule.condition == "Antecedent rule":
                antecedent_rules.append(neighbor)

    for entity_rule in entity_rules:
        solver = Solver()
        entity_rule_obj = graph.nodes[entity_rule]["data"]
        atomic_rules = entity_rule_obj.atomic_rules
        entity_rule_clauses = []
        for atomic_rule in atomic_rules:
            var = atomic_rule.var
            op = atomic_rule.op
            value = atomic_rule.value
            if var not in literals:
                literals[var] = Real("l" + str(literal_num))
                literal_num += 1

            if op == "=":
                entity_rule_clauses.append(literals[var] == value)
            elif op == "!=":
                entity_rule_clauses.append(literals[var] != value)
            elif op == ">":
                entity_rule_clauses.append(literals[var] > value)
            elif op == ">=":
                entity_rule_clauses.append(literals[var] >= value)
            elif op == "<":
                entity_rule_clauses.append(literals[var] < value)
            elif op == "<=":
                entity_rule_clauses.append(literals[var] <= value)
        if entity_rule_obj.logical_connective == "AND":
            entity_clauses.append(And(entity_rule_clauses))
        else:
            entity_clauses.append(Or(entity_rule_clauses))

        for antecedent_rule in antecedent_rules:
            antecedent_rule_obj = graph.nodes[antecedent_rule]["data"]
            atomic_rules = antecedent_rule_obj.atomic_rules
            antecedent_rule_clauses = []
            for atomic_rule in atomic_rules:
                var = atomic_rule.var
                op = atomic_rule.op
                value = atomic_rule.value
                if var not in literals:
                    literals[var] = Real("l" + str(literal_num))
                    literal_num += 1

                if op == "=":
                    antecedent_rule_clauses.append(literals[var] == value)
                elif op == "!=":
                    antecedent_rule_clauses.append(literals[var] != value)
                elif op == ">":
                    antecedent_rule_clauses.append(literals[var] > value)
                elif op == ">=":
                    antecedent_rule_clauses.append(literals[var] >= value)
                elif op == "<":
                    antecedent_rule_clauses.append(literals[var] < value)
                elif op == "<=":
                    antecedent_rule_clauses.append(literals[var] <= value)
            if antecedent_rule_obj.logical_connective == "AND":
                entity_clauses.append(And(antecedent_rule_clauses))
            else:
                entity_clauses.append(Or(antecedent_rule_clauses))

            solver.add(And(entity_clauses))
            if solver.check() == unsat:
                print(f"Conditional contradiction found in entity {entity}")
                print(f"Entity rule {entity_rule}, Antecedent rule {antecedent_rule}")
            solver.reset()
            entity_clauses = entity_rule_clauses.copy()
        entity_clauses = []
    print(f"No Conditional contradiction found in entity {entity}")


for entity in contradiction_entities:
    check_conditional_contradiction(graph, entity)