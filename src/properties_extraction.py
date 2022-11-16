import re

PROPERTIES = ["value", "error", "idempotent"]


def extract_properties(behaviours):
    behaviours = behaviours.split(";")
    behaviours = behaviours[:len(behaviours) - 1]
    properties = []
    for behaviour in behaviours:
        behaviour = behaviour.strip()
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
            properties.append(property_name + " " + operator + " " + value)
        elif re.search(r"\b" + "unknown" + r"\b", behaviour_property):
            property_name = "value"
            value = "unknown"
            if sentiment == "True":
                operator = "="
            else:
                operator = "!="
            properties.append(property_name + " " + operator + " " + value)
        elif re.search(r"\b\d+\b", behaviour_property) and "-" in behaviour_property:
            property_name = "value"
            values = re.findall(r"\b\d+\b", behaviour_property)
            properties.append(property_name + " " + ">" + " " + values[0])
            properties.append(property_name + " " + "<" + " " + values[1])
        elif re.search(r"\b" + "Empty" + r"\b", behaviour_property) or re.search(r"\b" + "empty" + r"\b",
                                                                                 behaviour_property):
            property_name = "value"
            value = "empty"
            if sentiment == "True":
                operator = "="
            else:
                operator = "!="
            properties.append(property_name + " " + operator + " " + value)
        elif re.search(r"\b" + "decreased" + r"\b", behaviour_property) and re.search(r"\b" + "below" + r"\b",
                                                                                      behaviour_property):
            property_name = "value"
            value = behaviour_property[re.search(r"\b" + "below" + r"\b", behaviour_property).end():].strip()
            if sentiment == "True":
                operator = "<"
            else:
                operator = ">="
            properties.append(property_name + " " + operator + " " + value)
        elif re.search(r"\b" + "class elective" + r"\b", behaviour_property):
            property_name = "unrecognized option"
            value = "elective"
            if sentiment == "True":
                operator = "="
            else:
                operator = "!="
            properties.append(property_name + " " + operator + " " + value)
        elif re.search(r"\b" + "class critical" + r"\b", behaviour_property):
            property_name = "unrecognized option"
            value = "critical"
            if sentiment == "True":
                operator = "="
            else:
                operator = "!="
            properties.append(property_name + " " + operator + " " + value)
        elif (re.search(r"\b" + "return" + r"\b", behaviour_property) or re.search(r"\b" + "returned" + r"\b",
                                                                                   behaviour_property)) and re.search(
                r"\b" + "response" + r"\b",
                behaviour_property):
            property_name = "value"
            value = ".".join(re.findall(r"\b\d+\b", behaviour_property))
            if sentiment == "True":
                operator = "="
            else:
                operator = "!="
            properties.append(property_name + " " + operator + " " + value)
        elif re.search(r"\b" + "Response Code" + r"\b", behaviour_property):
            property_name = "value"
            value = ".".join(re.findall(r"\b\d+\b", behaviour_property))
            if value == "":
                value = "error"
            if sentiment == "True":
                operator = "="
            else:
                operator = "!="
            properties.append(property_name + " " + operator + " " + value)
        elif re.search(r"\b" + "error code" + r"\b", behaviour_property):
            property_name = "value"
            value = ".".join(re.findall(r"\b\d+\b", behaviour_property))
            if value == "":
                value = "error"
            if sentiment == "True":
                operator = "="
            else:
                operator = "!="
            properties.append(property_name + " " + operator + " " + value)
        elif re.search(r"\b" + "default value" + r"\b", behaviour_property):
            property_name = "value"
            value = "default"
            if sentiment == "True":
                operator = "="
            else:
                operator = "!="
            properties.append(property_name + " " + operator + " " + value)
        elif re.search(r"\b" + "value" + r"\b", behaviour_property) and re.search(r"\b" + "be" + r"\b", behaviour_property):
            property_name = "value"
            value = behaviour_property[re.search(r"\b" + "be" + r"\b", behaviour_property).end():].strip()
            if sentiment == "True":
                operator = "="
            else:
                operator = "!="
            properties.append(property_name + " " + operator + " " + value)
        # Searching for property "error" in the behaviour
        elif re.search(r"\b" + "ignored" + r"\b", behaviour_property):
            property_name = "error"
            value = "ignore"
            if sentiment == "True":
                operator = "="
            else:
                operator = "!="
            properties.append(property_name + " " + operator + " " + value)
        elif re.search(r"\b" + "ignore" + r"\b", behaviour_property):
            property_name = "error"
            value = "ignore"
            if sentiment == "True":
                operator = "="
            else:
                operator = "!="
            properties.append(property_name + " " + operator + " " + value)
        elif re.search(r"\b" + "message format error" + r"\b", behaviour_property):
            property_name = "error"
            value = "message format error"
            if sentiment == "True":
                operator = "="
            else:
                operator = "!="
            properties.append(property_name + " " + operator + " " + value)
        elif re.search(r"\b" + "reject" + r"\b", behaviour_property) and re.search(r"\b" + "message" + r"\b",
                                                                                   behaviour_property):
            property_name = "error"
            value = "reject message"
            if sentiment == "True":
                operator = "="
            else:
                operator = "!="
            properties.append(property_name + " " + operator + " " + value)
        elif re.search(r"\b" + "rejected" + r"\b", behaviour_property):
            property_name = "error"
            value = "reject"
            if sentiment == "True":
                operator = "="
            else:
                operator = "!="
            properties.append(property_name + " " + operator + " " + value)
        elif re.search(r"\b" + "Rejecting" + r"\b", behaviour_property) and re.search(r"\b" + "message" + r"\b",
                                                                                      behaviour_property):
            property_name = "error"
            value = "reject message"
            if sentiment == "True":
                operator = "="
            else:
                operator = "!="
            properties.append(property_name + " " + operator + " " + value)
        elif re.search(r"\b" + "ICMP errors" + r"\b", behaviour_property):
            property_name = "error"
            value = "ICMP errors"
            if sentiment == "True":
                operator = "="
            else:
                operator = "!="
            properties.append(property_name + " " + operator + " " + value)
        # Searching for property "idempotent" in the behaviour
        elif re.search(r"\b" + "idempotent" + r"\b", behaviour_property):
            property_name = "idempotent"
            value = sentiment
            properties.append(property_name + " " + "=" + " " + value)
        else:
            properties.append(behaviour_property + " " + "=" + " " + sentiment)

    return properties


print(extract_properties("Lengths 9-15 = True; message be sent = False; message be processed as a message format error = True;"))
