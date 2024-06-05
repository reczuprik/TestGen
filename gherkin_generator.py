import re
import json
import spacy

# Load the RE-DATA dataset
with open("re-data.json", "r") as file:
    dataset = json.load(file)

# Load the spaCy model for natural language processing
nlp = spacy.load("en_core_web_sm")

# Define a function to extract scenarios from requirements
def extract_scenarios(req_text):
    doc = nlp(req_text)
    scenarios = []
    current_scenario = []
    for token in doc:
        if token.text.lower() in ["when", "if", "after", "before"]:
            if current_scenario:
                scenarios.append(" ".join(current_scenario))
                current_scenario = []
        current_scenario.append(token.text)
    if current_scenario:
        scenarios.append(" ".join(current_scenario))
    return scenarios

# Define a function to generate Gherkin steps from scenarios
def generate_gherkin_steps(scenario):
    parts = scenario.split()
    given, when, then = "", "", ""
    for i, part in enumerate(parts):
        if part.lower() == "when":
            given = " ".join(parts[:i])
            when = " ".join(parts[i+1:])
        elif part.lower() == "then":
            given = " ".join(parts[:i])
            when = " ".join(parts[i:])
            break
    if given and when:
        then = scenario.split("then")[-1].strip()
        return f"Given {given}\nWhen {when}\nThen {then}\n"
    return ""

# Iterate over the dataset and generate Gherkin test cases
for project in dataset:
    for req in project["requirements"]:
        scenarios = extract_scenarios(req["text"])
        for scenario in scenarios:
            gherkin_steps = generate_gherkin_steps(scenario)
            if gherkin_steps:
                print(f"Requirement: {req['text']}")
                print(gherkin_steps)
                print("---")