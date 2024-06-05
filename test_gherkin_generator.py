import unittest
from gherkin_generator import extract_scenarios, generate_gherkin_steps

class TestGherkinGenerator(unittest.TestCase):
    def test_extract_scenarios(self):
        requirement = "When the user clicks the login button, the system must validate the credentials against the database. If the user's account is locked, then the system should display a message indicating that the account is locked."
        expected_scenarios = [
            "When the user clicks the login button, the system must validate the credentials against the database.",
            "If the user's account is locked, then the system should display a message indicating that the account is locked."
        ]
        scenarios = extract_scenarios(requirement)
        self.assertEqual(scenarios, expected_scenarios)

    def test_generate_gherkin_steps(self):
        scenario = "When the user clicks the login button, the system must validate the credentials against the database."
        expected_gherkin_steps = "Given the user clicks the login button\nWhen the system must validate the credentials against the database\nThen \n"
        gherkin_steps = generate_gherkin_steps(scenario)
        self.assertEqual(gherkin_steps, expected_gherkin_steps)

if __name__ == '__main__':
    unittest.main()