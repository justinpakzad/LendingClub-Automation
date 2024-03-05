from app import app
import unittest
import json


class FlaskAppTestCase(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True
        with open("test_data/test_data.json", "r") as f:
            self.test_data = json.load(f)

    def test_predict_accepted(self):
        url = "/predict/loan-approval"
        loan_approval_test_data = self.test_data["accepted"]

        response = self.client.post(
            url,
            data=json.dumps(loan_approval_test_data),
            content_type="application/json",
        )

        self.assertNotEqual(response.status_code, 500)
        self.assertEqual(response.status_code, 200)

    def test_predict_grade_rate(self):
        url = "/predict/grade-and-rate"

        grade_rate_test_data = self.test_data["grade_and_rate"]

        response = self.client.post(
            url, data=json.dumps(grade_rate_test_data), content_type="application/json"
        )
        self.assertNotEqual(response.status_code, 500)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
