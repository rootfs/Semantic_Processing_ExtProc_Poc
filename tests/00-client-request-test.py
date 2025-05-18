#!/usr/bin/env python3
"""
00-client-request-test.py - Basic client request tests

This test validates that the Envoy proxy is running and accepting requests,
and that basic request formatting works correctly.
"""

import json
import requests
import unittest
import sys
import os

# Add parent directory to path to allow importing common test utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
ENVOY_URL = "http://localhost:8801"
OPENAI_ENDPOINT = "/v1/chat/completions"


class ClientRequestTest(unittest.TestCase):
    """Test basic client requests to the Envoy proxy."""

    def setUp(self):
        """Check if the Envoy server is running before running tests."""
        try:
            # Use a basic POST request to check if Envoy is up
            payload = {
                "model": "gemma3:27b",
                "messages": [{"role": "user", "content": "test"}]
            }
            response = requests.post(
                f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            # If we get a 5xx error, the service is probably down
            if response.status_code >= 500:
                self.skipTest(f"Envoy server returned server error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            self.skipTest("Cannot connect to Envoy server. Is it running?")

    def test_basic_request_format(self):
        """Test that a well-formed request is accepted by the server."""
        payload = {
            "model": "gemma3:27b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10
        )
        
        # We're only checking that the request is accepted, not the content of the response
        # A 4xx response means the request format was rejected
        # A 5xx response might mean the backend is not available
        self.assertLess(response.status_code, 400, 
                        f"Request was rejected with status code {response.status_code}")
        
        # Check that response has proper content type
        self.assertEqual(response.headers.get("Content-Type"), "application/json")
        
        # Attempt to parse the response body as JSON
        try:
            response_json = response.json()
            # Basic structure check
            self.assertIn("model", response_json, "Response is missing 'model' field")
        except json.JSONDecodeError:
            self.fail("Response is not valid JSON")

    def test_malformed_request(self):
        """Test that a malformed request is properly rejected."""
        # Missing required fields
        payload = {
            "model": "gemma3:27b",
            # Missing 'messages' field
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10
        )
        
        # Expect a 4xx client error, not a 5xx server error
        self.assertTrue(400 <= response.status_code < 500, 
                        f"Expected 4xx status code, got {response.status_code}")

    def test_domain_specific_queries(self):
        """Test sending different domain-specific queries."""
        test_cases = [
            {
                "name": "Math Query",
                "messages": [
                    {"role": "system", "content": "You are a math teacher."},
                    {"role": "user", "content": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?"}
                ]
            },
            {
                "name": "Creative Writing Query",
                "messages": [
                    {"role": "system", "content": "You are a creative writer."},
                    {"role": "user", "content": "Write a short story about a space cat."}
                ]
            },
            {
                "name": "General Knowledge Query",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What are the main features of the Great Barrier Reef?"}
                ]
            }
        ]
        
        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                payload = {
                    "model": "gemma3:27b",  # Use a default model, routing should change it
                    "messages": test_case["messages"],
                    "temperature": 0.7
                }
                
                response = requests.post(
                    f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=30  # Longer timeout for actual responses
                )
                
                self.assertLess(response.status_code, 400, 
                                f"{test_case['name']} request failed with status {response.status_code}")
                
                # Print some information about the response
                print(f"\n{test_case['name']} response:")
                print(f"Status Code: {response.status_code}")
                print(f"Model: {response.json().get('model', 'Not specified')}")


if __name__ == "__main__":
    unittest.main() 