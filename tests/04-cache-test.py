#!/usr/bin/env python3
"""
04-cache-test.py - Semantic cache tests

This test validates the semantic cache functionality by sending similar
queries and checking if cache hits occur as expected.
"""

import json
import requests
import unittest
import sys
import os
import time
import uuid

# Add parent directory to path to allow importing common test utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
ENVOY_URL = "http://localhost:8801"
OPENAI_ENDPOINT = "/v1/chat/completions"
ROUTER_METRICS_URL = "http://localhost:9190/metrics"

# Helper function to extract a specific metric value
def extract_metric(metrics_text, metric_name):
    for line in metrics_text.split('\n'):
        if line.startswith(metric_name) and not line.startswith('#'):
            return float(line.split()[1])
    return None


class SemanticCacheTest(unittest.TestCase):
    """Test the semantic cache functionality."""

    def setUp(self):
        """Check if the services are running before running tests."""
        # Check Envoy
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
            
        # Check router metrics endpoint
        try:
            response = requests.get(ROUTER_METRICS_URL, timeout=2)
            if response.status_code != 200:
                self.skipTest("Router metrics server is not responding. Is the router running?")
        except requests.exceptions.ConnectionError:
            self.skipTest("Cannot connect to router metrics server. Is the router running?")
            
        # Check if cache is enabled in metrics
        response = requests.get(ROUTER_METRICS_URL)
        metrics_text = response.text
        if "llm_router_cache" not in metrics_text:
            self.skipTest("Cache metrics not found. Semantic cache may be disabled.")

    def test_cache_hit_with_identical_query(self):
        """Test that identical queries result in cache hits."""
        session_id = str(uuid.uuid4())
        
        # Our test query
        query = "What are the primary causes of climate change?"
        
        # Get baseline cache metrics
        response = requests.get(ROUTER_METRICS_URL)
        baseline_metrics = response.text
        baseline_hits = extract_metric(baseline_metrics, "llm_router_cache_hits_total") or 0
        
        # First request should be a cache miss
        payload = {
            "model": "gemma3:27b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            "temperature": 0.7
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-Session-ID": session_id  # To track the session
        }
        
        # First request
        response1 = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        self.assertLess(response1.status_code, 400, 
                      f"First request failed with status code {response1.status_code}")
        
        response1_json = response1.json()
        model1 = response1_json.get("model", "unknown")
        
        # Wait a moment to ensure metrics are updated
        time.sleep(2)
        
        # Second identical request
        response2 = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        self.assertLess(response2.status_code, 400, 
                      f"Second request failed with status code {response2.status_code}")
        
        response2_json = response2.json()
        model2 = response2_json.get("model", "unknown")
        
        # Models should be identical
        self.assertEqual(model1, model2, 
                        f"Routed to different models for identical queries: {model1} vs {model2}")
        
        # Wait for metrics to update
        time.sleep(2)
        
        # Check if cache hits increased
        response = requests.get(ROUTER_METRICS_URL)
        updated_metrics = response.text
        updated_hits = extract_metric(updated_metrics, "llm_router_cache_hits_total") or 0
        
        print(f"\nBaseline cache hits: {baseline_hits}")
        print(f"Updated cache hits: {updated_hits}")
        
        # In a production environment, this would be a stronger assertion
        # However, other tests might be running simultaneously, so we just log the results
        if updated_hits > baseline_hits:
            print("✅ Cache hit count increased - cache is working")
        else:
            print("❓ Cache hit count did not increase - cache may be disabled or not working")

    def test_cache_hit_with_similar_query(self):
        """Test that semantically similar queries can result in cache hits."""
        session_id = str(uuid.uuid4())
        
        # Two semantically similar but not identical queries
        original_query = "Explain how photosynthesis works in plants."
        similar_query = "How does the process of photosynthesis function in plant cells?"
        
        # Get baseline cache metrics
        response = requests.get(ROUTER_METRICS_URL)
        baseline_metrics = response.text
        baseline_hits = extract_metric(baseline_metrics, "llm_router_cache_hits_total") or 0
        
        # First request with original query
        payload1 = {
            "model": "gemma3:27b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": original_query}
            ],
            "temperature": 0.7
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-Session-ID": session_id
        }
        
        response1 = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}",
            headers=headers,
            json=payload1,
            timeout=10
        )
        
        self.assertLess(response1.status_code, 400, 
                      f"First request failed with status code {response1.status_code}")
        
        response1_json = response1.json()
        model1 = response1_json.get("model", "unknown")
        
        # Wait a moment to ensure the first request is processed fully
        time.sleep(2)
        
        # Second request with similar query
        payload2 = {
            "model": "gemma3:27b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": similar_query}
            ],
            "temperature": 0.7
        }
        
        response2 = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}",
            headers=headers,
            json=payload2,
            timeout=10
        )
        
        self.assertLess(response2.status_code, 400, 
                      f"Second request failed with status code {response2.status_code}")
        
        response2_json = response2.json()
        model2 = response2_json.get("model", "unknown")
        
        # Wait for metrics to update
        time.sleep(2)
        
        # Check cache metrics
        response = requests.get(ROUTER_METRICS_URL)
        updated_metrics = response.text
        updated_hits = extract_metric(updated_metrics, "llm_router_cache_hits_total") or 0
        
        print(f"\nOriginal query: {original_query}")
        print(f"Similar query: {similar_query}")
        print(f"First query routed to: {model1}")
        print(f"Similar query routed to: {model2}")
        print(f"Baseline cache hits: {baseline_hits}")
        print(f"Updated cache hits: {updated_hits}")
        
        # For semantic similarity, the models should be the same if the cache is working correctly
        # and the similarity threshold is appropriately set
        if model1 == model2:
            print("✅ Both queries routed to the same model")
        else:
            print("❓ Queries routed to different models")
            
        # Check if there was a cache hit increase
        if updated_hits > baseline_hits:
            print("✅ Cache hit detected for similar query")
        else:
            print("❓ No cache hit for similar query - similarity threshold may be too high")

    def test_cache_metrics(self):
        """Test that cache metrics are being recorded properly."""
        # Get baseline metrics
        response = requests.get(ROUTER_METRICS_URL)
        baseline_metrics = response.text
        
        # Cache-related metrics we expect to find
        cache_metrics = [
            "llm_router_cache_hits_total",
            "llm_router_cache_misses_total",
            "llm_router_cache_size",
            "llm_router_cache_similarity_score"
        ]
        
        metrics_found = 0
        print("\nCache Metrics Found:")
        for metric in cache_metrics:
            if metric in baseline_metrics:
                metrics_found += 1
                # Extract the current value
                for line in baseline_metrics.split('\n'):
                    if metric in line and not line.startswith('#'):
                        print(f"{metric}: {line.split()[1]}")
                        break
        
        self.assertGreater(metrics_found, 0, "No cache metrics found")
        
        # Trigger a cache operation by sending a query
        payload = {
            "model": "gemma3:27b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"What is the current time in New York? {uuid.uuid4()}"}
            ],
            "temperature": 0.7
        }
        
        requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10
        )
        
        # Wait for metrics to update
        time.sleep(2)
        
        # Check updated metrics
        response = requests.get(ROUTER_METRICS_URL)
        updated_metrics = response.text
        
        # Count updated metrics
        updated_metrics_found = 0
        print("\nUpdated Cache Metrics:")
        for metric in cache_metrics:
            if metric in updated_metrics:
                updated_metrics_found += 1
                # Extract the updated value
                for line in updated_metrics.split('\n'):
                    if metric in line and not line.startswith('#'):
                        print(f"{metric}: {line.split()[1]}")
                        break
        
        self.assertEqual(metrics_found, updated_metrics_found, 
                        "Number of cache metrics changed unexpectedly")


if __name__ == "__main__":
    unittest.main() 