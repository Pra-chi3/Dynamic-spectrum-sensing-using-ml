"""
RLM Certificate Fix - Ready to Run Script
Combines multiple solutions to fix the connection error
"""

import os
import httpx
from openai import OpenAI
from rlm import RLM
from rlm.logger import RLMLogger
from pathlib import Path

# ============================================================================
# CONFIGURATION - UPDATE THESE WITH YOUR VALUES
# ============================================================================

API_KEY = "eyJraWQiOiIxZTU5Zf..."  # Your full API key
BASE_URL = "https://genfactory.analytics.echonet/genai/api/v2"
CERT_PATH = r"C:/ai_agents/Router/shared/certificate/bundle.pem"
MODEL_NAME = "Meta-Llama-33-70B-Instruct"
TEMPERATURE = 0.2

# ============================================================================
# FIX 1: Set SSL Certificate Environment Variables
# ============================================================================

print("Applying Fix 1: Setting SSL environment variables...")
os.environ['SSL_CERT_FILE'] = CERT_PATH
os.environ['REQUESTS_CA_BUNDLE'] = CERT_PATH
os.environ['CURL_CA_BUNDLE'] = CERT_PATH
print("✓ Environment variables set\n")

# ============================================================================
# FIX 2: Monkey-Patch OpenAI Client to Use Certificate
# ============================================================================

print("Applying Fix 2: Monkey-patching OpenAI client...")

class CertifiedOpenAI(OpenAI):
    """
    Custom OpenAI client that always uses the certificate
    """
    def __init__(self, *args, **kwargs):
        # Force the http_client with certificate if not provided
        if 'http_client' not in kwargs:
            kwargs['http_client'] = httpx.Client(
                verify=CERT_PATH,
                timeout=60.0  # Increase timeout for safety
            )
        super().__init__(*args, **kwargs)

# Apply the monkey-patch to RLM's OpenAI client
import rlm.clients.openai as rlm_openai
original_openai_class = rlm_openai.OpenAI
rlm_openai.OpenAI = CertifiedOpenAI
print("✓ Monkey-patch applied\n")

# ============================================================================
# Initialize RLM with Logging
# ============================================================================

def setup_rlm_with_fixes():
    """
    Set up RLM with all fixes applied and logging enabled
    """
    print("="*70)
    print("INITIALIZING RLM WITH CERTIFICATE FIXES")
    print("="*70 + "\n")
    
    # Create log directory
    log_dir = Path("./rlm_logs_fixed")
    log_dir.mkdir(exist_ok=True)
    
    # Set up logger
    logger = RLMLogger(log_dir=str(log_dir))
    
    try:
        # Initialize RLM
        rlm = RLM(
            backend="openai",
            backend_kwargs={
                "model_name": MODEL_NAME,
                "api_key": API_KEY,
                "base_url": BASE_URL,
                "temperature": TEMPERATURE
            },
            environment="local",
            logger=logger,
            verbose=True
        )
        
        print("\n✓ RLM initialized successfully!\n")
        return rlm
        
    except Exception as e:
        print(f"\n✗ Failed to initialize RLM: {e}\n")
        # Restore original OpenAI class
        rlm_openai.OpenAI = original_openai_class
        raise

# ============================================================================
# Test Functions
# ============================================================================

def test_basic_completion(rlm):
    """Test 1: Simple completion"""
    print("="*70)
    print("TEST 1: Basic Completion")
    print("="*70)
    
    query = "What is 2+2? Answer in one sentence."
    print(f"Query: {query}\n")
    
    try:
        result = rlm.completion(query)
        print(f"✓ Response: {result.response}\n")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        return False


def test_with_context(rlm):
    """Test 2: Completion with context"""
    print("="*70)
    print("TEST 2: Completion with Context")
    print("="*70)
    
    context = """
    Machine Learning is a subset of Artificial Intelligence that enables 
    computers to learn from data without being explicitly programmed.
    Deep Learning uses neural networks with multiple layers.
    Natural Language Processing helps computers understand human language.
    """
    
    query = "What are the AI technologies mentioned?"
    print(f"Query: {query}")
    print(f"Context provided: {len(context)} characters\n")
    
    try:
        result = rlm.completion(query=query, context=context)
        print(f"✓ Response: {result.response}\n")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        return False


def test_code_generation(rlm):
    """Test 3: Code generation"""
    print("="*70)
    print("TEST 3: Code Generation")
    print("="*70)
    
    query = "Write a Python function to calculate factorial. Show the code."
    print(f"Query: {query}\n")
    
    try:
        result = rlm.completion(query)
        print(f"✓ Response: {result.response}\n")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        return False


def test_recursive_task(rlm):
    """Test 4: Task that benefits from recursion"""
    print("="*70)
    print("TEST 4: Recursive Processing")
    print("="*70)
    
    query = "List the first 20 prime numbers."
    print(f"Query: {query}\n")
    
    try:
        result = rlm.completion(query)
        print(f"✓ Response: {result.response}\n")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        return False


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Run all tests with fixes applied
    """
    print("\n" + "="*70)
    print("RLM CERTIFICATE FIX - AUTOMATED TEST SUITE")
    print("="*70 + "\n")
    
    # Initialize RLM
    try:
        rlm = setup_rlm_with_fixes()
    except Exception as e:
        print(f"Setup failed. Please check your configuration.")
        print(f"Error: {e}")
        return
    
    # Run tests
    tests = [
        ("Basic Completion", test_basic_completion),
        ("With Context", test_with_context),
        ("Code Generation", test_code_generation),
        ("Recursive Processing", test_recursive_task)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func(rlm)
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}\n")
            results[test_name] = False
    
    # Print summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! RLM is working correctly with your endpoint.")
    elif passed > 0:
        print("\n⚠️ Some tests passed. RLM is partially working.")
    else:
        print("\n❌ All tests failed. Additional troubleshooting needed.")
    
    print("\nLogs saved to: ./rlm_logs_fixed/")
    print("="*70 + "\n")


# ============================================================================
# Quick Start Example
# ============================================================================

def quick_start():
    """
    Minimal example for quick testing
    """
    # Apply fixes
    os.environ['SSL_CERT_FILE'] = CERT_PATH
    os.environ['REQUESTS_CA_BUNDLE'] = CERT_PATH
    
    class CertOpenAI(OpenAI):
        def __init__(self, *args, **kwargs):
            if 'http_client' not in kwargs:
                kwargs['http_client'] = httpx.Client(verify=CERT_PATH)
            super().__init__(*args, **kwargs)
    
    import rlm.clients.openai as rlm_openai
    rlm_openai.OpenAI = CertOpenAI
    
    # Create RLM
    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": MODEL_NAME,
            "api_key": API_KEY,
            "base_url": BASE_URL,
            "temperature": TEMPERATURE
        },
        verbose=True
    )
    
    # Test it
    result = rlm.completion("What is machine learning?")
    print(result.response)


# ============================================================================
# Run Script
# ============================================================================

if __name__ == "__main__":
    # Run full test suite
    main()
    
    # Or run quick start only:
    # quick_start()
