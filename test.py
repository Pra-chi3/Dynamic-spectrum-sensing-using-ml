"""
RLM Certificate Fix - CORRECTED VERSION
Fixes the attribute error and query parameter issues
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
# Test Functions - CORRECTED
# ============================================================================

def test_basic_completion(rlm):
    """Test 1: Simple completion"""
    print("="*70)
    print("TEST 1: Basic Completion")
    print("="*70)
    
    # Simple prompt - RLM expects just a string, not 'query' parameter
    prompt = "What is 2+2? Answer in one sentence."
    print(f"Prompt: {prompt}\n")
    
    try:
        # CORRECTED: Pass prompt directly, not as query=
        result = rlm.completion(prompt)
        print(f"✓ Response: {result.response}\n")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        import traceback
        traceback.print_exc()
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
    
    # CORRECTED: Include context in the prompt itself
    prompt = f"""Context:
{context}

Question: What are the AI technologies mentioned in the context above?"""
    
    print(f"Prompt length: {len(prompt)} characters\n")
    
    try:
        result = rlm.completion(prompt)
        print(f"✓ Response: {result.response}\n")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_code_generation(rlm):
    """Test 3: Code generation"""
    print("="*70)
    print("TEST 3: Code Generation")
    print("="*70)
    
    prompt = "Write a Python function to calculate factorial. Show the code."
    print(f"Prompt: {prompt}\n")
    
    try:
        result = rlm.completion(prompt)
        print(f"✓ Response: {result.response}\n")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_recursive_task(rlm):
    """Test 4: Task that benefits from recursion"""
    print("="*70)
    print("TEST 4: Recursive Processing")
    print("="*70)
    
    prompt = "List the first 20 prime numbers."
    print(f"Prompt: {prompt}\n")
    
    try:
        result = rlm.completion(prompt)
        print(f"✓ Response: {result.response}\n")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        import traceback
        traceback.print_exc()
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
        import traceback
        traceback.print_exc()
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
            import traceback
            traceback.print_exc()
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
# Alternative: Direct OpenAI Client Test (No RLM)
# ============================================================================

def test_direct_openai():
    """
    Test the endpoint directly with OpenAI client (bypass RLM)
    This helps verify the certificate fix works
    """
    print("\n" + "="*70)
    print("DIRECT OPENAI CLIENT TEST (No RLM)")
    print("="*70 + "\n")
    
    try:
        # Create client with certificate
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            http_client=httpx.Client(verify=CERT_PATH)
        )
        
        print("Sending test request...")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            temperature=TEMPERATURE
        )
        
        print(f"✓ Success!")
        print(f"Response: {response.choices[0].message.content}\n")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Quick Start Example - Minimal Code
# ============================================================================

def quick_start():
    """
    Minimal working example - just the essentials
    """
    print("\n" + "="*70)
    print("QUICK START - MINIMAL EXAMPLE")
    print("="*70 + "\n")
    
    # Apply certificate fixes
    os.environ['SSL_CERT_FILE'] = CERT_PATH
    os.environ['REQUESTS_CA_BUNDLE'] = CERT_PATH
    
    # Monkey-patch OpenAI
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
    print("Testing: What is machine learning?\n")
    result = rlm.completion("What is machine learning? Answer in 2 sentences.")
    print(f"Response: {result.response}\n")


# ============================================================================
# Run Script
# ============================================================================

if __name__ == "__main__":
    # First, test direct OpenAI connection
    print("Step 1: Testing direct OpenAI client...")
    if test_direct_openai():
        print("✓ Direct connection works! Proceeding with RLM tests...\n")
        
        # Run full RLM test suite
        main()
        
    else:
        print("✗ Direct connection failed. Fix certificate configuration first.")
        print("Check:")
        print("  1. Certificate path is correct")
        print("  2. Certificate file is readable")
        print("  3. API key is valid")
        print("  4. Base URL is correct")
    
    # Or run just quick start:
    # quick_start()







user_query_within_context_prompt = """
You must decide whether the user query is answerable within the application scope.

Context:
- The application overview defines what the system is designed to handle.
- The orchestrator role defines your responsibilities and constraints.

Instructions:
1. Determine if the user query is related to the application domain.
2. If it is clearly within scope, set "is_in_scope" to true.
3. If it is unrelated, vague, or requires capabilities outside the system, set "is_in_scope" to false.
4. Provide a concise reason.

Return STRICT JSON ONLY in this format:
{
  "is_in_scope": true | false,
  "reason": "<short explanation>"
}

Rules:
- Do NOT explain outside JSON.
- Do NOT include markdown.
- Be conservative: if unsure, mark as out of scope.
- Do NOT hallucinate system capabilities.
"""

