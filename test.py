"""
RLM Setup for Custom Meta-Llama Endpoint with Certificate Authentication
Complete working example for your specific configuration
"""

from rlm import RLM
from rlm.logger import RLMLogger
import httpx
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION - Update these with your actual values
# ============================================================================

API_KEY = "eyJraWQiOiIxZTU5Zf..."  # Your full API key
BASE_URL = "https://genfactory.analytics.echonet/genai/api/v2"
CERT_PATH = r"C:/ai_agents/Router/shared/certificate/bundle.pem"
MODEL_NAME = "Meta-Llama-33-70B-Instruct"
TEMPERATURE = 0.2

# ============================================================================
# METHOD 1: Direct RLM Configuration (Try this first)
# ============================================================================

def setup_rlm_method1():
    """
    Direct configuration of RLM with custom HTTP client
    """
    print("Setting up RLM - Method 1: Direct Configuration\n")
    
    # Create HTTP client with certificate
    http_client = httpx.Client(verify=CERT_PATH)
    
    # Initialize RLM with your custom endpoint
    rlm = RLM(
        backend="openai",  # Your endpoint is OpenAI-compatible
        backend_kwargs={
            "model_name": MODEL_NAME,
            "api_key": API_KEY,
            "base_url": BASE_URL,
            "http_client": http_client,
            "temperature": TEMPERATURE
        },
        environment="local",  # Use local REPL
        verbose=True  # See what RLM is doing
    )
    
    return rlm


# ============================================================================
# METHOD 2: With Logging and Monitoring
# ============================================================================

def setup_rlm_with_logging():
    """
    RLM setup with logging to track recursive calls
    """
    print("Setting up RLM - Method 2: With Logging\n")
    
    # Create log directory
    log_dir = Path("./rlm_logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set up logger
    logger = RLMLogger(log_dir=str(log_dir))
    
    # Create HTTP client with certificate
    http_client = httpx.Client(verify=CERT_PATH)
    
    # Initialize RLM
    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": MODEL_NAME,
            "api_key": API_KEY,
            "base_url": BASE_URL,
            "http_client": http_client,
            "temperature": TEMPERATURE
        },
        environment="local",
        logger=logger,
        verbose=True
    )
    
    print(f"Logging to: {log_dir.absolute()}\n")
    return rlm


# ============================================================================
# METHOD 3: Using Pre-configured OpenAI Client
# ============================================================================

def setup_rlm_method3():
    """
    If RLM doesn't accept http_client in backend_kwargs,
    try this approach with a pre-configured client
    """
    from openai import OpenAI
    
    print("Setting up RLM - Method 3: Pre-configured Client\n")
    
    # Create custom OpenAI client
    custom_client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=httpx.Client(verify=CERT_PATH)
    )
    
    # Try passing the client directly (may need RLM source modification)
    try:
        rlm = RLM(
            backend="openai",
            backend_kwargs={
                "model_name": MODEL_NAME,
                "client": custom_client,  # Pass pre-configured client
                "temperature": TEMPERATURE
            },
            environment="local",
            verbose=True
        )
        return rlm
    except Exception as e:
        print(f"Method 3 failed: {e}")
        print("You may need to modify RLM source code to support this.\n")
        return None


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_basic_completion(rlm):
    """Test basic RLM completion"""
    print("=" * 70)
    print("TEST 1: Basic Completion")
    print("=" * 70)
    
    query = "Explain what a recursive language model is in one sentence."
    print(f"Query: {query}\n")
    
    result = rlm.completion(query)
    print(f"Response: {result.response}\n")
    print(f"Tokens used: {result.usage}\n")


def test_long_context(rlm):
    """Test RLM with long context processing"""
    print("=" * 70)
    print("TEST 2: Long Context Processing")
    print("=" * 70)
    
    # Simulate a long document
    long_document = """
    Artificial Intelligence (AI) has revolutionized numerous industries.
    Machine learning, a subset of AI, enables computers to learn from data.
    Deep learning uses neural networks with multiple layers.
    Natural Language Processing (NLP) helps computers understand human language.
    Computer vision allows machines to interpret visual information.
    """ * 50  # Repeat to create a longer document
    
    query = "What are the main AI technologies mentioned in this document?"
    print(f"Query: {query}")
    print(f"Context length: {len(long_document)} characters\n")
    
    result = rlm.completion(query=query, context=long_document)
    print(f"Response: {result.response}\n")


def test_code_generation(rlm):
    """Test RLM's ability to generate and execute code"""
    print("=" * 70)
    print("TEST 3: Code Generation")
    print("=" * 70)
    
    query = "Generate Python code to calculate the first 10 Fibonacci numbers."
    print(f"Query: {query}\n")
    
    result = rlm.completion(query)
    print(f"Response: {result.response}\n")


def test_recursive_processing(rlm):
    """Test RLM's recursive capabilities"""
    print("=" * 70)
    print("TEST 4: Recursive Processing")
    print("=" * 70)
    
    # Create a scenario that benefits from recursion
    query = "List the first 50 prime numbers, showing your work."
    print(f"Query: {query}\n")
    
    result = rlm.completion(query)
    print(f"Response: {result.response}\n")


# ============================================================================
# VERIFICATION - Test without RLM first
# ============================================================================

def verify_endpoint_works():
    """
    Verify your endpoint works with direct OpenAI client before using RLM
    """
    from openai import OpenAI
    
    print("=" * 70)
    print("VERIFICATION: Testing Direct OpenAI Client")
    print("=" * 70)
    
    try:
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            http_client=httpx.Client(verify=CERT_PATH)
        )
        
        print("Sending test request...")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say 'Hello from Llama!'"}],
            temperature=TEMPERATURE
        )
        
        print(f"✓ Success! Response: {response.choices[0].message.content}\n")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        print("Fix this issue before using RLM.\n")
        return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - run all tests
    """
    print("\n" + "=" * 70)
    print("RLM SETUP FOR CUSTOM META-LLAMA ENDPOINT")
    print("=" * 70 + "\n")
    
    # Step 1: Verify endpoint works
    if not verify_endpoint_works():
        print("Please fix endpoint configuration before proceeding.")
        return
    
    # Step 2: Try to set up RLM (try Method 1 first)
    try:
        print("Attempting Method 1...")
        rlm = setup_rlm_method1()
    except Exception as e:
        print(f"Method 1 failed: {e}")
        try:
            print("\nAttempting Method 2...")
            rlm = setup_rlm_with_logging()
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            print("\nTrying Method 3...")
            rlm = setup_rlm_method3()
            if rlm is None:
                print("All methods failed. See troubleshooting guide.")
                return
    
    # Step 3: Run tests
    print("\nRLM successfully initialized! Running tests...\n")
    
    try:
        test_basic_completion(rlm)
        test_code_generation(rlm)
        test_long_context(rlm)
        test_recursive_processing(rlm)
        
        print("=" * 70)
        print("ALL TESTS COMPLETED!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("See error above for details.\n")


# ============================================================================
# QUICK START EXAMPLES
# ============================================================================

def quick_start_example():
    """
    Minimal example to get started quickly
    """
    # Create RLM instance
    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": "Meta-Llama-33-70B-Instruct",
            "api_key": "eyJraWQiOiIxZTU5Zf...",  # Your key
            "base_url": "https://genfactory.analytics.echonet/genai/api/v2",
            "http_client": httpx.Client(
                verify=r"C:/ai_agents/Router/shared/certificate/bundle.pem"
            ),
            "temperature": 0.2
        },
        verbose=True
    )
    
    # Use it!
    result = rlm.completion("What is machine learning?")
    print(result.response)


if __name__ == "__main__":
    # Run full test suite
    main()
    
    # Or run just the quick start example:
    # quick_start_example()
