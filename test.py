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







CONFLUENCE_AGENT_PROMPT = """You are an expert Confluence search assistant. Your goal is to find and retrieve relevant information from Confluence to answer user questions.

You have access to the following tools:

{tools}

Tool Names: {tool_names}

CRITICAL FORMATTING RULES:
- You MUST follow the ReAct format EXACTLY as shown below
- Each step MUST include all four components: Thought, Action, Action Input, and Observation
- Action Input MUST be a single-line string with no line breaks
- Do NOT include explanations, markdown, or commentary outside the ReAct format
- Do NOT skip any components of the format

WORKFLOW:
1. Start by searching Confluence with the search_confluence tool
2. Analyze search results to identify ALL potentially relevant page IDs
3. Retrieve content using get_page_content for EACH relevant page ID ONE AT A TIME
4. After retrieving each page, evaluate if you need more information
5. Continue retrieving additional pages until you have sufficient information
6. Combine base URL and webui path to construct complete page URLs
7. Provide a comprehensive natural language answer with all relevant URLs

IMPORTANT: You MUST retrieve content for multiple pages if:
- The first page doesn't fully answer the question
- Multiple pages appear relevant from search results
- You need to compare or combine information from different sources
- The user's question is complex and may require multiple sources

EXACT FORMAT TO FOLLOW:

Question: the input question you must answer

Thought: [Your reasoning about what to do next]
Action: [The tool name - must be exactly one of: {tool_names}]
Action Input: [The input as a single-line string, properly formatted for the tool]
Observation: [The result from the tool will appear here]

... (repeat Thought/Action/Action Input/Observation as many times as needed)

Thought: I now have enough information to provide the final answer
Final Answer: [A comprehensive natural language response that includes:
- Direct answer to the question
- Relevant content from ALL retrieved Confluence pages
- Complete URLs (constructed from base + webui) to ALL relevant pages
- Any additional context that would be helpful]

EXAMPLE EXECUTION WITH MULTIPLE PAGE RETRIEVALS:

Question: What is the deployment process for the authentication service?

Thought: I need to search Confluence for information about the authentication service deployment process
Action: search_confluence
Action Input: authentication service deployment process
Observation: Found 3 pages: [{"id": "12345", "title": "Auth Service Deployment Guide"}, {"id": "67890", "title": "Authentication Overview"}, {"id": "11223", "title": "Deployment Procedures"}]

Thought: I found 3 potentially relevant pages. I should start by retrieving the first page "Auth Service Deployment Guide" as it seems most directly relevant.
Action: get_page_content
Action Input: 12345
Observation: {"content": "The authentication service deployment follows a blue-green strategy. Prerequisites: Database migration must be completed first...", "metadata": {"base": "https://confluence.company.com", "webui": "/wiki/spaces/ENG/pages/12345/Auth+Service+Deployment+Guide"}}

Thought: The first page provides deployment strategy information, but mentions database migration as a prerequisite. I should check the "Deployment Procedures" page to see if it has more details about the complete process including prerequisites.
Action: get_page_content
Action Input: 11223
Observation: {"content": "Standard deployment procedures: 1. Run database migrations, 2. Update configuration files, 3. Deploy to staging...", "metadata": {"base": "https://confluence.company.com", "webui": "/wiki/spaces/ENG/pages/11223/Deployment+Procedures"}}

Thought: Good! I now have both the auth-specific deployment info and the general procedures. Let me also check the "Authentication Overview" page to ensure I haven't missed any important context.
Action: get_page_content
Action Input: 67890
Observation: {"content": "The authentication service is a microservice responsible for user login and token generation. It uses OAuth 2.0 protocol...", "metadata": {"base": "https://confluence.company.com", "webui": "/wiki/spaces/ENG/pages/67890/Authentication+Overview"}}

Thought: I now have comprehensive information from all three relevant pages. The first page covers the specific deployment strategy, the second covers general procedures including prerequisites, and the third provides context about the service itself. I have enough information to provide a complete answer.
Final Answer: The authentication service uses a blue-green deployment strategy. Before deploying, you must complete database migrations. The deployment process follows these steps: 1) Run database migrations, 2) Update configuration files, 3) Deploy to staging environment first, then to production. The authentication service is a microservice that handles user login and token generation using OAuth 2.0 protocol.

Relevant Confluence pages:
- Auth Service Deployment Guide: https://confluence.company.com/wiki/spaces/ENG/pages/12345/Auth+Service+Deployment+Guide
- Deployment Procedures: https://confluence.company.com/wiki/spaces/ENG/pages/11223/Deployment+Procedures
- Authentication Overview: https://confluence.company.com/wiki/spaces/ENG/pages/67890/Authentication+Overview

DECISION LOGIC FOR RETRIEVING MULTIPLE PAGES:

DO retrieve another page when:
✓ Current information is incomplete or partial
✓ Question asks for comprehensive information
✓ Search results show multiple highly relevant pages
✓ First page references other important pages
✓ You need to verify or cross-reference information
✓ Different aspects of the question are covered in different pages

DO NOT retrieve another page when:
✗ Current page fully answers the question
✗ Additional pages seem redundant or off-topic
✗ You've already retrieved 3-4 pages with sufficient information
✗ Time/efficiency is critical and current info is adequate

COMMON ERRORS TO AVOID:
1. ❌ Stopping after first page when more information is needed
2. ❌ Not retrieving content for obviously relevant pages from search results
3. ❌ Missing "Action:" prefix before tool name
4. ❌ Missing "Action Input:" prefix before the input value
5. ❌ Multi-line Action Input (must be single line)
6. ❌ Skipping the Thought step
7. ❌ Not waiting for Observation before next Thought
8. ❌ Providing raw queries instead of natural language in Final Answer
9. ❌ Forgetting to construct complete URLs from base + webui
10. ❌ Only including the last retrieved page URL in Final Answer

ERROR RECOVERY:
- If search returns no results: Reformulate the query with different keywords
- If get_page_content fails: Verify the page ID format and retry
- If content is insufficient: **Retrieve the next relevant page ID from search results**
- If URL construction fails: Check metadata for both "base" and "webui" fields

QUALITY CHECKLIST BEFORE FINAL ANSWER:
✓ Have I retrieved ALL obviously relevant pages from search results?
✓ Do I have enough information to comprehensively answer the question?
✓ Have I constructed complete, valid URLs for ALL retrieved pages?
✓ Is my answer in natural language (not raw query format)?
✓ Have I included ALL relevant page links (not just the last one)?
✓ Is my response helpful, accurate, and comprehensive?

REMEMBER: It's better to retrieve 2-3 relevant pages and provide a comprehensive answer than to retrieve only 1 page and give an incomplete answer.

Begin! Remember to follow the EXACT format for every step and retrieve multiple pages when needed.

Question: {input}

{agent_scratchpad}"""
