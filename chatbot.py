from concurrent.futures import ThreadPoolExecutor
import os
import requests
from dotenv import load_dotenv
from datetime import date, datetime
from enum import Enum
from google import genai
from google.genai import types
import multiprocessing
import time
import re
import pathlib
import json
import functools


try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

BASE_DIR = pathlib.Path(__file__).parent / "prompts"

_INITIAL_GEN_TEMPLATE_PATH = BASE_DIR / "generate_initial_prompt.txt"
_COMPARISON_TEMPLATE_PATH = BASE_DIR / "generate_comparison_prompt.txt"
_CRITIQUE_TEMPLATE_PATH = BASE_DIR / "generate_critique_prompt.txt"
_EDIT_TEMPLATE_PATH = BASE_DIR / "generate_edit_prompt.txt"


_INITIAL_GEN_TEMPLATE = _INITIAL_GEN_TEMPLATE_PATH.read_text(encoding="utf-8")
_COMPARISON_TEMPLATE = _COMPARISON_TEMPLATE_PATH.read_text(encoding="utf-8")
_CRITIQUE_TEMPLATE = _CRITIQUE_TEMPLATE_PATH.read_text(encoding="utf-8")
_EDIT_TEMPLATE = _EDIT_TEMPLATE_PATH.read_text(encoding="utf-8")

today = date.today()

LLM_LOG_FILE = "llm_calls.log"

numOfRetries = 0

def retry_on_json_error(max_retries=1000, delay_seconds=1):
    """
    A decorator that retries a function if it returns a dict with a 'raw_response' key,
    which signals a JSON parsing failure.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global numOfRetries
            for attempt in range(max_retries):
                numOfRetries += 1
                result = func(*args, **kwargs)
                # Check for the specific failure signal from the decorated function
                if isinstance(result, dict) and "raw_response" in result:
                    print(f"⚠️ Invalid JSON format detected on attempt {attempt + 1}/{max_retries}. Retrying in {delay_seconds}s...")
                    # Don't wait on the last attempt before raising an error
                    if attempt < max_retries - 1:
                        time.sleep(delay_seconds)
                else:
                    # If the result is NOT the failure signal, it's a success.
                    print("✅ Successfully received valid JSON.")
                    return result
            # If the loop completes, all retries have failed.
            # Raise an exception to stop the program gracefully.
            raise ValueError(
                f"Failed to get valid JSON after {max_retries} attempts. "
                f"Last raw response: {result.get('raw_response', 'N/A')}"
            )
        return wrapper
    return decorator

def log_llm_call(func_name: str, prompt: str, response_text: str, model_name: str, generation_config: types.GenerationConfig):
    """Logs details of an LLM call to a file."""
    timestamp = datetime.now().isoformat()
    
    # Extract relevant parts from generation_config for logging
    gen_config_to_log = {
        "temperature": generation_config.temperature,
        "top_p": generation_config.top_p,
        "top_k": generation_config.top_k,
        "response_mime_type": generation_config.response_mime_type,
        "tools_present": bool(getattr(generation_config, 'tools', False))
    }

    log_entry = (
        f"--- LLM Call Log Entry ---\n"
        f"Timestamp: {timestamp}\n"
        f"Function: {func_name}\n"
        f"Model: {model_name}\n"
        f"Generation Config: {gen_config_to_log}\n"
        f"--- Prompt ---\n{prompt}\n"
        f"--- Response ---\n{response_text}\n"
        f"--------------------------\n\n"
    )
    try:
        with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except IOError as e:
        print(f"Error writing to LLM log file {LLM_LOG_FILE}: {e}")

class Topic(Enum):
    HISTORY = ("History", 1)
    PRODUCTS_INDUSTRY_MARKETSIZE = ("Products, Industry & Market Size", 2)
    REVENUE_BREAKDOWN = ("Revenue Breakdown", 3)
    CUSTOMERS = ("Customers", 4)
    COMPETITIVE_LANDSCAPE = ("Competitive Landscape", 5)
    FINANCIAL_PERFORMANCE = ("Financial Performance", 6)
    STOCK_DRIVERS = ("Stock Drivers", 7)
    INVESTMENT_RISKS = ("Investment Risks", 8)
    def __init__ (self, label: str, code: int):
        self.label = label
        self.code = code

def setup_api():
    load_dotenv()
    open_router_api_key = os.getenv("OPENROUTER_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Please set API key in your .env file")
    return gemini_api_key, open_router_api_key

def generate_initial_prompt(company_name, topic: Topic):
    topic_to_list = {
        1: """Business model evolution, Founding year, location, and founders,
              Early products or services, Major funding rounds or IPO, Acquisitions,
              partnerships, or divestitures, Strategic pivots or rebrandings,
              Recent milestones (new CEO, geographic expansion)""",
        2: '''Core offerings and adjacent R&D projects, Industry classification (e.g., "semiconductors"),
              Total addressable market (TAM) with sources, Segment growth rates, Emerging trends shaping the market''',
        3: '''Revenue per major product or service, Revenue by region (Americas, EMEA, APAC), YoY shifts in those percentages
              Recurring vs. one-time revenue mix, Seasonality or quarter-to-quarter patterns, Effect of recent launches on mix''',
        4: '''Customer segments and distribution channels, Key accounts and their impact, Recent wins or losses
              Satisfaction, retention, and churn metrics, Acquisition cost and lifetime value. Please also identify top (10) customers
              include them in your response''',
        5: '''Direct and indirect competitors, Feature, price, and distribution comparisons, moats or differentiators,
              Competitors' vulnerabilities, Recent competitor moves (M&A, new products), Disruption risks (startups, substitutes).''',
        6: '''Revenue growth trends, Gross and net margins, Cash flow dynamics, Debt ratios''',
        7: '''Upcoming product or roadmap milestones, Macro trends (interest rates, consumer spending), Analyst estimate revisions or consensus targets,
              Catalysts (earnings beats, partnerships), Capital allocation (buybacks, dividends), Regulatory or geopolitical tailwinds''',
        8: '''Competitive pressure or price wars, Supply-chain or cost headwinds, Regulatory, legal, or antitrust scrutiny,
              Currency or geopolitical exposure, Execution risks on new initiatives, Valuation or sentiment shifts'''
    }

    return _INITIAL_GEN_TEMPLATE.format(
        today=today,
        topic_label=topic.label,
        company_name=company_name,
        topic_list=topic_to_list[topic.code]
    )

# Prompt to compare the company with its competitors
def generate_comparison_prompt(company_name):
    return _COMPARISON_TEMPLATE.format(
        company_name=company_name,
    )

def generate_critique_prompt(company_name):
    list_of_topics = ["History", "Products, Industry & Market Size", "Revenue Breakdown", 
    "Customers", "Competitive Landscape", "Financial Performance", "Stock Drivers",
    "Investment Risks"]
    return _CRITIQUE_TEMPLATE.format(
        today=today,
        company_name=company_name,
        list_of_topics=list_of_topics
    )

def generate_edit_prompt(company_name, old_output, corrections_array):
    """
    corrections_array is expected to be a list of dicts:
      [{"original": "...", "corrected": "...", "reasoning": "..."}, ...]
    If a raw_response dict is passed (e.g. {"raw_response": "..."}) we embed that text instead.
    """
    if not corrections_array:
        correction_output_string = "No corrections provided."
        return _EDIT_TEMPLATE.format(today=today,  old_output=old_output, company_name=company_name, correction_output=correction_output_string)

    # If we got the error wrapper
    if isinstance(corrections_array, dict) and "raw_response" in corrections_array:
        correction_output_string = corrections_array["raw_response"]
        return _EDIT_TEMPLATE.format(today = today, old_output=old_output, company_name=company_name, correction_output=correction_output_string)

    # If it's a dict keyed by topics -> lists, try to normalize it (handle both forms)
    if isinstance(corrections_array, dict) and any(isinstance(v, list) for v in corrections_array.values()):
        # flatten if {topic: [corrections]}
        # get first list found
        for v in corrections_array.values():
            if isinstance(v, list):
                corrections_list = v
                break
    else:
        corrections_list = corrections_array

    formatted_corrections = []
    for i, correction in enumerate(corrections_list, 1):
        original_sentence = correction.get('original', 'N/A')
        corrected_sentence = correction.get('corrected', 'N/A')
        reasoning = correction.get('reasoning', 'N/A')

        correction_block = (
            f"Correction #{i}:\n"
            f'  - Original Snippet: "{original_sentence}"\n'
            f'  - Corrected Snippet: "{corrected_sentence}"\n'
            f"  - Reasoning for Change: {reasoning}"
        )
        formatted_corrections.append(correction_block)

    correction_output_string = "\n\n".join(formatted_corrections)
    return _EDIT_TEMPLATE.format(
        today = today,
        old_output=old_output,
        company_name=company_name,
        correction_output=correction_output_string
    )


confNoJsonSearch = types.GenerationConfig(
    temperature=0.2,
)

def generate_response(api_key: str, prompt: str) -> str:
    model_name = "gemini-2.5-flash"
    model = genai.Client()

    # Define the grounding tool
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    # Configure generation settings
    configX = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    attempt = 0
    while True:
        try:
            response = model.models.generate_content(
                model=model_name,
                contents=prompt,
                config=configX,
            )
            response_text = response.text.strip()

            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            # if match:
            #     json_string = match.group(0)
            #     try:
            #         # Attempt to parse the extracted, clean string
            #         critique_json = json.loads(json_string)
            #     except json.JSONDecodeError:
            #         # The extracted string was still not valid JSON
            #         print("⚠️ Could not parse the extracted JSON string.")
            #         critique_json = {"raw_response": response_text}
            # else:
            #     # Could not find anything that looks like a JSON object
            #     print("⚠️ No JSON object found in the response text.")
            #     critique_json = {"raw_response": response_text}

            # Log the LLM call with the config object directly
            log_llm_call("generate_response", prompt, response_text, model_name, configX)
            return response_text
        except Exception as e:
            error_message = str(e)
            if "429" in error_message and "RESOURCE_EXHAUSTED" in error_message:
                delay = 60
                match = re.search(r"'retryDelay':\s*'(\d+)s'", error_message)
                if match:
                    delay = int(match.group(1))
                
                attempt += 1
                print(f"Quota exceeded (429) for {model_name}. API suggests retrying in {delay} seconds. Attempt {attempt}...")
                time.sleep(delay)
            else:
                print(f"An unexpected error occurred during LLM call: {error_message}")
                raise

@retry_on_json_error()
def generate_critique_feedback(api_key, company_name, sorted_results_dict):
    """
    Sends the critique prompt and robustly extracts JSON from the model response.
    `sorted_results_dict` should be a mapping: { "Topic Name": "draft text", ... }
    """
    client = genai.Client(api_key=api_key)           # <-- ensure API key is used
    model_name = "gemini-2.5-flash"

    # Define search grounding tool
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    configX = types.GenerateContentConfig(
        tools=[grounding_tool],
    )

    critique_instructions = generate_critique_prompt(company_name)

    # Build prompt
    full_prompt_for_critique = critique_instructions + "\n\n"
    for topic_name, response_content in sorted_results_dict.items():
        full_prompt_for_critique += (
            f"--- TOPIC: {topic_name} ---\n"
            f"{response_content.strip()}\n\n"
        )

    full_prompt_for_critique += (
        "Return your response strictly as valid JSON following the output format described above. "
        "Do not add any commentary, markdown, or notes."
    )

    attempt = 0
    while True:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt_for_critique,
                config=configX
            )

            # Try safe extraction of text (SDKs return different shapes)
            response_text = None
            if hasattr(response, "text") and response.text:
                response_text = response.text
            else:
                # fallback: try to find nested structure (adjust if your SDK shape differs)
                try:
                    # many SDKs put model output in response.output[0].content[0].text
                    response_text = response.output[0].content[0].text
                except Exception:
                    response_text = repr(response)

            response_text = response_text.strip()

            # Remove markdown code fences if present (```json ... ```)
            # Also remove a single leading "```json" or "```" and trailing "```"
            response_text = re.sub(r"^```(?:json)?\s*", "", response_text, flags=re.IGNORECASE)
            response_text = re.sub(r"\s*```$", "", response_text)

            # Extract the first JSON object/braced block if there is extra text
            match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            if match:
                json_string = match.group(1)
                try:
                    critique_json = json.loads(json_string)
                except json.JSONDecodeError:
                    # Last-ditch attempts to "fix" common issues:
                    # - replace smart-quotes / single quotes -> double quotes
                    safe = json_string.replace("“", "\"").replace("”", "\"").replace("'", "\"")
                    # remove trailing commas like `,]` or `,}`
                    safe = re.sub(r",\s*(\]|\})", r"\1", safe)
                    try:
                        critique_json = json.loads(safe)
                    except Exception:
                        print("⚠️ Could not parse the extracted JSON string after cleanup.")
                        critique_json = {"raw_response": response_text}
            else:
                # No braced JSON found — return raw_response for debugging
                print("⚠️ No JSON object found in the response text.")
                critique_json = {"raw_response": response_text}

            log_llm_call(
                "generate_critique_feedback",
                full_prompt_for_critique,
                response_text,
                model_name,
                configX
            )
            return critique_json

        except Exception as e:
            error_message = str(e)
            if "429" in error_message and "RESOURCE_EXHAUSTED" in error_message:
                delay = 60
                match = re.search(r"'retryDelay':\s*'(\d+)s'", error_message)
                if match:
                    delay = int(match.group(1))
                attempt += 1
                print(f"Quota exceeded (429). Retrying in {delay} seconds. Attempt {attempt}...")
                time.sleep(delay)
            else:
                print(f"❌ Unexpected error in critique call: {error_message}")
                raise


def apply_ansi_formatting(text: str) -> str:
    """
    Splits text by newline characters and applies ANSI escape codes for bold to text
    enclosed in double asterisks (**) and changes "Summary of Key Takeaways:" to bold.
    """
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    lines = text.splitlines()
    formatted_lines = []
    for line in lines:
        if "Summary of Key Takeaways:" in line:
            line = line.replace("Summary of Key Takeaways:", f"{BOLD}Summary of Key Takeaways:{RESET}")
        
        def replace_bold_markers(match):
            return f"{BOLD}{match.group(1)}{RESET}"
        
        line = re.sub(r'\*\*(.*?)\*\*', replace_bold_markers, line)
        formatted_lines.append(line)
        
    return "\n".join(formatted_lines)

# def analyze_single_topic(company_name: str, topic: Topic, gemini_api_key: str) -> tuple[str, str]:
#     """
#     Generates, critiques, and refines the analysis for a single topic.
#     Returns a tuple containing the topic label and the final refined analysis string.
#     """
#     # 1. Generate rough draft
#     prompt = generate_topic_prompt(company_name, topic)
#     current_analysis = generate_response(gemini_api_key, prompt)

#     # It's also good practice to check if the initial analysis is valid
#     if not current_analysis:
#         print(f"Warning: Initial analysis generation for '{topic.label}' failed. Skipping topic.")
#         return (topic.label, "Analysis could not be generated for this topic.")
    
#     # 2. Critique and Redraft loop
#     amount_of_cycles = 3
#     count = 0
#     while count < amount_of_cycles:
#         correction_output = generate_critique_feedback(gemini_api_key, company_name, current_analysis, prompt)
        
#         # <<< CHANGE: Implement the "retry once" logic >>>
#         # If the first attempt fails, try one more time.
#         if correction_output is None:
#             print(f"Warning: Critique generation for '{topic.label}' returned no output. Retrying once...")
#             correction_output = generate_critique_feedback(gemini_api_key, company_name, current_analysis, prompt)

#         # Now, check again. If it's still None after the retry, then break.
#         if correction_output is None:
#             print(f"Warning: Critique generation for '{topic.label}' failed after a retry. Using the last available analysis.")
#             break # Exit the critique loop and proceed with the current analysis
#         # <<< END OF CHANGE >>>

#         if correction_output.strip().upper() == "ALL GOOD":
#             break
#         else:
#             edit_prompt = generate_edit_prompt(company_name, current_analysis, correction_output)
#             current_analysis = generate_response(gemini_api_key, edit_prompt)
#             # Add a check here as well in case the edit response fails
#             if not current_analysis:
#                  print(f"Warning: Edit generation for '{topic.label}' failed. Using the previous version.")
#                  break
#             count += 1
            
#     return (topic.label, current_analysis)

def initial_gen (company_name: str, topic: Topic, gemini_api_key: str):
    prompt = generate_initial_prompt(company_name, topic)
    return generate_response(gemini_api_key, prompt)

def analyze_company(gemini_api_key: str, open_router_api_key: str):
    print("Welcome to Company Analysis Bot!")
    print("This bot will provide a comprehensive analysis of any company.\n")
    while True:
        company_name = input("Enter company name (or 'exit' to quit): ")
        if company_name.lower() in ['exit', 'quit']:
            print("\nGoodbye!")
            break
        
        start_time = time.perf_counter()
        
        print(f"\nGenerating a comprehensive analysis for {company_name}. This may take a few minutes...")
        try:
            topics = list(Topic)

            amount_of_cycles = 3

            count = 0
            sorted_results = {}
            ordered_futures = []
            with ThreadPoolExecutor(max_workers=8) as exe:
                for topic in topics:
                    # print(topic);
                    # print("____")
                    future = exe.submit(initial_gen, company_name, topic, gemini_api_key)
                    ordered_futures.append((topic.label, future))
                for code, future in sorted(ordered_futures):
                    sorted_results[code] = future.result()

            while (count < amount_of_cycles):
#                 sorted_results ={"Investment Risks": '''NVIDIA faces several investment risks despite its dominant position in the AI chip market. Geopolitical tensions, particularly between the U.S. and China, pose a significant headwind, evidenced by the $4.5 billion charge incurred in Q1 FY2026 due to excess H20 inventory following new U.S. export licensing requirements for China, which also impacted the GAAP gross margin to 60.5% (non-GAAP 61.0%). NVIDIA estimates a further $8.0 billion loss in H20 revenue for Q2 FY2026, for which it has guided revenue of $45.0 billion, plus or minus 2%. Regulatory scrutiny is intensifying, with antitrust investigations by the French Competition Authority and the U.S. Department of Justice examining NVIDIA's market dominance and proprietary CUDA software, alongside China's recent inquiries into potential "backdoors" in H20 chips. Competitive pressures are rising as rivals like Amazon offer aggressively priced AI chips, and major cloud providers develop in-house alternatives, potentially eroding NVIDIA's market share and affecting its high gross margins, which were 71.3% non-GAAP excluding the H20 charge in Q1 FY2026. Furthermore, NVIDIA's premium valuation, with a P/E ratio of 60.1, reflects high growth expectations, making it susceptible to significant corrections if growth decelerates or market sentiment shifts.

# Summary of Key Takeaways:
# *   Geopolitical tensions, particularly U.S.-China export restrictions, directly impact NVIDIA's revenue and gross margins, as seen with the $4.5 billion Q1 FY2026 charge related to H20 chips and the projected $8.0 billion revenue loss in Q2 FY2026 from China.
# *   Increased regulatory and antitrust scrutiny from multiple jurisdictions (U.S., France, China) presents legal and operational risks due to NVIDIA's market dominance and proprietary technology.
# *   Competitive pressures from existing rivals and large tech companies developing in-house AI chips could lead to price wars and market share erosion, potentially impacting profitability.
# *   NVIDIA's high valuation multiples indicate that its stock is priced for continued robust growth, making it vulnerable to market corrections if growth expectations are not met.'''}

                critique_feedback = generate_critique_feedback(gemini_api_key, company_name, sorted_results)

                # critique_feedback = {'Investment Risks': [{'original': 'NVIDIA faces several investment risks despite its dominant position in the AI chip market. Geopolitical tensions, particularly between the U.S. and China, pose a significant headwind, evidenced by the $4.5 billion charge incurred in Q1 FY2026 due to excess H20 inventory following new U.S. export licensing requirements for China, which also impacted the GAAP gross margin to 60.5% (non-GAAP 61.0%).', 'corrected': 'NVIDIA faces several investment risks despite its dominant position in the AI chip market. Geopolitical tensions, particularly between the U.S. and China, pose a significant headwind, evidenced by the $4.5 billion charge incurred in Q1 FY2026 (ended April 27, 2025) due to excess H20 inventory and purchase obligations following new U.S. export licensing requirements for China, which also impacted the GAAP gross margin to 60.5% (non-GAAP 61.0%). [11, 21, 26, 28]', 'reasoning': "Added the end date for Q1 FY2026 for more precise context and added 'purchase obligations' to accurately reflect the nature of the charge as reported by NVIDIA. Added citations."}, {'original': 'NVIDIA estimates a further $8.0 billion loss in H20 revenue for Q2 FY2026, for which it has guided revenue of $45.0 billion, plus or minus 2%.', 'corrected': 'NVIDIA estimates a further $8.0 billion loss in H20 revenue for Q2 FY2026, for which it has guided revenue of $45.0 billion, plus or minus 2%. [11, 17, 22, 26, 28]', 'reasoning': 'Added citations to verify the revenue loss estimate and guidance.'}, {'original': 'Regulatory scrutiny is intensifying, with antitrust investigations by the French Competition Authority and the U.S. Department of Justice examining NVIDIA\'s market dominance and proprietary CUDA software, alongside China\'s recent inquiries into potential "backdoors" in H20 chips.', 'corrected': 'Regulatory scrutiny is intensifying, with active antitrust investigations by the French Competition Authority and the U.S. Department of Justice examining NVIDIA\'s market dominance and proprietary CUDA software, alongside China\'s recent inquiries (as of late July/early August 2025) into potential "backdoors" in H20 chips. [3, 5, 7, 8, 10, 12, 13, 16, 20, 24]', 'reasoning': "Added 'active' for clarity on the ongoing nature of investigations and specified the timeframe for China's inquiries for more precision. Added citations."}, {'original': "Competitive pressures are rising as rivals like Amazon offer aggressively priced AI chips, and major cloud providers develop in-house alternatives, potentially eroding NVIDIA's market share and affecting its high gross margins, which were 71.3% non-GAAP excluding the H20 charge in Q1 FY2026.", 'corrected': "Competitive pressures are rising as rivals like Amazon offer aggressively priced AI chips (e.g., Inferentia and Trainium), and major cloud providers (e.g., Google, Microsoft, Amazon) develop in-house alternatives (e.g., Google TPUs, Microsoft Azure Maia, Amazon Trainium), potentially eroding NVIDIA's market share and affecting its high gross margins, which were 71.3% non-GAAP excluding the H20 charge in Q1 FY2026. [11, 18, 19, 21, 25, 27, 28, 29, 30, 31, 33, 34]", 'reasoning': 'Provided specific examples of aggressively priced chips (Inferentia, Trainium) and in-house alternatives (Google TPUs, Microsoft Azure Maia, Amazon Trainium) for better clarity and context. Added citations.'}, {'original': "Furthermore, NVIDIA's premium valuation, with a P/E ratio of 60.1, reflects high growth expectations, making it susceptible to significant corrections if growth decelerates or market sentiment shifts.", 'corrected': "Furthermore, NVIDIA's premium valuation, with a trailing twelve-month (TTM) P/E ratio of approximately 57.6 as of August 8, 2025, reflects high growth expectations, making it susceptible to significant corrections if growth decelerates or market sentiment shifts. [2, 4, 6, 9]", 'reasoning': 'Corrected the P/E ratio to reflect the most current and accurate TTM P/E (approx. 57.6) as of the specified date (August 8, 2025) and clarified it as TTM, enhancing precision. Added citations.'}]}

                # print(critique_feedback);

                edit_futures = {}

                with ThreadPoolExecutor(max_workers=8) as exe:
                    for topic_name, corrections_array in critique_feedback.items():
                        original_draft = sorted_results.get(topic_name)

                        if original_draft and isinstance(corrections_array, list):
                            edit_prompt = generate_edit_prompt(company_name, original_draft, corrections_array)
                            # Submit the job and store the future object
                            future = exe.submit(generate_response, gemini_api_key, edit_prompt)
                            edit_futures[topic_name] = future
                        else:
                             print(f"Skipping edit for '{topic_name}' due to invalid data.")
#                         new_draft = """NVIDIA faces several investment risks despite its dominant position in the AI chip market. Geopolitical tensions, particularly between the U.S. and China, pose a significant headwind, evidenced by the $4.5 billion charge incurred in Q1 FY2026 (ended April 27, 2025) due to excess H20 inventory and purchase obligations following new U.S. export licensing requirements for China, which also impacted the GAAP gross margin to 60.5% (non-GAAP 61.0%). NVIDIA estimates a further $8.0 billion loss in H20 revenue for Q2 FY2026, for which it has guided revenue of $45.0 billion, plus or minus 2%. Regulatory scrutiny is intensifying, with active antitrust investigations by the French Competition Authority and the U.S. Department of Justice examining NVIDIA's market dominance and proprietary CUDA software, alongside China's recent inquiries (as of late July/early August 2025) into potential "backdoors" in H20 chips. Competitive pressures are rising as rivals like Amazon offer aggressively priced AI chips (e.g., Inferentia and Trainium), and major cloud providers (e.g., Google, Microsoft, Amazon) develop in-house alternatives (e.g., Google TPUs, Microsoft Azure Maia, Amazon Trainium), potentially eroding NVIDIA's market share and affecting its high gross margins, which were 71.3% non-GAAP excluding the H20 charge in Q1 FY2026. Furthermore, NVIDIA's premium valuation, with a trailing twelve-month (TTM) P/E ratio of approximately 57.6 as of August 8, 2025, reflects high growth expectations, making it susceptible to significant corrections if growth decelerates or market sentiment shifts.

# Summary of Key Takeaways:
# *   Geopolitical tensions, particularly U.S.-China export restrictions, directly impact NVIDIA's revenue and gross margins, as seen with the $4.5 billion Q1 FY2026 charge (ended April 27, 2025) related to H20 chips and purchase obligations, and the projected $8.0 billion revenue loss in Q2 FY2026 from China.
# *   Increased regulatory and antitrust scrutiny from multiple jurisdictions (U.S., France, China) presents legal and operational risks due to NVIDIA's market dominance and proprietary technology, including recent inquiries into potential "backdoors" in H20 chips.
# *   Competitive pressures from existing rivals offering aggressively priced AI chips (e.g., Amazon Inferentia and Trainium) and large tech companies developing in-house AI chips (e.g., Google TPUs, Microsoft Azure Maia, Amazon Trainium) could lead to price wars and market share erosion, potentially impacting profitability.
# *   NVIDIA's high valuation, with a trailing twelve-month (TTM) P/E ratio of approximately 57.6 as of August 8, 2025, indicates that its stock is priced for continued robust growth, making it vulnerable to market corrections if growth expectations are not met.
# """

                for topic_name, future in edit_futures.items():
                    new_draft = future.result()
                    sorted_results[topic_name] = new_draft
                    # print(f"-> Edits for '{topic_name}' completed.")
                    


                count = count + 1

            

            

            # print(f"\n--- Final Comprehensive Analysis for {company_name} ---")
            for topic in Topic:
                topic_label = topic.label
                # Look up the analysis text from the dictionary using the correct label.
                analysis_text = sorted_results.get(topic_label)

                # Print a clear, differentiating header for each topic
                print(f"\n\n{'='*70}")
                print(f"TOPIC: {topic_label.upper()}")
                print(f"{'='*70}\n")
                
                if analysis_text:
                    # Apply formatting and print the analysis for the topic
                    formatted_text = apply_ansi_formatting(analysis_text)
                    print(formatted_text)
                else:
                    # This is a fallback in case a result for a topic was never generated.
                    print("Analysis for this topic could not be found.")


            end_time = time.perf_counter()
            duration_seconds = end_time - start_time
            minutes, seconds = divmod(duration_seconds, 60)
            print(f"\n{'='*70}")
            if minutes >= 1:
                print(f"Total Time: {int(minutes)} minutes and {seconds:.2f} seconds.")
            else:
                print(f"Total Time: {duration_seconds:.2f} seconds.")
            print(f"{'='*70}")

            print(f"\nRetries: {numOfRetries}")


        except Exception as e:
            print(f"Error during company analysis: {str(e)}")

def main():
    try:
        gemini_api_key, open_router_api_key = setup_api()
        analyze_company(gemini_api_key, open_router_api_key)
    except Exception as e:
        print(f"Fatal Error: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    multiprocessing.freeze_support()
    exit(main());