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

# IMPORTANT: Set the start method for multiprocessing to 'spawn'
# This must be called at the very beginning of the main execution block
# to ensure a clean slate for child processes.
# It helps avoid pickling issues with libraries that create unpicklable global state.
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # This can happen if set_start_method is called more than once
    # or if it's already implicitly set in some environments.
    # 'force=True' helps, but catching the error makes it robust.
    pass


# Define the log file for LLM calls
LLM_LOG_FILE = "llm_calls.log"

def log_llm_call(func_name: str, prompt: str, response_text: str, model_name: str, generation_config: types.GenerationConfig):
    """Logs details of an LLM call to a file."""
    timestamp = datetime.now().isoformat()
    
    # Extract relevant parts from generation_config for logging
    gen_config_to_log = {
        "temperature": generation_config.temperature,
        "top_p": generation_config.top_p,
        "top_k": generation_config.top_k,
        "response_mime_type": generation_config.response_mime_type,
        "tools_present": bool(getattr(generation_config, 'tools', False)) # Check if tools are configured, safely
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


# Enum for the 8 topics that the analysis should be about
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

# Accessing Gemini and OpenRouter API Key
def setup_api():
    load_dotenv()
    open_router_api_key = os.getenv("OPENROUTER_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Please set API key in your .env file")
    return gemini_api_key, open_router_api_key


# --- REMOVED: grounding_tool_retrieval definition ---

# Define the configuration objects globally without any search tools
confJson = types.GenerationConfig(
    response_mime_type="application/json",
)
confJsonCreative = types.GenerationConfig(
    response_mime_type="application/json",
    temperature = 2,
    top_p = 0.98,
    top_k = 1000,
)

# Configuration for non-JSON response without search, keeping your desired temperature
confNoJsonSearch = types.GenerationConfig(
    temperature=0.2,
    # Tools parameter is removed here
)
# Configuration for JSON response without search, keeping your desired temperature
confJsonSearch = types.GenerationConfig(
    response_mime_type="application/json",
    temperature=0.2,
    # Tools parameter is removed here
)


# Prompt to generate for the LLM given the company and the topic. Will probably be looped through 8 times
# once for each of the topic.
def generate_topic_prompt(company_name, topic):
    today = date.today()
    # create a dictionary, mapping topic to the list of topic examples to help the LLM.
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
              Satisfaction, retention, and churn metrics, Acquisition cost and lifetime value''',
        5: '''Direct and indirect competitors, Feature, price, and distribution comparisons, moats or differentiators,
              Competitors' vulnerabilities, Recent competitor moves (M&A, new products), Disruption risks (startups, substitutes)''',
        6: '''Revenue growth trends, Gross and net margins, Cash flow dynamics, Debt ratios''',
        7: '''Upcoming product or roadmap milestones, Macro trends (interest rates, consumer spending), Analyst estimate revisions or consensus targets,
              Catalysts (earnings beats, partnerships), Capital allocation (buybacks, dividends), Regulatory or geopolitical tailwinds''',
        8: '''Competitive pressure or price wars, Supply-chain or cost headwinds, Regulatory, legal, or antitrust scrutiny,
              Currency or geopolitical exposure, Execution risks on new initiatives, Valuation or sentiment shifts'''
    }

    # IMPORTANT: Instructions to use search tool are KEPT as requested,
    # even though the actual tool is removed from GenerationConfig.
    prompt_string = f"""
        You are a knowledgeable financial senior analyst with expertise in company analysis.
        Today is {today}. **You must use your search tool to find the most up-to-date and verifiable information available.**
        
        **Your Task:** Write a cohesive paragraph in full sentences that is ~250-500 words about the **{topic.label}** of {company_name}.
        **Focus exclusively on this topic.** Do not include any headers or titles in your response.

        When discussing financial performance (e.g., revenue, earnings, margins), always cite **actual reported figures** from the latest earnings reports and the most current **analyst consensus or company guidance** for future periods, clearly distinguishing between them.
        Use your search tool to find the latest data.
        (Some topics or ideas you can talk about, but are not limited to are: {topic_to_list[topic.code]}).
        While writing this analysis, use financial terms precisely and provide valuation context.
        
        Lastly, end your response with a brief bullet-pointed summary titled 'Summary of Key Takeaways:'. This summary should extract the main points you brought up in the paragraph.
        Don't add any extra follow up sentences after the summary.

        Here is an example of a great response on the key stock drivers of Amazon on 6/22/2025:

            Over the next twelve months, Amazon's share performance will be underpinned primarily by its
            ongoing transformation into an an AI-first enterprise, with Amazon Web Services (AWS) at the vanguard
            of both top-line growth and margin expansion. AWS is set to introduce a suite of next-generation offerings—ranging
            from Aurora PostgreSQL version 17 and enhanced Bedrock foundation models to its quantum-compute "Ocelot" chip—that
            should sustain its industry-leading ~33% operating margin and drive low-teens revenue growth through mid-2026.
            Concurrently, Amazon's retail business is embedding advanced machine-learning algorithms across warehousing,
            last-mile delivery, and inventory forecasting, which management expects will compress fulfillment costs by
            approximately 50-75 basis points in FY 2026, helping offset softening unit demand in mature markets. In entertainment,
            Prime Video's rollout of AI-powered dubbing and an expanded Upfront 2025 lineup will bolster ad-supported streaming
            revenue and improve incremental ARPU per member.

            Macroeconomic conditions remain broadly favorable: with the Federal Reserve signifying a pause in rate hikes and the
            U.S. consumer savings rate hovering near its decade average, discretionary spending should continue to support Amazon's
            retail GMV. That said, geopolitical risks—particularly imported-goods tariffs—and intensifying antitrust scrutiny in
            both North America and the EU represent potential headwinds to margin trajectory. On the sell-side, 47 analysts peg AMZN's
            twelve-month consensus price target at roughly $246, implying ~17% upside from current levels, with EPS forecasts of $6.30
            in FY 2025 (up ~14% YoY) driving a target P/E of ~39x.

            Finally, although capital expenditures surged past $100 billion in 2024 for AI and distribution capacity, management has
            signaled more disciplined spending in 2025-2026, which could free up incremental free cash flow for continued share repurchases.
            This disciplined capital allocation, combined with robust free cash flow conversion (above 20% of revenue), should support a
            rising return on invested capital and narrow the valuation discount to peers—even absent a formal dividend policy.

            Summary of Key Takeaways:
            **AWS AI & Cloud Innovation:** New AI-driven services and Aurora v17 to sustain low-teens revenue growth and ~33% margins
            **Retail Efficiency Gains:** Machine-learning in logistics to compress fulfillment costs by 50-75 bps in FY 2026
            **Prime Video & Advertising:** AI dubbing and Upfront 2025 to enhance ARPU and ad revenues
            **Macro & Geopolitical:** Fed pause and resilient consumer spending vs. tariff and antitrust risks
            **Analyst Consensus:** ~$246 price target; FY 2025 EPS $6.30 (+14%), implying ~39x forward P/E
            **Capital Allocation:** Evolving capex discipline to bolster free cash flow and fund share buybacks


        Here is an example of a great response on the key investment risks of Amazon on 6/22/2025:

            Amazon's investment profile, though underpinned by robust scale and diversified operations,
            is shadowed by several material risks that could erode shareholder value. In the quarter ended March 31, 2025,
            Amazon reported net sales of $155.7 billion—up 9% year-over-year—but this figure masks a $1.4 billion unfavorable impact
            from foreign-exchange fluctuations, highlighting its sensitivity to currency volatility across Europe, Latin America, and Asia.
            Supply-chain and cost headwinds remain acute: on its May 1, 2025 earnings call, management cautioned that elevated labor,
            logistics, and infrastructure expenses, compounded by potential U.S.-China tariff increases up to 145% on consumer goods,
            could compress retail and marketplace margins throughout 2025.

            Competitive pressures are intensifying across both e-commerce and cloud segments.
            In online retail, players such as Walmart, Alibaba, and fast-fashion newcomers like Temu continue to undercut pricing
            and encroach on market share, while in cloud computing, Microsoft Azure—whose server products and cloud services
            revenue grew 30% in the quarter ended June 30, 2024—and Google Cloud—up 30% in revenue in Q4 2024—has narrowed AWS's
            dominance and pressure AWS to sustain above-market growth rates.

            Regulatory and legal scrutiny further darkens the outlook. A Federal Trade Commission lawsuit alleging the use of
            "dark patterns" to trick consumers into auto-renewing Prime memberships is set for a bench trial in June 2025,
            and a broader antitrust suit accusing Amazon of abusing its marketplace monopoly will not reach trial until October 2026—risks
            that could result in injunctive relief, fines, or mandated business-model changes.

            Execution risks on capital-intensive initiatives, including the satellite-broadband Project Kuiper rollout and multi-billion-dollar
            AWS AI infrastructure investments (such as Trainium 2 chips and Bedrock model expansion), may strain free cash flow if adoption lags expectations.
            Finally, Amazon's valuation remains at a premium—trading near 34x trailing twelve-month earnings as of June 20,
            2025—making the stock particularly vulnerable to shifts in investor sentiment if revenue growth or operating-income guidance
            disappoints, as evidenced by a 3% share drop following the cautious Q2 outlook issued on May 1, 2025.

            Summary of Key Takeaways:
            **Currency & Geopolitical Exposure:** $1.4 billion FX headwind in Q1 2025; potential U.S.-China tariffs up to 145%.
            **Cost & Supply-Chain Pressures:** Elevated labor, logistics, and infrastructure expenses squeezing margins.
            **Competitive Intensity:** E-commerce undercut by Walmart, Alibaba, Temu; AWS challenged by Azure and Google Cloud.
            **Regulatory & Legal Scrutiny:** FTC "dark patterns" trial in June 2025; broader antitrust suit trial in October 2026.
            **Execution Risks:** High-cost projects (Project Kuiper, AI infrastructure) may under-deliver on returns.
            **Valuation Sensitivity:** Trading at ~34x TTM earnings (as of June 20, 2025), shares fell 3% on cautious Q2 guidance.

    """
    return prompt_string

# Prompt to compare the company with its competitors
def generate_comparison_prompt(company_name):
    return f"""
    You are a financial analyst comparing companies. Perform a detailed comparison between {company_name} and its 2-3 closest competitors.

    Include:
    1. Market position comparison (market share, growth rates)
    2. Financial metrics (P/E ratio, revenue growth, profit margins)
    3. Competitive advantages/differentiators
    4. Recent strategic moves (acquisitions, partnerships)
    5. Valuation comparison (forward P/E, EV/EBITDA)

    Present the analysis in clear sections with subheadings.
    Conclude with a summary table comparing key metrics.

    Example format for Apple:

    ### Competitive Landscape: Apple vs. Major Peers

    **Market Position:**
    - Apple: 55% smartphone market share in US (IDC Q2 2025)
    - Samsung: 28% market share, stronger in Android segment
    - Google Pixel: 12% share, growing in AI features

    **Financial Comparison (TTM):**
    | Metric       | Apple | Samsung | Google |
    |--------------|-------|---------|--------|
    | P/E Ratio    | 28.5  | 12.3    | 24.1   |
    | Revenue Growth | 7.2% | 3.8%    | 9.1%   |

    **Key Differentiators:**
    - Apple's ecosystem lock-in vs. Samsung's hardware variety
    - Google's AI-first approach...
    """

def generate_critique_prompt(company_name):
    """
    Generates a more demanding and specific critique prompt for the LLM.
    """
    today = date.today()
    # IMPORTANT: Instructions to use search tool are KEPT as requested,
    # even though the actual tool is removed from GenerationConfig.
    return f"""
        You are an exceptionally meticulous and skeptical senior equity research analyst. Your primary role is to fact-check and enhance a draft analysis of {company_name}. Today is {today}.
        You will be given the **Original Prompt** that was used to generate a draft, followed by the **Draft Analysis** itself.

        Your tasks are:
        1.  **Adherence to Instructions:** First, assess if the **Draft Analysis** fully adheres to all instructions in the **Original Prompt** (e.g., word count, tone, format, specific points to cover, inclusion of valuation context, staying on topic).
        2.  **Rigorous Fact-Checking:** Scrutinize every number, statistic, date, and proper noun in the draft. **Use your search tool to verify these against the latest available public information.** Pay special attention to financial metrics (revenue, EPS, margins), market data, and historical dates.
        3.  **Contextual Analysis:** Identify statements that lack crucial context. Is a growth rate impressive compared to competitors? Is a valuation high for its sector? Flag any missing context.
        4.  **Eliminate Vague Language:** Challenge and correct imprecise terms like "recently," "significant," or "some" with specific data points where possible.

        For each issue you find, prepare one entry in a **Corrections Summary** list using this exact format:
        - **Original:** "<The exact original sentence>"
        - **Corrected:** "<The updated, fully accurate sentence with precise terminology.>"
        - **Reasoning:** "<A brief explanation of why the correction is necessary (e.g., 'Outdated data,' 'Fails to provide valuation context as requested by prompt,' 'Imprecise language').>"

        Example of a good correction:
        - **Original:** "NVIDIA has seen significant growth in its data center segment."
        - **Corrected:** "For the fiscal year ended January 28, 2024, NVIDIA's Data Center segment reported revenue of $47.5 billion, a 217% increase year-over-year."
        - **Reasoning:** "Imprecise language. Replaced 'significant growth' with a specific, verifiable data point and timeframe for accuracy and impact."

        Do not output the original draft. Only output the **Corrections Summary** list.
        If the entire analysis is accurate, well-contextualized, and precise, output the single phrase: "ALL GOOD"
        """

def generate_edit_prompt(company_name, old_output, correction_output):
    return f"""
        You are a senior equity research analyst and publication-ready writer.

        Here is a draft of the company analysis of {company_name}:
        {old_output}

        You've just received the following Corrections Summary for a draft company analysis of {company_name}:
        {correction_output}

        Your task is to integrate these corrections into a fully polished, cohesive analysis.

        Make sure you are :
        - Using the **corrected facts** and adhering to the **reasoning** exactly as stated in the summary.
        - Employing precise financial terminology and absolute dates (e.g. “Q1 FY2025,” “May 22, 2024”).
        - Providing smooth narrative transitions between sections.
        - Concluding with a concise, bullet-point list of the most important takeaways after each section.

        Output **only** the final analysis. Do not include any headers or titles.
    """

def generate_response(api_key: str, prompt: str) -> str:
    """
    Generates a response from the Google "gemini-2.5-flash" model.
    Includes retry logic for quota errors.

    Args:
        api_key: Your Google Generative AI API key.
        prompt: The prompt to send to the model.

    Returns:
        The generated text from the model.
    """
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

    MAX_RETRIES = 3 # Maximum number of retries for quota errors
    for attempt in range(MAX_RETRIES):
        try:
            response = model.models.generate_content(
                model=model_name,
                contents=prompt,
                config=configX,
            )
            response_text = response.text

            # Log the LLM call with the config object directly
            log_llm_call("generate_response", prompt, response_text, model_name, confNoJsonSearch)
            return response_text
        except Exception as e:
            error_message = str(e)
            if "429" in error_message and "RESOURCE_EXHAUSTED" in error_message:
                delay = 60
                match = re.search(r"'retryDelay':\s*'(\d+)s'", error_message)
                if match:
                    delay = int(match.group(1))
                
                print(f"Quota exceeded (429) for {model_name}. API suggests retrying in {delay} seconds. Attempt {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(delay)
            else:
                print(f"An unexpected error occurred during LLM call: {error_message}")
                raise
    
    print(f"Failed to get a response from {model_name} after {MAX_RETRIES} attempts due to quota limits.")
    raise Exception("Max retries exceeded for Gemini API due to quota limits.")

def generate_critique_feedback (api_key, company_name, response_content, original_prompt):
    """
    Generates critique feedback from the Google "gemini-2.5-flash" model.
    Does NOT use web search capabilities via tools.
    Includes retry logic for quota errors.
    """
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
    
    critique_instructions = generate_critique_prompt(company_name)
    full_prompt_for_critique = (
        f"{critique_instructions}\n\n"
        f"--- ORIGINAL PROMPT ---\n{original_prompt}\n\n"
        f"--- DRAFT ANALYSIS TO CRITIQUE ---\n{response_content}"
    )
    
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            response = model.models.generate_content(
                model=model_name,
                contents=full_prompt_for_critique,
                config=configX,
            )
            response_text = response.text

            log_llm_call("generate_critique_feedback", full_prompt_for_critique, response_text, model_name, confNoJsonSearch)
            return response_text
        except Exception as e:
            error_message = str(e)
            if "429" in error_message and "RESOURCE_EXHAUSTED" in error_message:
                delay = 60
                match = re.search(r"'retryDelay':\s*'(\d+)s'", error_message)
                if match:
                    delay = int(match.group(1))

                print(f"Quota exceeded (429) for {model_name} (critique feedback). API suggests retrying in {delay} seconds. Attempt {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(delay)
            else:
                print(f"An unexpected error occurred during critique LLM call: {error_message}")
                raise
    
    print(f"Failed to get critique feedback from {model_name} after {MAX_RETRIES} attempts due to quota limits.")
    raise Exception("Max retries exceeded for Gemini API (critique feedback) due to quota limits.")

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


def analyze_single_topic(company_name: str, topic: Topic, gemini_api_key: str) -> tuple[str, str]:
    """
    Generates, critiques, and refines the analysis for a single topic.
    Returns a tuple containing the topic label and the final refined analysis string.
    """
    # 1. Generate rough draft
    prompt = generate_topic_prompt(company_name, topic)
    current_analysis = generate_response(gemini_api_key, prompt)
    
    # 2. Critique and Redraft loop
    amount_of_cycles = 2
    count = 0
    while count < amount_of_cycles:
        correction_output = generate_critique_feedback(gemini_api_key, company_name, current_analysis, prompt)
        if correction_output.strip().upper() == "ALL GOOD":
            break
        else:
            edit_prompt = generate_edit_prompt(company_name, current_analysis, correction_output)
            current_analysis = generate_response(gemini_api_key, edit_prompt)
            count += 1
            
    # *** CHANGE: Return a tuple of (label, text) to separate content from formatting ***
    return (topic.label, current_analysis)


def analyze_company(gemini_api_key: str, open_router_api_key: str):
    print("Welcome to Company Analysis Bot!")
    print("This bot will provide a comprehensive analysis of any company.\n")
    while True:
        company_name = input("Enter company name (or 'exit' to quit): ")
        if company_name.lower() in ['exit', 'quit']:
            print("\nGoodbye!")
            break
        
        print(f"\nGenerating a comprehensive analysis for {company_name}. This may take a few minutes...")
        try:
            topics = list(Topic)
            
            ordered_futures = []
            with ThreadPoolExecutor(max_workers=8) as exe:
                for topic in topics:
                    future = exe.submit(analyze_single_topic, company_name, topic, gemini_api_key)
                    ordered_futures.append((topic.code, future))

                sorted_results = []
                for code, future in sorted(ordered_futures):
                    sorted_results.append(future.result())

            # *** CHANGE: Centralized and structured output formatting ***
            print(f"\n--- Final Comprehensive Analysis for {company_name} ---")
            for topic_label, analysis_text in sorted_results:
                # Print a clear, differentiating header for each topic
                print(f"\n\n{'='*70}")
                print(f"TOPIC: {topic_label.upper()}")
                print(f"{'='*70}\n")
                
                # Apply formatting and print the analysis for the topic
                formatted_text = apply_ansi_formatting(analysis_text)
                print(formatted_text)

            print(f"\n\n--- End of Analysis for {company_name} ---")

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