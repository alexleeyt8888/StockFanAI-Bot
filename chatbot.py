import os
import requests
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv
from datetime import date, datetime
from enum import Enum
from google import genai
from google.genai import types


console = Console()

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
    if not open_router_api_key:
        raise ValueError("Please set API key in your .env file")
    return gemini_api_key, open_router_api_key


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

    prompt_string = f"""
        You are a knowledgeable financial senior analyst with expertise in company analysis. 
        Today is {today}. Using up-to-date information write a cohesive paragraph in full sentences 
        that is ~ 250-500 words about the {topic.label} of {company_name}.
        (Some topics or ideas you can talk about, but are not limited to are: {topic_to_list[topic.code]}).
        While writing this analysis, use financial terms percisely and provide valuation context. 
        Lastly, end your response with a brief bullet-pointed summary of the in-depth response that summarizes
        extracts the main points you brought up.
        Don't add any extra follow up sentences after the summary.
        
        Here is an example of a great response on the key stock drivers of Amazon on 6/22/2025:

            Over the next twelve months, Amazon's share performance will be underpinned primarily by its 
            ongoing transformation into an AI-first enterprise, with Amazon Web Services (AWS) at the vanguard 
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

            AWS AI & Cloud Innovation: New AI-driven services and Aurora v17 to sustain low-teens revenue growth and ~33% margins

            Retail Efficiency Gains: Machine-learning in logistics to compress fulfillment costs by 50-75 bps in FY 2026

            Prime Video & Advertising: AI dubbing and Upfront 2025 to enhance ARPU and ad revenues

            Macro & Geopolitical: Fed pause and resilient consumer spending vs. tariff and antitrust risks

            Analyst Consensus: ~$246 price target; FY 2025 EPS $6.30 (+14%), implying ~39x forward P/E

            Capital Allocation: Evolving capex discipline to bolster free cash flow and fund share buybacks


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
            revenue grew 30% in the quarter ended June 30, 2024—and Google Cloud—up 30% in revenue in Q4 2024—have narrowed AWS's
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
            
            Summary of Key Investment Risks:

            Currency & Geopolitical Exposure: $1.4 billion FX headwind in Q1 2025; potential U.S.-China tariffs up to 145%.

            Cost & Supply-Chain Pressures: Elevated labor, logistics, and infrastructure expenses squeezing margins.

            Competitive Intensity: E-commerce undercut by Walmart, Alibaba, Temu; AWS challenged by Azure and Google Cloud.

            Regulatory & Legal Scrutiny: FTC "dark patterns" trial in June 2025; broader antitrust suit trial in October 2026.

            Execution Risks: High-cost projects (Project Kuiper, AI infrastructure) may under-deliver on returns.

            Valuation Sensitivity: Trading at ~34x TTM earnings (as of June 20, 2025), shares fell 3% on cautious Q2 guidance.

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

# Prompt for critic model (gemini)
# def generate_critique_prompt(company_name, topic):
#     today = date.today()
#     return f"""
#             You are a senior equity research analyst and fact-checker. Today is {today}.  
#             You have just read a draft analysis of {company_name} on “{topic}.”  

#             Your tasks are:  
#             1. Verify every factual claim, statistic, and date in the draft. Update any out-of-date or incorrect figures to the latest available data (cite your sources inline).  
#             2. Correct any misstatements or inconsistencies.  
#             3. Restructure and polish the narrative into a cohesive, publication-ready analysis with clear subheadings (e.g. Overview, Key Metrics, Valuation, Risks).  
#             4. Ensure financial terminology is used precisely and that any valuation context is clearly explained.  
#             5. Conclude with a concise bullet-point summary of the main takeaways.  

#             Please output only the fully polished analysis (no internal notes), with in-text citations where you updated or confirmed a fact.
#             """

def generate_critique_prompt(company_name):
    today = date.today()
    return f"""
        You are a senior equity research analyst and fact-checker. Today is {today}.

        You have just read a draft analysis of {company_name}

        Your tasks are:
        1. Verify every factual claim, statistic, and date in the draft by searching the web for up-to-date information.
        2. For each incorrect or outdated sentence, prepare one entry in a **Corrections Summary** list:
        - **Original:** “<the exact original sentence>”
        - **Corrected:** “<the updated, fully accurate sentence>” (Source: <inline citation>)
        3. Do not output the draft itself—only the Corrections Summary list.
        4. Use precise financial terminology and exact dates where applicable.
        5. If everything is factual and correct please output "ALL GOOD"
        """

def generate_edit_prompt(company_name, old_output, correction_output):
    return f"""
        You are a senior equity research analyst and publication-ready writer. 
        
        Here is a draft of the companay analysis of {company_name}:
        {old_output}
        
        You've just received the following Corrections Summary for a draft company analysis of {company_name}:
        {correction_output}
        
        Your task is to integrate these corrections into a fully polished, cohesive analysis.

        Make sure you are :
        - Useing the **corrected facts** exactly as stated in the summary, with in-text citations matching the sources.  
        - Employing precise financial terminology and absolute dates (e.g. “Q1 FY2025,” “May 22, 2024”).  
        - Providing smooth narrative transitions between sections.  
        - Concluding with a concise, bullet-point list of the most important takeaways after each section.

        Output **only** the final analysis.  
    """

# Response from deepseek r1 model. Drafting model
def generate_response_with_deepseek(api_key, prompt):
    """Generate a response using the OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "deepseek/deepseek-r1-0528:free", 
        "messages": [
            {"role": "system", "content": '''
             You are a knowledgeable financial senior analyst with expertise in company analysis. 
             You have access to up-to-date financial data and news. You are also able to access the latest financial reports and news. 
             For each request, you should: Be thorough and specific with your analysis in your response
             Include absolute dates (e.g. "QoQ growth in Q1 2025 was…")
             When possible reference your data sources (e.g. SEC filings, company presentations).
             Furthermore, write in a tone that is suitable for your audience of stock investors.
             Lastly, make sure everything is in paragraph form with well chosen subheadings that describe what is contained in the body.
             '''},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6,
        "top_p": 0.85,
        "stop": ["Are there any other companies you would like me to analyze?"]
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]API Error:[/bold red] {str(e)}")
        if hasattr(e.response, 'text'):
            console.print(f"[bold red]Response:[/bold red] {e.response.text}")
        return "I apologize, but I encountered an error while generating the analysis. Please try again."

# Response to critique the current draft
def generate_critique_feedback (api_key, company_name, response):
    client = genai.Client(api_key=api_key)
    critique_prompt = generate_critique_prompt(company_name)
    critique_content = f'''
                            Here is the draft you should critique and refine:

                            {response}
                        '''
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=critique_prompt, temperature = 0.2),
        contents=critique_content,
    )
    return response.text

def analyze_company(deepseek_api_key, gemini_api_key):
    console.print("[bold green]Welcome to Company Analysis Bot![/bold green]")
    console.print("This bot will provide a comprehensive analysis of any company.\n")
    while True:
        company_name = console.input("[bold blue]Enter company name (or 'exit' to quit):[/bold blue] ")
        
        if company_name.lower() in ['exit', 'quit']:
            console.print("\n[bold yellow]Goodbye![bold yellow]")
            break
        try:
            console.print(f"\n[yellow]Analyzing {company_name}...[yellow]")

            console.print("\n[bold green]Analysis:[/bold green]")

            response = "\n"

            topic_enums = list(Topic)
            # loop through the 8 topics
            for topic in topic_enums:
                # rough draft of response
                prompt = generate_topic_prompt(company_name, topic)
                draft_analysis = generate_response_with_deepseek(deepseek_api_key, prompt)
                response = response + f"[bold green]{topic.label}[/bold green]\n" + draft_analysis + "\n"

                # TODO ADD THE COMPARISON
                # console.print("\n[bold green]Competitive Comparison[/bold green]", justify="center")

            # critiquing the output
            correction_output = generate_critique_feedback(gemini_api_key, company_name, draft_analysis)
            
            # debug
            console.print(Markdown(correction_output))
            console.print()
            count = 0
            amount_of_cycles = 1
            # go through the cycle of critiquing and redrafting 
            # or until there is no incorrect information
            while count < amount_of_cycles and correction_output != "ALL GOOD":
                edit_prompt = generate_edit_prompt(company_name, response, correction_output)
                response = generate_response_with_deepseek(deepseek_api_key, edit_prompt)
                correction_output = generate_critique_feedback(gemini_api_key, company_name, response)
                console.print(Markdown(correction_output))
                console.print()
                count+=1

            console.print(Markdown(response))

            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")  

def main():
    try:
        gemini_api_key, deepseek_api_key = setup_api()
        analyze_company(deepseek_api_key, gemini_api_key)
    except Exception as e:
        console.print(f"[bold red]Fatal Error:[/bold red] {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
