import os
import requests
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv
from datetime import date, datetime
from enum import Enum

console = Console()

class Topic(Enum):
    HISTORY = ("history", 1)
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
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY in your .env file")
    return api_key

def generate_analysis_prompt(company_name, recent_news):
    return f"""You are a senior investor with deep domain expertise. 
    Provide specific, factual, up-to-date, and in-depth analysis of {company_name}. 
    For each topic below, write a cohesive paragraph in full sentences that should be around 1000-1500 words. 
    While writing this analysis, use financial terms percisely and provide valuation context. 
    If a brief bullet list would clarify key items (e.g., milestones, top product lines), 
    append it at the end of that paragraph. However, these don't count towards the above word count.
    Feel free to add any other relevant details you find on each of these topics if the list below isn't enough
    Lastly, write in a tone that is suitable for your audience of stock investors:

1. History  
   • Business model evolution  
   • Founding year, location, and founders  
   • Early products or services  
   • Major funding rounds or IPO  
   • Acquisitions, partnerships, or divestitures  
   • Strategic pivots or rebrandings  
   • Recent milestones (new CEO, geographic expansion)

2. Products, Industry & Market Size  
   • Core offerings and adjacent R&D projects  
   • Industry classification (e.g., “semiconductors”)  
   • Total addressable market (TAM) with sources  
   • Segment growth rates  
   • Emerging trends shaping the market

3. Revenue Breakdown (by Product & Geography)  
   • Revenue per major product or service  
   • Revenue by region (Americas, EMEA, APAC)  
   • YoY shifts in those percentages  
   • Recurring vs. one-time revenue mix  
   • Seasonality or quarter-to-quarter patterns  
   • Effect of recent launches on mix

4. Customers  
   • Customer segments and distribution channels  
   • Key accounts and their impact  
   • Recent wins or losses  
   • Satisfaction, retention, and churn metrics  
   • Acquisition cost and lifetime value

5. Competitive Landscape  
   • Direct and indirect competitors  
   • Feature, price, and distribution comparisons  
   • {company_name}'s moats or differentiators  
   • Competitors' vulnerabilities  
   • Recent competitor moves (M&A, new products)  
   • Disruption risks (startups, substitutes)

6. Financial Performance (2021-2025)  
   • Revenue growth trends  
   • Gross and net margins  
   • Cash flow dynamics  
   • Debt ratios

7. Key Stock Drivers (Next 12 Months)  
   • Upcoming product or roadmap milestones  
   • Macro trends (interest rates, consumer spending)  
   • Analyst estimate revisions or consensus targets  
   • Catalysts (earnings beats, partnerships)  
   • Capital allocation (buybacks, dividends)  
   • Regulatory or geopolitical tailwinds

8. Investment Risks  
   • Competitive pressure or price wars  
   • Supply-chain or cost headwinds  
   • Regulatory, legal, or antitrust scrutiny  
   • Currency or geopolitical exposure  
   • Execution risks on new initiatives  
   • Valuation or sentiment shifts

For each of the 8 topics, please organize your response in this format:
- Heading
- One bullet point discussing each of the subtopics that are written above. If there is more information, you can add additionall bullet points. Make sure these bullet points are 1 sentence and discuss the main idea
- For each bullet point, write a paragraph providing a detailed summary that goes more indepth into each of the bullet points that are the main idea. (ex. how this affects {company_name}, why this matters to {company_name}, etc.)

Please end your response with "Are there any other companies you would like me to analyze?"
"""

def generate_news_prompt(company_name):
    today = date.today()
    return f"""
        You are a financial news analyst with expertise in corporate reporting.
        Today is {today}.
        Please gather and summarize the most recent news and developments for {company_name}
        over the past 30 days. Make sure you output your response in order from newest to latest.
        Here is how you should organize your response:

        1. Date of publiciation
        2. Headline
        3. Source (with a link that can be clicked)
        4. In-depth summary of the news being reported (100-200 words)
        5. **Implication** for the company's business or stock (200-300 words)

        Here is a good example of a news entry for Amazon on June 22, 2025:

        Date of Publication: June 19, 2025

        Headline: Amazon to invest $233 million in India to expand operations, improve technology

        Source: reuters.com

        In-Depth Summary: On June 19, 2025, Amazon announced a planned investment of ₹20 billion (~$233 million)
        in India over the coming year to bolster its fulfillment and delivery network. The capital will fund new 
        fulfillment centers, upgrades to existing facilities, and the deployment of advanced technologies aimed 
        at improving processing speed and delivery safety. Part of the funds will support real-time monitoring tools 
        that alert delivery personnel to unsafe driving speeds and optimize route allocation. Additionally, Amazon 
        intends to expand employee welfare programs, enhancing health and financial benefits for its Indian workforce. 
        This commitment builds on Amazon's earlier pledge in June 2023 to invest $26 billion in India by 2030, underscoring 
        the market's strategic importance to its global growth strategy.

        Implication:
        Amazon's substantial infusion of capital into India reinforces its long-term commitment to one of the world's fastest-growing 
        e-commerce markets. By expanding infrastructure and integrating advanced logistics technology, Amazon aims to reduce delivery 
        lead times and increase customer satisfaction, which could translate into higher market share against local rival Flipkart and 
        emerging discount platforms like JioMart. The increased welfare spending signals a strategic effort to mitigate high attrition 
        rates in India's logistics workforce, improving operational stability. However, the substantial upfront investment will weigh 
        on free cash flow in the near term, particularly as global cost pressures persist. Investors should monitor whether these enhancements 
        drive measurable improvements in Indian revenue growth and margin expansion over the next several quarters; failure to realize efficiencies 
        could dampen returns on this capital outlay and place pressure on Amazon's overall cash-flow profile.
    """

def generate_topic_prompt (company_name, topic):
    today = date.today()
    # create a dictionary mapping topic to the list of topic examples.
    topic_to_list = {
        1: """Business model evolution, Founding year, location, and founders, 
              Early products or services, Major funding rounds or IPO, Acquisitions, 
              partnerships, or divestitures, Strategic pivots or rebrandings, 
              Recent milestones (new CEO, geographic expansion)""",
        2: '''Core offerings and adjacent R&D projects, Industry classification (e.g., “semiconductors”), 
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

    return f"""
        You are a knowledgeable financial senior analyst with expertise in company analysis. 
        Today is {today}. Use up-to-date financial data, reports, and news to write your response.
        Write a cohesive paragraph in full sentences that is ~ 250-500 words about the {topic.label} of {company_name}.
        (Some topics or ideas you can talk about, but are not limited to are: {topic_to_list[topic.code]}).
        While writing this analysis, use financial terms percisely and provide valuation context. 
        Lastly, end your response with a brief bullet-pointed summary of the in-depth response that summarizes
        extracts the main points you brought up.
        Don't add any extra follow up sentences after the summary.

        
        Here is an example of a great response on the key stock drivers of Amazon on 6/22/2025:

            Over the next twelve months, Amazon's share performance will be underpinned primarily by its 
            ongoing transformation into an AI-first enterprise, with Amazon Web Services (AWS) at the vanguard 
            of both top-line growth and margin expansion. AWS is set to introduce a suite of next-generation offerings—ranging 
            from Aurora PostgreSQL version 17 and enhanced Bedrock foundation models to its quantum-compute “Ocelot” chip—that 
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
            “dark patterns” to trick consumers into auto-renewing Prime memberships is set for a bench trial in June 2025, 
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

            Regulatory & Legal Scrutiny: FTC “dark patterns” trial in June 2025; broader antitrust suit trial in October 2026.

            Execution Risks: High-cost projects (Project Kuiper, AI infrastructure) may under-deliver on returns.

            Valuation Sensitivity: Trading at ~34x TTM earnings (as of June 20, 2025), shares fell 3% on cautious Q2 guidance.

    """

def generate_key_drivers_prompt(company_name):
    return f'''
        You are a knowledgeable financial senior analyst with expertise in company analysis. 
        You have access to up-to-date financial data and news. You are also able to access the latest financial reports and news.
        Write a cohesive paragraph in full sentences that is ~ 250-500 words about the key stock drivers of {company_name}. 
        (Some topics you can talk about but are not limited to is Upcoming product or roadmap milestones,
        Macro trends (interest rates, consumer spending)  , Analyst estimate revisions or consensus targets , Catalysts 
        (earnings beats, partnerships)  , Capital allocation (buybacks, dividends)  , Regulatory or geopolitical tailwinds, etc.)
        While writing this analysis, use financial terms percisely and provide valuation context. 
        Lastly, end your response with a brief bullet-pointed summary of the in-depth response that summarizes
        extracts the main points you brought up. Don't add any extra follow up sentences after the summary.
        
        Here is an example of a great response on the key stock drivers of Amazon on 6/22/2025:

        Over the next twelve months, Amazon's share performance will be underpinned primarily by its 
        ongoing transformation into an AI-first enterprise, with Amazon Web Services (AWS) at the vanguard 
        of both top-line growth and margin expansion. AWS is set to introduce a suite of next-generation offerings—ranging 
        from Aurora PostgreSQL version 17 and enhanced Bedrock foundation models to its quantum-compute “Ocelot” chip—that 
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

    '''

def generate_risk_prompt(company_name):
    return f'''
        You are a knowledgeable financial senior analyst with expertise in company analysis.
        Write a cohesive paragraph in full sentences that is ~ 250-500 words about the investment risks of {company_name}. 
        (Some examples you can talk about, but are not limited to, is 
        Competitive pressure or price wars, Supply-chain or cost headwinds, 
        Regulatory, legal, or antitrust scrutiny, Currency or geopolitical exposure,
        Execution risks on new initiatives, Valuation or sentiment shifts, etc.)
        While writing this analysis, use financial terms percisely and provide valuation context. 
        Lastly, end your response with a brief bullet-pointed summary of the in-depth response that summarizes
        extracts the main points you brought up. Don't add any extra follow up sentences after the summary.
        
        Here is an example of a great response on the investment risks of Amazon on 6/22/2025:

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
            “dark patterns” to trick consumers into auto-renewing Prime memberships is set for a bench trial in June 2025, 
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

            Regulatory & Legal Scrutiny: FTC “dark patterns” trial in June 2025; broader antitrust suit trial in October 2026.

            Execution Risks: High-cost projects (Project Kuiper, AI infrastructure) may under-deliver on returns.

            Valuation Sensitivity: Trading at ~34x TTM earnings (as of June 20, 2025), shares fell 3% on cautious Q2 guidance.

    '''

def generate_response(api_key, prompt):
    """Generate a response using the OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "deepseek/deepseek-chat-v3-0324:free", 
        "messages": [
            {"role": "system", "content": '''
             You are a knowledgeable financial senior analyst with expertise in company analysis. 
             You have access to up-to-date financial data and news. You are also able to access the latest financial reports and news. 
             For each request, you should: Be thorough and specific with your analysis in your response
             Include absolute dates (e.g. “QoQ growth in Q1 2025 was…”)
             When possible reference your data sources (e.g. SEC filings, company presentations).
             Furthermore, write in a tone that is suitable for your audience of stock investors.
             Lastly, make sure everything is in paragraph form with well chosen subheadings that describe what is contained in the body.
             '''},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "top_p": 0.85,
        "max_tokens": 5000,
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

def analyze_company(api_key):
    console.print("[bold green]Welcome to Company Analysis Bot![/bold green]")
    console.print("This bot will provide a comprehensive analysis of any company.\n")
    
    while True:
        company_name = console.input("[bold blue]Enter company name (or 'exit' to quit):[/bold blue] ")
        
        if company_name.lower() in ['exit', 'quit']:
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break
        
        try:
            console.print(f"\n[yellow]Analyzing {company_name}...[/yellow]")
            # risk_prompt = generate_risk_prompt(company_name)
            # drivers_prompt = generate_key_drivers_prompt(company_name)
            # risk_analysis = generate_response(api_key, risk_prompt)
            # drivers_analysis = generate_response(api_key, drivers_prompt)

            history_prompt = generate_topic_prompt(company_name, Topic.HISTORY)
            fin_performance_prompt = generate_topic_prompt(company_name, Topic.FINANCIAL_PERFORMANCE)
            drivers_prompt = generate_topic_prompt(company_name, Topic.STOCK_DRIVERS)
            risk_prompt = generate_topic_prompt(company_name, Topic.INVESTMENT_RISKS)

            history_analysis = generate_response(api_key, history_prompt)
            fin_performance_analysis = generate_response(api_key, fin_performance_prompt)
            risk_analysis = generate_response(api_key, risk_prompt)
            drivers_analysis = generate_response(api_key, drivers_prompt)

            console.print("\n[bold green]Analysis:[/bold green]")

            console.print("\n[bold green]History[/bold green]", justify="center")
            console.print(Markdown(history_analysis))
            console.print() 

            console.print("\n[bold green]Financial Performance[/bold green]", justify="center")
            console.print(Markdown(fin_performance_analysis))
            console.print() 
            
            console.print("\n[bold green]Key Stock Drivers[/bold green]", justify="center")
            console.print(Markdown(drivers_analysis))
            console.print() 
            
            console.print("\n[bold green]Investment Risks[/bold green]", justify="center")
            console.print(Markdown(risk_analysis))
            console.print() 
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

def main():
    try:
        api_key = setup_api()
        analyze_company(api_key)
    except Exception as e:
        console.print(f"[bold red]Fatal Error:[/bold red] {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
