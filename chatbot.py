import os
import requests
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

console = Console()

def setup_api():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY in your .env file")
    return api_key

def generate_analysis_prompt(company_name):
    return f"""You are a senior investor with a deep understanding of the stock market. Prepare a detailed analysis of {company_name} covering:

    1. {company_name}'s History:
      - business model
      - founding year, place, and founders
      - early business model or first products/services
      - major funding rounds or IPO date
      - key acquisitions, partnerships, or divestitures
      - strategic pivots or rebrandings
      - recent corporate milestones (e.g., new CEO, geographic expansions)
    2. {company_name}'s products, industry, and market size  
      - Core product or service categories
      - adjacent offerings or R&D projects in the pipeline
      - industry classification (e.g., “semiconductors,” “cloud computing”)
      - estimated total addressable market (TAM) and sources
      - growth rate of the industry segment
      - emerging trends or technologies shaping the market
    3. {company_name}'s Revenue breakdown by product lines and by geography
      - revenue by major product or service line
      - revenue by region (Americas, EMEA, APAC, etc.)
      - year-over-year shifts in those percentages
      - contribution of recurring vs. one-time revenue
      - seasonality effects or quarter-to-quarter patterns
      - impact of new launches on revenue mix
    4. {company_name}'s customers
      - customer base and distribution channels
      - key customers and their importance
      - recent customer wins or losses
      - customer satisfaction or loyalty metrics
      - customer retention or churn rates
      - customer acquisition costs
      - customer lifetime value
    5. {company_name}'s competitors, and both {company_name}'s and its competitors' advantages and weaknesses
      - direct vs. indirect competitors
      - head-to-head feature / price / distribution comparisons
      - {company_name}'s moats or differentiators (technology, scale, brand)
      - gaps or vulnerabilities (R&D spend, customer concentration, regulation)
      - recent moves by competitors (M&A, product launches)
      - potential disruption threats (startups, substitutes)
    6. {company_name}'s 2021-2025 financial performance
      - revenue growth
      - gross margin
      - net income
      - cash flow
      - debt ratio
    7. {company_name}'s key drivers for the stock over the next 12 months 
      - product launches or roadmap milestones
      - macro trends (e.g., interest rates, consumer spending)
      - analyst estimate revisions or consensus targets
      - upcoming catalysts (earnings beats, partnerships)
      - share buybacks or dividend policy changes
      - regulatory or geopolitical tailwinds
    8. {company_name}'s main risks involved in investing in the stock
      - competitive threats or price wars
      - supply-chain or input-cost pressures
      - regulatory, legal, or antitrust scrutiny
      - currency or geopolitical exposure
      - execution risk on new initiatives
      - valuation overshoot or market sentiment shifts

Provide specific, factual,  up-to-date and most importantly in-depth information about {company_name}. If there is more information available in any of the above topics, please include them. Ensure that each topic has a specific title font.
Please end your response with "Are there any other companies you would like me to analyze?"
"""

def generate_response(api_key, prompt):
    """Generate a response using the OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": "deepseek/deepseek-chat-v3-0324:free", 
        "messages": [
            {"role": "system", "content": '''You are a knowledgeable financial senior analyst with expertise in company analysis. You have access to up-to-date financial data and news. You are also able to access the latest financial reports and news. For each request, you should:
              Organize your answer into eight sections with the title of each section clearly labeled in a title font.
              Be thorough and specific with your analysis in your response
              Include absolute dates (e.g. “QoQ growth in Q1 2025 was…”)
              When possible reference your data sources (e.g. SEC filings, company presentations)   
             '''},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 2000,
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
            prompt = generate_analysis_prompt(company_name)
            analysis = generate_response(api_key, prompt)
            
            console.print("\n[bold green]Analysis:[/bold green]")
            console.print(Markdown(analysis))
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
