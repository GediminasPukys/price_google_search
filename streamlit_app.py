import streamlit as st
from googleapiclient.discovery import build
from openai import OpenAI
from pydantic import BaseModel, Field
import pandas as pd
import re
import json
from typing import List, Dict, Optional, Union, Any
from firecrawl import FirecrawlApp

# Import database operations
from db_operations import (
    get_product_specifications,
    get_specification_parameters,
    get_demo_specs_data,
    get_demo_spec_params
)


class ProductPrice(BaseModel):
    provider: str
    provider_website: str
    provider_url: str
    product_name: str
    product_properties: str
    product_sku: str
    product_price: str
    price_per_: str
    evaluation: str


class ProductList(BaseModel):
    products: list[ProductPrice]


class ProductUrl(BaseModel):
    product_name: str
    product_url: str


class ProductUrlList(BaseModel):
    products: list[ProductUrl]


class PageJudge(BaseModel):
    is_aggregator: bool


class SearchPhrase(BaseModel):
    """Model for structured search phrase output from OpenAI."""
    search_phrase: str = Field(description="Optimized search phrase for finding products")
    keywords: list[str] = Field(description="Individual keywords extracted from specifications")


def get_urls(client, category, subcategory, product_type, specification_name, scrape_result):
    """Extract product URLs from aggregator pages."""
    prompt = f"""In the given content of a price aggregator website, find relevant products and their corresponding URLs.

    Relevant products should match: category: {category} {subcategory} {product_type} and product type: {specification_name}

    Content: {scrape_result}
    """

    response = client.responses.parse(
        model="gpt-4o",
        input=prompt,
        text_format=ProductUrlList,
    )

    parsed_product_urls = []
    for product in response.output_parsed.products:
        parsed_product_urls.append(product.model_dump())
    return parsed_product_urls


def get_prompt(category, subcategory, product_type, specification_name, tech_spec, scrape_result):
    """Generate analysis prompt for product webpage content."""
    prompt = f"""
    Analyze the given product webpage content:
    Content: {scrape_result}

    Next steps:
    1. Find all products in the content and gather detailed product information (this means return as many products as possible).
    2. Judge whether products meet the provided specifications:
       - Category: {category} {subcategory} {product_type} 
       - Product type: {specification_name}
       - Product specifications:
{tech_spec}
    3. Verify the product is currently available for purchase
    4. Gather accurate pricing in EUR
    5. Evaluate technical specification requirements one by one
    6. Calculate and include price per unit for each product
    """

    # JSON format instructions
    prompt += """
    IMPORTANT: Your response MUST be formatted EXACTLY as a valid JSON array of product objects.
    Each product in the array should have the following fields:

    [
      {
        "provider": "Company selling the product",
        "provider_website": "Main website domain (e.g., telia.lt)",
        "provider_url": "Full URL to the specific product page",
        "product_name": "Complete product name with model",
        "product_properties": "Detailed product specifications as text",
        "product_sku": "Any product identifiers (SKU, UPC, model number)",
        "product_price": "Price in EUR with currency symbol",
        "price_per_": "Price per unit if applicable",
        "evaluation": "Detailed assessment of how the product meets or fails each technical specification"
      }
    ]

    DO NOT include any explanation, preamble, or additional text - ONLY provide the JSON array.
    """
    return prompt


def analyze_product_url(url, search_parameters, openai_api_key, firecrawl_api_key, is_aggregator_page=False):
    """Analyze a product page URL to extract price information using improved scraping and analysis."""
    client = OpenAI(api_key=openai_api_key)
    app = FirecrawlApp(api_key=firecrawl_api_key)

    category = search_parameters.get("grupe", "")
    subcategory = search_parameters.get("modulis", "")
    product_type = search_parameters.get("dalis", "")
    specification_name = search_parameters.get("specification_name", "")

    # Format the tech_spec from specification_parameters
    tech_spec = ""
    for param in search_parameters.get("specification_parameters", []):
        tech_spec += f"- {param.get('parametras', '')}: {param.get('reikalavimas parametrui', '')}\n"

    try:
        # Scrape the website using FireCrawl
        scrape_result = app.scrape_url(url, formats=['markdown']).model_dump()

        if not scrape_result or 'markdown' not in scrape_result:
            st.error(f"Failed to scrape URL: {url}")
            return None

        content = scrape_result['markdown']
        product_list = []

        if is_aggregator_page:
            st.info(f"🔗 Processing as aggregator page: {url}")
            # Get individual product URLs from the aggregator
            urls_list = get_urls(client, category, subcategory, product_type, specification_name, content)

            if not urls_list:
                st.warning(f"No relevant product URLs found on aggregator page: {url}")
                return []

            st.info(f"Found {len(urls_list)} product URLs to analyze")

            # Analyze each individual product URL
            for i, url_data in enumerate(urls_list):
                product_url = url_data.get("product_url", "")
                if not product_url:
                    continue

                try:
                    st.info(f"Analyzing product {i + 1}/{len(urls_list)}: {product_url}")

                    # Scrape individual product page
                    product_scrape_result = app.scrape_url(product_url, formats=['markdown']).model_dump()

                    if product_scrape_result and 'markdown' in product_scrape_result:
                        product_content = product_scrape_result['markdown']

                        # Generate analysis prompt
                        prompt = get_prompt(category, subcategory, product_type, specification_name, tech_spec,
                                            product_content)

                        # Analyze the product page
                        response = client.responses.parse(
                            model="gpt-4o",
                            input=prompt,
                            text_format=ProductList,
                        )

                        # Add products to the list
                        if response.output_parsed.products:
                            for product in response.output_parsed.products:
                                product_list.append(product.model_dump())

                except Exception as e:
                    st.warning(f"Error analyzing product URL {product_url}: {str(e)}")
                    continue
        else:
            st.info(f"🛍️ Processing as direct product page: {url}")
            # Direct product page analysis
            prompt = get_prompt(category, subcategory, product_type, specification_name, tech_spec, content)

            response = client.responses.parse(
                model="gpt-4o",
                input=prompt,
                text_format=ProductList,
            )

            # Convert to list of dictionaries
            for product in response.output_parsed.products:
                product_list.append(product.model_dump())

        return product_list

    except Exception as e:
        st.error(f"Error analyzing product URL {url}: {str(e)}")
        return None


def generate_search_phrase(grupe, modulis, dalis, specifikacija, spec_params, openai_api_key):
    """Generate an optimized search phrase using OpenAI."""
    client = OpenAI(api_key=openai_api_key)

    # Format specification parameters for prompt
    params_text = ""
    if spec_params:
        params_text = "Specification Parameters:\n"
        for param in spec_params:
            params_text += f"- {param['parametras']}: {param['reikalavimas parametrui']}\n"

    # System message with examples for few-shot learning
    system_message = """You are an assistant that creates optimized search phrases for product searches. 
Return only JSON with 'search_phrase' and 'keywords' fields.

The search phrase should be optimized for finding prices of products. Focus on technical specifications and include the word "kaina" (price) in Lithuanian.

Here are examples of good search phrases:

Example 1:
Input:
{
  "group": "Kanceliarinės ir biuro prekės",
  "module": "Elektros prekių užsakymai",
  "part": {
    "code": "ELL23",
    "name": "Elektros lemputės"
  },
  "specification": {
    "code": "ELL1",
    "name": "Šviesos diodų (\"LED\") elektros lemputė, ne daugiau 4W, ne mažiau 230 Lm, GU5.3"
  },
  "specificationParameters": [
    {
      "name": "Šviesos diodų (\"LED\") elektros lemputė, ne daugiau 4W, ne mažiau 230 Lm, GU5.3",
      "value": "Gamintojo pavadinimas, prekės modelio pavadinimas, prekės kodas"
    },
    {
      "name": "GU5.3 cokolis",
      "value": "Taip"
    },
    {
      "name": "Galingumas",
      "value": "ne daugiau 4W"
    },
    {
      "name": "Šviesos srautas",
      "value": "ne mažiau 230 Lm"
    },
    {
      "name": "GU5.3 cokolio, šaltai (4000 K) baltos spalvos",
      "value": "Taip"
    },
    {
      "name": "Aplinkosauginiai reikalavimai",
      "value": "Taip"
    }
  ]
}
Output:
{
  "search_phrase": "LED elektros lemputė 4W 230 Lm GU5.3 4000K kaina",
  "keywords": ["LED", "elektros lemputė", "4W", "230 Lm", "GU5.3", "4000K", "kaina"]
}

Example 2:
Input:
{
  "group": "Degalai",
  "module": "95 benzinas ir dyzelinas iš degalinių",
  "part": {
    "code": "",
    "name": "95 benzinas"
  },
  "specification": {
    "code": "",
    "name": "95 benzinas"
  },
  "specificationParameters": [
    {
      "name": "95 benzinas",
      "value": "None"
    }
  ]
}
Output:
{
  "search_phrase": "degalai 95 benzinas kaina degalinese",
  "keywords": ["degalai", "95 benzinas", "kaina", "degalinese"]
}

Example 3:
Input:
{
  "group": "Sveikatos srities pirkimai",
  "module": "Vaistai 2022",
  "part": {
    "code": "",
    "name": "Vaistai (2023) 3 dalis"
  },
  "specification": {
    "code": "VST2023_1",
    "name": "2,4-dichlorobenzilo alkoholis / Amilmetakrezolis / Askorbo rūgštis"
  },
  "specificationParameters": [
    {
      "name": "(VST2023_1) 2,4-dichlorobenzilo alkoholis / Amilmetakrezolis / Askorbo rūgštis",
      "value": "None"
    }
  ]
}
Output:
{
  "search_phrase": "vaistas 2,4-dichlorobenzilo alkoholis amilmetakrezolis askorbo rūgštis kaina",
  "keywords": ["vaistas", "2,4-dichlorobenzilo alkoholis", "amilmetakrezolis", "askorbo rūgštis", "kaina"]
}

Create an optimized search phrase in Lithuanian that includes key specifications and the word "kaina" (price).
"""

    user_prompt = f"""
Create an optimized search phrase for finding products with these specifications:

Group: {grupe}
Module: {modulis}
Part: {dalis}
Specification: {specifikacija}
{params_text}

The search phrase should be effective for finding specific products that match these criteria and their prices.
Focus on technical aspects and make it suitable for a product search engine.
Always include the word "kaina" in the search phrase.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]
        )

        import json
        response_json = json.loads(response.choices[0].message.content)
        return SearchPhrase(
            search_phrase=response_json.get("search_phrase", ""),
            keywords=response_json.get("keywords", [])
        )
    except Exception as e:
        st.error(f"Error generating search phrase: {e}")
        return None


def retrieve_search_results(query, restricted_domains=None, included_domains=None, excluded_domains=None,
                            google_api_key=None,
                            google_cse_id=None, num_results=10):
    """
    Retrieve search results using Google Custom Search API with domain filtering.

    Args:
        query (str): Search query
        restricted_domains (list): List of domains to restrict search to (whitelist only)
        included_domains (list): List of domains to include when not restricted
        excluded_domains (list): List of domains to exclude when not restricted
        google_api_key (str): Google API key
        google_cse_id (str): Google Custom Search Engine ID
        num_results (int): Number of results to retrieve

    Returns:
        dict: Search results or error information
    """
    try:
        # Initialize the Custom Search API service
        service = build("customsearch", "v1", developerKey=google_api_key)

        # Handle domain restrictions
        if restricted_domains and len(restricted_domains) > 0:
            # For restricted domains, we'll search each domain separately and combine results
            st.info(f"🔒 Restricted search to domains: {', '.join(restricted_domains)}")

            all_results = []
            results_per_domain = max(1, num_results // len(restricted_domains))

            for domain in restricted_domains:
                try:
                    domain_query = f"{query} site:{domain}"
                    st.info(f"Searching {domain}: {domain_query}")

                    result = service.cse().list(
                        q=domain_query,
                        cx=google_cse_id,
                        num=min(results_per_domain, 10)  # Google API max is 10 per request
                    ).execute()

                    if "items" in result:
                        for item in result["items"]:
                            # Extract domain from URL
                            url = item.get("link", "")
                            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
                            extracted_domain = domain_match.group(1) if domain_match else "Unknown domain"

                            all_results.append({
                                "title": item.get("title", "No title"),
                                "url": url,
                                "snippet": item.get("snippet", "No description"),
                                "domain": extracted_domain,
                                "is_priority_domain": True
                            })

                except Exception as e:
                    st.warning(f"Error searching domain {domain}: {str(e)}")
                    continue

            # Sort by relevance and limit to requested number
            search_results = all_results[:num_results]
            modified_query = f"{query} (restricted to: {', '.join(restricted_domains)})"

        else:
            # Use include/exclude approach (default mode)
            modified_query = f"{query} site:.lt -filetype:pdf"

            # Exclude domains if specified
            if excluded_domains and len(excluded_domains) > 0:
                for domain in excluded_domains:
                    modified_query += f" -site:{domain}"

            st.info("🌐 Searching all Lithuanian (.lt) domains with exclusions")
            st.info(f"Search query: {modified_query}")

            # Execute the search
            result = service.cse().list(
                q=modified_query,
                cx=google_cse_id,
                num=num_results
            ).execute()

            # Extract results
            search_results = []
            if "items" in result:
                for item in result["items"]:
                    # Extract domain from URL
                    url = item.get("link", "")
                    domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
                    domain = domain_match.group(1) if domain_match else "Unknown domain"

                    # Mark if domain is in included list
                    is_priority_domain = False
                    if included_domains:
                        is_priority_domain = any(inc_domain in domain for inc_domain in included_domains)

                    search_results.append({
                        "title": item.get("title", "No title"),
                        "url": url,
                        "snippet": item.get("snippet", "No description"),
                        "domain": domain,
                        "is_priority_domain": is_priority_domain
                    })

            # Sort results to prioritize included domains if specified
            if included_domains:
                search_results.sort(key=lambda x: not x.get("is_priority_domain", False))

        return {"results": search_results, "query": modified_query}

    except Exception as e:
        return {"error": f"Google Search API request failed: {str(e)}"}


# Streamlit App
def main():
    st.title("Lithuanian Product Search")
    st.markdown("Find and analyze product prices in the Lithuanian market")

    # Set demo mode once at the beginning
    demo_mode = st.sidebar.checkbox("Demo Mode (No MySQL)", value=False, key="demo_mode_checkbox")

    # Check for API keys in secrets
    missing_keys = []
    if not st.secrets.get("config", {}).get("google_api_key"):
        missing_keys.append("google_api_key")
    if not st.secrets.get("config", {}).get("google_cse_id"):
        missing_keys.append("google_cse_id")
    if not st.secrets.get("config", {}).get("openai_api_key"):
        missing_keys.append("openai_api_key")
    if not st.secrets.get("config", {}).get("firecrawl_api_key"):
        missing_keys.append("firecrawl_api_key")

    # Only check for MySQL if not in demo mode
    if not demo_mode:
        if not st.secrets.get("mysql", {}).get("host"):
            missing_keys.append("mysql configuration")

    if missing_keys:
        st.error(f"""
        ⚠️ Missing configuration in secrets: {', '.join(missing_keys)}

        Create a `.streamlit/secrets.toml` file with:
        ```
        [config]
        google_api_key = "YOUR_GOOGLE_API_KEY"
        google_cse_id = "YOUR_CUSTOM_SEARCH_ENGINE_ID"
        openai_api_key = "YOUR_OPENAI_API_KEY"
        firecrawl_api_key = "YOUR_FIRECRAWL_API_KEY"

        [mysql]
        host = "YOUR_MYSQL_HOST"
        user = "YOUR_MYSQL_USERNAME"
        password = "YOUR_MYSQL_PASSWORD"
        database = "YOUR_MYSQL_DATABASE"
        ```
        """)

        with st.expander("How to set up required APIs"):
            st.markdown("""
            ### FireCrawl API Setup
            1. Go to [FireCrawl](https://firecrawl.dev/)
            2. Create an account or log in
            3. Navigate to API Keys section
            4. Create a new API key

            ### MySQL Database Setup
            Configure your MySQL database with the schema that includes the required tables.

            ### Google Search API Setup
            1. Go to [Google Cloud Console](https://console.cloud.google.com/)
            2. Create a new project or select an existing one
            3. Enable the "Custom Search API" from the API Library
            4. Create API credentials (API key) on the Credentials page
            5. Go to the [Programmable Search Engine](https://programmablesearchengine.google.com/)
            6. Create a new search engine
            7. Set it to search the entire web
            8. Note your Search Engine ID (cx) from the details page

            ### OpenAI API Setup
            1. Go to [OpenAI API](https://platform.openai.com/)
            2. Create an account or log in
            3. Navigate to API Keys section
            4. Create a new API key
            """)

        st.stop()

    # Get API keys from secrets
    google_api_key = st.secrets["config"]["google_api_key"]
    google_cse_id = st.secrets["config"]["google_cse_id"]
    openai_api_key = st.secrets["config"]["openai_api_key"]
    firecrawl_api_key = st.secrets["config"]["firecrawl_api_key"]

    # Tab layout for different search modes
    tab1, tab2, tab3 = st.tabs(["Specification-Based Search", "Direct Keyword Search", "Search History"])

    with tab1:
        if demo_mode:
            # Demo data for testing without MySQL
            specs_data = get_demo_specs_data()
            st.info("Running in demo mode with sample data. Disable demo mode to use actual MySQL data.")
        else:
            # Fetch real specifications from MySQL
            with st.spinner("Loading product specifications from database..."):
                specs_data = get_product_specifications()

            if not specs_data:
                st.error("Failed to retrieve product specifications from database.")
                st.stop()

            st.success(f"Loaded {len(specs_data)} product specifications")

        # Create specification selectors
        st.subheader("Select Product Specifications")

        # Extract unique values for each level
        unique_grupe = sorted([item for item in set(item['grupe'] for item in specs_data) if item is not None])

        # First level - Grupe (Group)
        selected_grupe = st.selectbox("Grupe (Group)", options=[""] + unique_grupe)

        selected_modulis = None
        selected_dalis = None
        selected_spec = None

        if selected_grupe:
            # Filter data based on selected group
            filtered_by_grupe = [item for item in specs_data if item['grupe'] == selected_grupe]

            # Second level - Modulis (Module)
            unique_modulis = sorted(
                [item for item in set(item['modulis'] for item in filtered_by_grupe) if item is not None])
            selected_modulis = st.selectbox("Modulis (Module)", options=[""] + unique_modulis)

            if selected_modulis:
                # Filter further based on selected module
                filtered_by_modulis = [item for item in filtered_by_grupe if item['modulis'] == selected_modulis]

                # Third level - Dalis (Part)
                unique_dalis = sorted(
                    [item for item in set(item['dalis'] for item in filtered_by_modulis) if item is not None])
                selected_dalis = st.selectbox("Dalis (Part)", options=[""] + unique_dalis)

                if selected_dalis:
                    # Filter further based on selected part
                    filtered_by_dalis = [item for item in filtered_by_modulis if item['dalis'] == selected_dalis]

                    # Fourth level - Specifikacija (Specification)
                    unique_spec = sorted(
                        [item for item in set(item['specifikacija'] for item in filtered_by_dalis) if item is not None])
                    selected_spec = st.selectbox("Specifikacija (Specification)", options=[""] + unique_spec)

                    if selected_spec:
                        # Find the spec_id for the selected specification
                        filtered_by_spec = [item for item in filtered_by_dalis if
                                            item['specifikacija'] == selected_spec]
                        if filtered_by_spec:
                            spec_id = filtered_by_spec[0]['spec_id']

                            # Store spec_id in session state
                            st.session_state.spec_id = spec_id

                            # Get specification parameters
                            if demo_mode:
                                # Demo specification parameters
                                spec_params = get_demo_spec_params()
                            else:
                                spec_params = get_specification_parameters(spec_id)

                            # Store specification parameters in session state
                            st.session_state.spec_params = spec_params

                            # Display specification parameters
                            if spec_params:
                                st.subheader("Specification Parameters")
                                params_df = pd.DataFrame(spec_params)
                                st.dataframe(params_df, hide_index=True)
                            else:
                                st.info("No additional specification parameters available.")
                        else:
                            st.warning("Could not find spec_id for the selected specification.")

        # Domain configuration section
        st.subheader("Search Domain Configuration")

        # Domain restriction mode selector
        domain_mode = st.radio(
            "Domain Search Mode:",
            options=["All Lithuanian domains", "Restrict to specific domains", "Include/Exclude specific domains"],
            help="Choose how to handle domain filtering in search results"
        )

        restricted_domains = None
        included_domains = None
        excluded_domains = None

        if domain_mode == "Restrict to specific domains":
            st.info("🔒 **Restricted Mode**: Search will be limited to ONLY the domains you specify below")

            # Initialize restricted domains in session state
            if "restricted_domains" not in st.session_state:
                st.session_state.restricted_domains = []

            # Input for adding new restricted domain
            col1, col2 = st.columns([3, 1])
            with col1:
                new_restricted_domain = st.text_input(
                    "Add domain to whitelist (e.g., varle.lt):",
                    placeholder="Enter domain without http:// or www.",
                    key="new_restricted_domain"
                )
            with col2:
                if st.button("Add Domain") and new_restricted_domain:
                    clean_domain = new_restricted_domain.lower().strip()
                    clean_domain = clean_domain.replace("http://", "").replace("https://", "").replace("www.", "")
                    clean_domain = clean_domain.split("/")[0]

                    if clean_domain not in st.session_state.restricted_domains:
                        st.session_state.restricted_domains.append(clean_domain)
                        st.success(f"Added {clean_domain} to restricted domains")
                    else:
                        st.info(f"{clean_domain} is already in restricted domains")

            # Display current restricted domains
            if st.session_state.restricted_domains:
                st.write("**Restricted to these domains only:**")
                domains_to_remove = []

                cols = st.columns(3)
                for i, domain in enumerate(st.session_state.restricted_domains):
                    col_idx = i % 3
                    with cols[col_idx]:
                        if not st.checkbox(domain, value=True, key=f"restricted_{domain}_{i}"):
                            domains_to_remove.append(domain)

                for domain in domains_to_remove:
                    st.session_state.restricted_domains.remove(domain)

                restricted_domains = st.session_state.restricted_domains
            else:
                st.warning("No domains specified. Please add domains to search.")

        elif domain_mode == "Include/Exclude specific domains":
            st.info("🎯 **Include/Exclude Mode**: Search Lithuanian domains with specific inclusions/exclusions")

            # Initialize domains in session state
            if "included_domains" not in st.session_state:
                st.session_state.included_domains = []
            if "excluded_domains" not in st.session_state:
                st.session_state.excluded_domains = [
                    "ic24.lt", "skrastas.lt", "kauno.diena.lt", "klaipeda.diena.lt", "reidasofficial.lt",
                    "delfi.lt", "15min.lt", "lrytas.lt", "lrt.lt", "vz.lt", "ve.lt", "valstietis.lt",
                    "aidas.lt", "bernardinai.lt", "kurier.lt", "technologijos.lt", "startuplithuania.com",
                    "investlithuania.com", "b2lithuania.com", "ktu.lt", "dainavoszodis.lt",
                    "jonavosnaujienos.lt", "gyvenimas.eu", "gargzdai.lt", "baltictimes.com",
                    "debesyla.lt", "dziaugiuosisavimi.lt", "ieskantysmenulio.lt", "domreg.lt",
                    "rrt.lt", "boreapanda.lt", "lithuania.travel"
                ]

            # Included domains section
            with st.expander("Included Domains (prioritized)", expanded=False):
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_included_domain = st.text_input(
                        "Add domain to prioritize:",
                        placeholder="Enter domain without http:// or www.",
                        key="new_included_domain"
                    )
                with col2:
                    if st.button("Add Included") and new_included_domain:
                        clean_domain = new_included_domain.lower().strip()
                        clean_domain = clean_domain.replace("http://", "").replace("https://", "").replace("www.", "")
                        clean_domain = clean_domain.split("/")[0]

                        if clean_domain not in st.session_state.included_domains:
                            st.session_state.included_domains.append(clean_domain)
                            st.success(f"Added {clean_domain} to included domains")

                if st.session_state.included_domains:
                    st.write("Prioritized domains:")
                    domains_to_remove = []
                    cols = st.columns(3)
                    for i, domain in enumerate(st.session_state.included_domains):
                        col_idx = i % 3
                        with cols[col_idx]:
                            if not st.checkbox(domain, value=True, key=f"included_{domain}_{i}"):
                                domains_to_remove.append(domain)
                    for domain in domains_to_remove:
                        st.session_state.included_domains.remove(domain)

            # Excluded domains section
            with st.expander("Excluded Domains", expanded=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_excluded_domain = st.text_input(
                        "Add domain to exclude:",
                        placeholder="Enter domain without http:// or www.",
                        key="new_excluded_domain"
                    )
                with col2:
                    if st.button("Add Excluded") and new_excluded_domain:
                        clean_domain = new_excluded_domain.lower().strip()
                        clean_domain = clean_domain.replace("http://", "").replace("https://", "").replace("www.", "")
                        clean_domain = clean_domain.split("/")[0]

                        if clean_domain not in st.session_state.excluded_domains:
                            st.session_state.excluded_domains.append(clean_domain)
                            st.success(f"Added {clean_domain} to excluded domains")

                if st.session_state.excluded_domains:
                    st.write("Excluded domains:")
                    domains_to_keep = []
                    cols = st.columns(3)
                    for i, domain in enumerate(st.session_state.excluded_domains):
                        col_idx = i % 3
                        with cols[col_idx]:
                            if st.checkbox(domain, value=True, key=f"excluded_{domain}_{i}"):
                                domains_to_keep.append(domain)
                    st.session_state.excluded_domains = domains_to_keep

            included_domains = st.session_state.included_domains if st.session_state.included_domains else None
            excluded_domains = st.session_state.excluded_domains if st.session_state.excluded_domains else None

        else:  # "All Lithuanian domains"
            st.info("🌐 **All Domains Mode**: Searching all Lithuanian (.lt) domains")

        # Initialize session state variables if they don't exist
        if "search_phrase_generated" not in st.session_state:
            st.session_state.search_phrase_generated = False
        if "search_phrase" not in st.session_state:
            st.session_state.search_phrase = ""

        # Generate search phrase and search if all specifications are selected
        if selected_grupe and selected_modulis and selected_dalis and selected_spec:
            # Store current selections in session state
            if "selected_grupe" not in st.session_state or st.session_state.selected_grupe != selected_grupe:
                st.session_state.selected_grupe = selected_grupe
                st.session_state.search_phrase_generated = False

            if "selected_modulis" not in st.session_state or st.session_state.selected_modulis != selected_modulis:
                st.session_state.selected_modulis = selected_modulis
                st.session_state.search_phrase_generated = False

            if "selected_dalis" not in st.session_state or st.session_state.selected_dalis != selected_dalis:
                st.session_state.selected_dalis = selected_dalis
                st.session_state.search_phrase_generated = False

            if "selected_spec" not in st.session_state or st.session_state.selected_spec != selected_spec:
                st.session_state.selected_spec = selected_spec
                st.session_state.search_phrase_generated = False

            # If phrase is not yet generated or needs regeneration, show generation button
            if not st.session_state.search_phrase_generated:
                if st.button("Generate Search Phrase", type="primary"):
                    with st.spinner("Generating optimized search phrase..."):
                        spec_params = st.session_state.get("spec_params", [])

                        search_phrase_data = generate_search_phrase(
                            selected_grupe,
                            selected_modulis,
                            selected_dalis,
                            selected_spec,
                            spec_params,
                            openai_api_key
                        )

                        if search_phrase_data:
                            st.success("Search phrase generated successfully!")
                            st.session_state.search_phrase = search_phrase_data.search_phrase
                            st.session_state.search_keywords = search_phrase_data.keywords
                            st.session_state.search_phrase_generated = True
                            st.rerun()

            # If phrase is already generated, show it and the search interface
            if st.session_state.search_phrase_generated:
                st.subheader("Generated Search Phrase")
                st.write(st.session_state.search_phrase)

                with st.expander("Keywords Used"):
                    if "search_keywords" in st.session_state:
                        st.write(", ".join(st.session_state.search_keywords))

                # Option to edit the search phrase
                if "edited_search_phrase" not in st.session_state:
                    st.session_state.edited_search_phrase = st.session_state.search_phrase

                edited_phrase = st.text_input(
                    "Edit search phrase if needed:",
                    value=st.session_state.search_phrase,
                    key="edited_search_phrase"
                )

                # Number of results slider
                if "num_results" not in st.session_state:
                    st.session_state.num_results = 10

                num_results = st.slider("Number of results", 1, 10, 10, key="num_results")

                # Search interface
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("Regenerate Phrase"):
                        st.session_state.search_phrase_generated = False
                        st.rerun()

                with col2:
                    if st.button("Search Products", type="primary"):
                        with st.spinner("Searching for products..."):
                            search_results = retrieve_search_results(
                                st.session_state.edited_search_phrase,
                                restricted_domains,
                                included_domains,
                                excluded_domains,
                                google_api_key,
                                google_cse_id,
                                st.session_state.num_results
                            )

                            st.session_state.search_results = search_results
                            st.session_state.search_completed = True

                # Display search results if they exist in session state
                if "search_completed" in st.session_state and st.session_state.search_completed:
                    if "search_results" in st.session_state:
                        search_results = st.session_state.search_results

                        if "error" in search_results:
                            st.error(search_results["error"])
                        else:
                            st.success(f"Search completed: {len(search_results.get('results', []))} results found")

                            search_parameters = {
                                "grupe": st.session_state.selected_grupe,
                                "modulis": st.session_state.selected_modulis,
                                "dalis": st.session_state.selected_dalis,
                                "specification_name": st.session_state.selected_spec,
                                "specification_parameters": st.session_state.spec_params
                            }

                            # Display search parameters
                            with st.expander("Search Parameters", expanded=True):
                                st.write(f"**Group:** {st.session_state.selected_grupe}")
                                st.write(f"**Module:** {st.session_state.selected_modulis}")
                                st.write(f"**Part:** {st.session_state.selected_dalis}")
                                st.write(f"**Specification:** {st.session_state.selected_spec}")

                                if "spec_params" in st.session_state and st.session_state.spec_params:
                                    st.write("**Specification Parameters:**")
                                    for param in st.session_state.spec_params:
                                        st.write(f"- **{param['parametras']}:** {param['reikalavimas parametrui']}")

                                st.write(f"**Search Phrase:** {st.session_state.edited_search_phrase}")

                                if "search_keywords" in st.session_state:
                                    st.write("**Keywords:**")
                                    st.write(", ".join(st.session_state.search_keywords))

                            # Display results
                            display_search_results(search_results, search_parameters,
                                                   openai_api_key, firecrawl_api_key)

    with tab2:
        st.header("Direct Keyword Search")

        # Search form
        with st.form("direct_search_form"):
            # Search query input
            search_query = st.text_input(
                "Search Keywords",
                placeholder="Example: Samsung phone 8GB RAM kaina"
            )

            # Domain mode selector
            domain_mode_direct = st.radio(
                "Domain Search Mode:",
                options=["All Lithuanian domains", "Restrict to specific domains", "Include/Exclude specific domains"],
                help="Choose how to handle domain filtering in search results",
                key="domain_mode_direct"
            )

            if domain_mode_direct == "Restrict to specific domains":
                st.info("🔒 **Restricted Mode**: Search will be limited to ONLY the domains you specify")
                restricted_domains_input = st.text_area(
                    "Restricted Domains (one per line)",
                    placeholder="varle.lt\npigu.lt\nsenukai.lt",
                    help="Enter domains to restrict search to (one per line, without http:// or www.)"
                )
            elif domain_mode_direct == "Include/Exclude specific domains":
                st.info("🎯 **Include/Exclude Mode**: Search Lithuanian domains with specific inclusions/exclusions")
                included_domains_input = st.text_area(
                    "Included Domains (one per line)",
                    placeholder="varle.lt\npigu.lt\nsenukai.lt",
                    help="Enter domains to prioritize (one per line, without http:// or www.)"
                )
                excluded_domains_input = st.text_area(
                    "Excluded Domains (one per line)",
                    value="delfi.lt\n15min.lt\nlrytas.lt\nlrt.lt",
                    help="Enter domains to exclude (one per line, without http:// or www.)"
                )
            else:
                st.info("🌐 **All Domains Mode**: Searching all Lithuanian (.lt) domains")

            num_results = st.slider("Number of results", 5, 30, 10)
            submit_button = st.form_submit_button("Search", type="primary")

        # Execute search when form is submitted
        if submit_button and search_query:
            restricted_domains = None
            included_domains = None
            excluded_domains = None

            if domain_mode_direct == "Restrict to specific domains" and restricted_domains_input:
                restricted_domains = []
                for line in restricted_domains_input.split('\n'):
                    domain = line.strip().lower()
                    if domain:
                        domain = domain.replace("http://", "").replace("https://", "").replace("www.", "")
                        domain = domain.split("/")[0]
                        restricted_domains.append(domain)

            elif domain_mode_direct == "Include/Exclude specific domains":
                if included_domains_input:
                    included_domains = []
                    for line in included_domains_input.split('\n'):
                        domain = line.strip().lower()
                        if domain:
                            domain = domain.replace("http://", "").replace("https://", "").replace("www.", "")
                            domain = domain.split("/")[0]
                            included_domains.append(domain)

                if excluded_domains_input:
                    excluded_domains = []
                    for line in excluded_domains_input.split('\n'):
                        domain = line.strip().lower()
                        if domain:
                            domain = domain.replace("http://", "").replace("https://", "").replace("www.", "")
                            domain = domain.split("/")[0]
                            excluded_domains.append(domain)

            with st.spinner("Searching..."):
                search_results = retrieve_search_results(
                    search_query,
                    restricted_domains,
                    included_domains,
                    excluded_domains,
                    google_api_key,
                    google_cse_id,
                    num_results
                )

                if "error" in search_results:
                    st.error(search_results["error"])
                else:
                    st.success(f"Search completed: {len(search_results.get('results', []))} results found")

                    # Store in search history
                    if "search_history" not in st.session_state:
                        st.session_state.search_history = []

                    history_entry = {
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "query": search_query,
                        "results": search_results.get('results', [])
                    }
                    st.session_state.search_history.append(history_entry)

                    # Display results with new selection-based analysis
                    display_search_results(search_results, search_query, openai_api_key, firecrawl_api_key)

    with tab3:
        st.header("Search History")

        if "search_history" not in st.session_state or not st.session_state.search_history:
            st.info("No search history yet. Search for products to see your history here.")
        else:
            for i, entry in enumerate(reversed(st.session_state.search_history)):
                with st.expander(f"{entry['timestamp']} - {entry['query']}"):
                    st.write(f"**Search Query:** {entry['query']}")
                    st.write(f"**Results Found:** {len(entry['results'])}")

                    # Display results summaries
                    for j, result in enumerate(entry['results']):
                        st.write(f"**{j + 1}. [{result['title']}]({result['url']})**")
                        st.write(f"Domain: {result['domain']}")
                        st.divider()


def display_search_results(search_results, search_parameters, openai_api_key=None, firecrawl_api_key=None):
    """Display search results with optional price analysis."""
    st.subheader("Search Results")

    if search_results.get('results'):
        # First, display all search results and let user select which ones to analyze
        st.write("### 📋 Search Results Overview")
        st.write(f"Found {len(search_results['results'])} results. Select which ones you want to analyze for prices:")

        # Initialize selected results in session state
        if "selected_results" not in st.session_state:
            st.session_state.selected_results = []

        # Display results with selection checkboxes
        selected_results = []
        for i, result in enumerate(search_results['results']):
            col1, col2 = st.columns([1, 4])

            with col1:
                is_selected = st.checkbox(
                    f"Select {i + 1}",
                    key=f"select_result_{i}",
                    help="Check to include this result in price analysis"
                )
                if is_selected:
                    selected_results.append((i, result))

            with col2:
                st.write(f"**{i + 1}. [{result['title']}]({result['url']})**")
                st.write(f"**Domain:** {result['domain']}")
                st.write(f"**URL:** {result['url']}")
                st.write(f"**Description:** {result['snippet']}")
                st.divider()

        # Store selected results in session state
        st.session_state.selected_results = selected_results

        # Show analysis section only if there are selected results and API keys are available
        if selected_results and openai_api_key and firecrawl_api_key:
            st.write("### 🔍 Price Analysis")
            st.write(f"You have selected {len(selected_results)} results for analysis.")

            # Add aggregator selection checkboxes
            st.write("**Configure Analysis Settings:**")
            aggregator_settings = {}

            for i, (result_idx, result) in enumerate(selected_results):
                col1, col2 = st.columns([1, 3])
                with col1:
                    is_aggregator = st.checkbox(
                        f"Aggregator",
                        key=f"aggregator_{result_idx}",
                        help="Check if this page is a price comparison/aggregator site"
                    )
                    aggregator_settings[result_idx] = is_aggregator
                with col2:
                    st.write(f"**{result['title']}** - {result['domain']}")

            # Analysis button
            if st.button("🚀 Start Price Analysis", type="primary"):
                # Initialize progress tracking for price analysis
                price_analyses = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                st.info("🔍 Analyzing prices for selected products... This may take a few minutes.")

                for i, (result_idx, result) in enumerate(selected_results):
                    # Update progress status
                    progress_value = (i + 1) / len(selected_results)
                    progress_bar.progress(progress_value)
                    status_text.text(f"🔍 Analyzing result {i + 1}/{len(selected_results)}")

                    try:
                        with st.spinner(f"Analyzing {result['title']}..."):
                            is_aggregator = aggregator_settings.get(result_idx, False)
                            price_analysis = analyze_product_url(
                                result['url'],
                                search_parameters,
                                openai_api_key,
                                firecrawl_api_key,
                                is_aggregator_page=is_aggregator
                            )
                            if price_analysis:
                                price_analyses[result['url']] = price_analysis
                    except Exception as e:
                        st.error(f"Error analyzing {result['title']}: {str(e)}")

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Price analysis completed!")

                # Display analysis results
                st.write("### 💰 Analysis Results")
                for result_idx, result in selected_results:
                    if result['url'] in price_analyses:
                        analysis_list = price_analyses[result['url']]

                        with st.expander(f"📊 Analysis: {result['title']}", expanded=True):
                            if analysis_list and len(analysis_list) > 0:
                                # Display each product found
                                for j, analysis in enumerate(analysis_list):
                                    if j > 0:
                                        st.divider()

                                    col1, col2 = st.columns([2, 1])

                                    with col1:
                                        st.write(f"**Product {j + 1}:** {analysis['product_name']}")

                                        # Price information
                                        if 'product_price' in analysis:
                                            st.write(f"**💰 Price:** {analysis['product_price']}")
                                        if 'price_per_' in analysis and analysis['price_per_']:
                                            st.write(f"**📏 Price per:** {analysis['price_per_']}")

                                        # Display product properties
                                        if 'product_properties' in analysis and analysis['product_properties']:
                                            st.write("**📋 Properties:**")
                                            st.text(analysis['product_properties'])

                                        # Display evaluation/judgment
                                        if 'evaluation' in analysis:
                                            st.write("**✅ Evaluation:**")
                                            st.markdown(analysis['evaluation'])

                                    with col2:
                                        # Display provider information
                                        if 'provider' in analysis:
                                            st.write(f"**🏪 Provider:** {analysis['provider']}")
                                            if 'provider_website' in analysis:
                                                st.write(f"**🌐 Website:** {analysis.get('provider_website', 'N/A')}")
                                            if 'provider_url' in analysis:
                                                st.write(
                                                    f"**🔗 URL:** [{analysis.get('provider_url', 'N/A')}]({analysis.get('provider_url', '#')})")
                            else:
                                st.info("No products found matching the specifications on this page.")
                    else:
                        with st.expander(f"❌ Failed: {result['title']}", expanded=False):
                            st.error("Analysis failed for this URL")

                # Export option
                if st.button("📥 Export Analysis Results as JSON"):
                    export_data = []
                    for result_idx, result in selected_results:
                        result_data = result.copy()
                        if result['url'] in price_analyses:
                            result_data['price_analysis'] = price_analyses[result['url']]
                        export_data.append(result_data)
                    st.json(export_data)

        elif selected_results and (not openai_api_key or not firecrawl_api_key):
            st.warning("⚠️ Price analysis requires both OpenAI and FireCrawl API keys to be configured.")
        elif not selected_results:
            st.info("ℹ️ Select some results above to enable price analysis.")
    else:
        st.warning("No results found. Try modifying your search terms.")


if __name__ == "__main__":
    main()