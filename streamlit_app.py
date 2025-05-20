import streamlit as st
from googleapiclient.discovery import build
from openai import OpenAI
from pydantic import BaseModel, Field
import pandas as pd
import re
import json
from typing import List, Dict, Optional, Union, Any

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


class SearchPhrase(BaseModel):
    """Model for structured search phrase output from OpenAI."""
    search_phrase: str = Field(description="Optimized search phrase for finding products")
    keywords: list[str] = Field(description="Individual keywords extracted from specifications")


class ProductPrice(BaseModel):
    """Model for structured product price analysis output from OpenAI."""
    product_name: str = Field(description="Complete name of the product")
    regular_price: Optional[float] = Field(None, description="Regular/Current price in EUR")
    sale_price: Optional[float] = Field(None, description="Sale price if available in EUR")
    price_per_unit: Optional[float] = Field(None, description="Price per unit if applicable")
    unit_type: Optional[str] = Field(None, description="Unit type (kg, liter, piece, etc.)")
    availability: str = Field(description="Product availability status (in stock, out of stock, etc.)")
    price_judgment: str = Field(description="Assessment of price competitiveness in the market")
    provider: str = Field(description="Name of the store/website selling the product")
    last_updated: Optional[str] = Field(None, description="When the price was last updated if available")


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


def analyze_product_url(url, search_parameters, openai_api_key):
    """Analyze a product page URL to extract price information using OpenAI."""
    client = OpenAI(api_key=openai_api_key)

    category = search_parameters.get("grupe", "")
    subcategory = search_parameters.get("modulis", "")
    product_type = search_parameters.get("dalis", "")
    specification_name = search_parameters.get("specification_name", "")

    # Format the tech_spec from specification_parameters
    tech_spec = ""
    for param in search_parameters.get("specification_parameters", []):
        tech_spec += f"- {param.get('parametras', '')}: {param.get('reikalavimas parametrui', '')}\n"

    # Construct the prompt with the formatted parameters
    prompt = f"""
        Analyze the given webpage URL={url} for {category} {subcategory} {product_type} and gather detailed product information according to the following:
                     product type: {specification_name}
                     product specification:
    {tech_spec}

        2. Verify the product is currently available for purchase
        3. Gather accurate pricing in EUR
        4. Evaluate technical specification requirements one by one

        Calculate and include price per unit for each product
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
            "product_properties": {
              "key_spec1": "value1",
              "key_spec2": "value2"
            },
            "product_sku": "Any product identifiers (SKU, UPC, model number)",
            "product_price": 299.99,
        """

    try:
        response = client.responses.parse(
            model="gpt-4.1",
            tools=[{
                # "type": 'asdf',
                "type": "web_search_preview",
                "user_location": {
                    "type": "approximate",
                    "country": "LT",
                    "city": "Vilnius",
                }
            }],
            temperature=0.2,
            input=prompt,
            text_format=ProductList,
        )

        # Process the response exactly as in the working example
        products_json = []
        for product in response.output_parsed.products:
            products_json.append(product.model_dump())

        return products_json

    except Exception as e:
        st.error(f"Error analyzing product URL: {e}")
        return None


def retrieve_search_results(query, required_domains=None, google_api_key=None, google_cse_id=None, num_results=10):
    """
    Retrieve search results using Google Custom Search API with Lithuanian domain filtering and PDF exclusion.

    Args:
        query (str): Search query
        required_domains (list): List of domains that must be included in the search
        google_api_key (str): Google API key
        google_cse_id (str): Google Custom Search Engine ID
        num_results (int): Number of results to retrieve

    Returns:
        dict: Search results or error information
    """
    try:
        # Initialize the Custom Search API service
        service = build("customsearch", "v1", developerKey=google_api_key)

        # Add Lithuanian domain and PDF exclusion to the query
        modified_query = f"{query} site:.lt -filetype:pdf"

        # Handle required domains if specified - but keep the query simpler
        # Instead of trying to combine with OR operators, we'll just run the search
        # with the Lithuanian filter and then prioritize results from required domains

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

                # If we have required domains, mark this result as "required" if it matches
                is_required_domain = False
                if required_domains:
                    for req_domain in required_domains:
                        if req_domain in domain:
                            is_required_domain = True
                            break

                search_results.append({
                    "title": item.get("title", "No title"),
                    "url": url,
                    "snippet": item.get("snippet", "No description"),
                    "domain": domain,
                    "is_required_domain": is_required_domain
                })

        # Sort results to prioritize required domains if specified
        if required_domains:
            search_results.sort(key=lambda x: not x.get("is_required_domain", False))

        return {"results": search_results, "query": modified_query}

    except Exception as e:
        return {"error": f"Google Search API request failed: {str(e)}"}


# Streamlit App
def main():
    st.title("Lithuanian Product Search")
    st.markdown("Find and analyze product prices in the Lithuanian market")

    # Set demo mode once at the beginning
    demo_mode = st.sidebar.checkbox("Demo Mode (No MySQL)", value=True, key="demo_mode_checkbox")

    # Check for API keys in secrets
    missing_keys = []
    if not st.secrets.get("config", {}).get("google_api_key"):
        missing_keys.append("google_api_key")
    if not st.secrets.get("config", {}).get("google_cse_id"):
        missing_keys.append("google_cse_id")
    if not st.secrets.get("config", {}).get("openai_api_key"):
        missing_keys.append("openai_api_key")

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

        [mysql]
        host = "YOUR_MYSQL_HOST"
        user = "YOUR_MYSQL_USERNAME"
        password = "YOUR_MYSQL_PASSWORD"
        database = "YOUR_MYSQL_DATABASE"
        ```
        """)

        with st.expander("How to set up required APIs"):
            st.markdown("""
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

        # Let users add required domains
        st.markdown("#### Required Domains")
        st.markdown("These domains will be included in the search (but search won't be limited to only these)")

        # Initialize required domains in session state if not present
        if "required_domains" not in st.session_state:
            st.session_state.required_domains = []

        # Input for adding new required domain
        col1, col2 = st.columns([3, 1])
        with col1:
            new_required_domain = st.text_input(
                "Add required domain (e.g., example.lt):",
                placeholder="Enter a domain name without http:// or www."
            )
        with col2:
            if st.button("Add Domain") and new_required_domain:
                # Clean up domain (remove http://, www., trailing slashes)
                clean_domain = new_required_domain.lower().strip()
                clean_domain = clean_domain.replace("http://", "").replace("https://", "").replace("www.", "")
                clean_domain = clean_domain.split("/")[0]  # Remove any path

                if clean_domain not in st.session_state.required_domains:
                    st.session_state.required_domains.append(clean_domain)
                    st.success(f"Added {clean_domain} to required domains")
                else:
                    st.info(f"{clean_domain} is already in required domains")

        # Display current required domains
        if st.session_state.required_domains:
            st.write("Current required domains:")
            domains_to_remove = []

            # Ensure we have unique domains before creating checkboxes
            unique_domains = list(dict.fromkeys(st.session_state.required_domains))

            # Update session state to contain only unique domains
            st.session_state.required_domains = unique_domains

            # Create columns for domain display
            cols = st.columns(3)
            for i, domain in enumerate(unique_domains):
                col_idx = i % 3
                with cols[col_idx]:
                    # Generate a truly unique key for each checkbox
                    unique_key = f"required_domain_{domain}_{i}_{id(domain)}"
                    if not st.checkbox(domain, value=True, key=unique_key):
                        domains_to_remove.append(domain)

            # Remove unchecked domains
            for domain in domains_to_remove:
                st.session_state.required_domains.remove(domain)
        else:
            st.info("No required domains added. Search will include all Lithuanian (.lt) domains by default.")

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
                # Reset downstream selections when group changes
                st.session_state.search_phrase_generated = False

            if "selected_modulis" not in st.session_state or st.session_state.selected_modulis != selected_modulis:
                st.session_state.selected_modulis = selected_modulis
                # Reset downstream selections when module changes
                st.session_state.search_phrase_generated = False

            if "selected_dalis" not in st.session_state or st.session_state.selected_dalis != selected_dalis:
                st.session_state.selected_dalis = selected_dalis
                # Reset downstream selections when part changes
                st.session_state.search_phrase_generated = False

            if "selected_spec" not in st.session_state or st.session_state.selected_spec != selected_spec:
                st.session_state.selected_spec = selected_spec
                # Reset search phrase when specification changes
                st.session_state.search_phrase_generated = False

            # If phrase is not yet generated or needs regeneration, show generation button
            if not st.session_state.search_phrase_generated:
                if st.button("Generate Search Phrase", type="primary"):
                    with st.spinner("Generating optimized search phrase..."):
                        # Get specification parameters from session state
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
                            # Store in session state
                            st.session_state.search_phrase = search_phrase_data.search_phrase
                            st.session_state.search_keywords = search_phrase_data.keywords
                            st.session_state.search_phrase_generated = True
                            # Force a rerun to update UI
                            st.rerun()

            # If phrase is already generated, show it and the search interface
            if st.session_state.search_phrase_generated:
                st.subheader("Generated Search Phrase")
                st.write(st.session_state.search_phrase)

                with st.expander("Keywords Used"):
                    if "search_keywords" in st.session_state:
                        st.write(", ".join(st.session_state.search_keywords))

                # Initialize the edited_search_phrase in session state if not present
                if "edited_search_phrase" not in st.session_state:
                    st.session_state.edited_search_phrase = st.session_state.search_phrase

                # Option to edit the search phrase
                edited_phrase = st.text_input(
                    "Edit search phrase if needed:",
                    value=st.session_state.search_phrase,
                    key="edited_search_phrase"
                )

                # Initialize num_results in session state if not present
                if "num_results" not in st.session_state:
                    st.session_state.num_results = 10

                # Number of results slider
                num_results = st.slider("Number of results", 5, 30, 10, key="num_results")

                # Option to regenerate search phrase
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("Regenerate Phrase"):
                        st.session_state.search_phrase_generated = False
                        st.rerun()

                with col2:
                    if st.button("Search Products", type="primary"):
                        with st.spinner("Searching for products..."):
                            # Use edited phrase for search
                            search_results = retrieve_search_results(
                                st.session_state.edited_search_phrase,
                                st.session_state.required_domains if st.session_state.required_domains else None,
                                google_api_key,
                                google_cse_id,
                                st.session_state.num_results
                            )

                            # Store search results in session state
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
                            st.write(search_parameters)
                            # Display search parameters
                            with st.expander("Search Parameters", expanded=True):
                                st.write(f"**Group:** {st.session_state.selected_grupe}")
                                st.write(f"**Module:** {st.session_state.selected_modulis}")
                                st.write(f"**Part:** {st.session_state.selected_dalis}")
                                st.write(f"**Specification:** {st.session_state.selected_spec}")

                                # Display specification parameters if available
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
                                                   openai_api_key)

    with tab2:
        st.header("Direct Keyword Search")

        # Search form
        with st.form("direct_search_form"):
            # Search query input
            search_query = st.text_input(
                "Search Keywords",
                placeholder="Example: Samsung phone 8GB RAM kaina"
            )

            # Required domains section
            st.subheader("Domain Configuration")

            # Multi-line input for required domains
            required_domains_input = st.text_area(
                "Required Domains (one per line)",
                placeholder="varle.lt\npigu.lt\nsenukai.lt",
                help="Enter domains that must be included in the search (one per line, without http:// or www.)"
            )

            num_results = st.slider("Number of results", 5, 30, 10)

            # Analyze prices with OpenAI option
            analyze_prices = st.checkbox("Analyze prices with OpenAI", value=True,
                                         key="analyze_prices_checkbox",
                                         help="This will use OpenAI to analyze each product page for detailed price information")

            submit_button = st.form_submit_button("Search", type="primary")

        # Execute search when form is submitted
        if submit_button and search_query:
            # Process required domains
            required_domains = []
            if required_domains_input:
                for line in required_domains_input.split('\n'):
                    domain = line.strip().lower()
                    if domain:
                        # Clean up domain
                        domain = domain.replace("http://", "").replace("https://", "").replace("www.", "")
                        domain = domain.split("/")[0]  # Remove any path
                        required_domains.append(domain)

            with st.spinner("Searching..."):
                search_results = retrieve_search_results(
                    search_query,
                    required_domains if required_domains else None,
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

                    # Display results
                    display_search_results(search_results, search_query, openai_api_key if analyze_prices else None)

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


def display_search_results(search_results, search_parameters, openai_api_key=None):
    """Display search results with optional price analysis."""
    st.subheader("Search Results")

    if search_results.get('results'):
        # Initialize progress tracking for price analysis
        price_analyses = {}
        if openai_api_key:
            progress_bar = st.progress(0)
            status_text = st.empty()
            st.info("Analyzing prices for each product... This may take a minute.")

        for i, result in enumerate(search_results['results']):
            with st.expander(f"{i + 1}. {result['title']}"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**URL:** [{result['url']}]({result['url']})")
                    st.write(f"**Domain:** {result['domain']}")
                    st.write(f"**Description:** {result['snippet']}")

                # If OpenAI API key is provided, analyze product price
                if openai_api_key:
                    with col2:
                        # Update progress status
                        progress_value = (i + 1) / len(search_results['results'])
                        progress_bar.progress(progress_value)
                        status_text.text(f"Analyzing product {i + 1}/{len(search_results['results'])}")

                        # Check if we've already analyzed this URL
                        if result['url'] not in price_analyses:
                            try:
                                with st.spinner(f"Analyzing price for product {i + 1}..."):
                                    price_analysis = analyze_product_url(result['url'], search_parameters,
                                                                         openai_api_key)
                                    if price_analysis:
                                        price_analyses[result['url']] = price_analysis
                            except Exception as e:
                                st.error(f"Error analyzing product: {str(e)}")

                        # Display price analysis if available
                        if result['url'] in price_analyses:
                            analysis_list = price_analyses[result['url']]
                            if analysis_list and len(analysis_list) > 0:  # Check if the list is not empty
                                # Take the first product from the list
                                analysis = analysis_list[0]
                                st.write("### Price Analysis")
                                st.write(f"**Product:** {analysis['product_name']}")

                                # Price information
                                if 'product_price' in analysis:
                                    st.write(f"**Price:** {analysis['product_price']}")
                                if 'price_per_' in analysis and analysis['price_per_']:
                                    st.write(f"**Price per:** {analysis['price_per_']}")

                                # Display product properties
                                if 'product_properties' in analysis and analysis['product_properties']:
                                    st.write("**Properties:**")
                                    st.text(analysis['product_properties'])

                                # Display evaluation/judgment
                                if 'evaluation' in analysis:
                                    st.markdown(f"**Evaluation:** {analysis['evaluation']}")

                                # Display provider information
                                if 'provider' in analysis:
                                    st.write(f"**Provider:** {analysis['provider']} ({analysis['provider_website']})")

                                # Display more products if available
                                if len(analysis_list) > 1:
                                    show_more = st.checkbox(f"View {len(analysis_list) - 1} more similar products",
                                                            key=f"more_products_{i}")
                                    if show_more:
                                        for j, product in enumerate(analysis_list[1:], 1):
                                            st.write(f"**Alternative {j}: {product.get('product_name', 'N/A')}**")
                                            st.write(f"**Price:** {product.get('product_price', 'N/A')}")
                                            if 'product_properties' in product and product['product_properties']:
                                                st.write("**Properties:**")
                                                st.text(product['product_properties'])
                                            if 'evaluation' in product:
                                                st.write(f"**Evaluation:** {product['evaluation']}")
                                            st.write(
                                                f"**Provider:** {product.get('provider', 'N/A')} ({product.get('provider_website', 'N/A')})")
                                            st.divider()

        # Clear progress indicators once done
        if openai_api_key:
            progress_bar.empty()
            status_text.empty()
            st.success("Price analysis completed!")

        # Export option
        if st.button("Export Results as JSON"):
            # Include price analyses in export if available
            if openai_api_key and price_analyses:
                export_data = []
                for result in search_results['results']:
                    result_data = result.copy()
                    if result['url'] in price_analyses:
                        result_data['price_analysis'] = price_analyses[result['url']]
                    export_data.append(result_data)
                st.json(export_data)
            else:
                st.json(search_results['results'])
    else:
        st.warning("No results found. Try modifying your search terms.")

if __name__ == "__main__":
    main()