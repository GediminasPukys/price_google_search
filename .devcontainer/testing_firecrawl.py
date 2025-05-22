import streamlit as st
from googleapiclient.discovery import build
from openai import OpenAI
from pydantic import BaseModel, Field
import pandas as pd
import re
import json
from typing import List, Dict, Optional, Union, Any

from firecrawl import FirecrawlApp

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


def get_urls(client, category, subcategory, product_type, specification_name, scrape_result):
    prompt = f"""In a given content of price aggregator website find relevant products and corresponding urls.
    Relevant product is un-strictly defined by: category: {category} {subcategory} {product_type} and product type: {specification_name}
    content: {scrape_result}
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

def is_aggregator(client, scrape_result):
    prompt = f""" Does this content appear to be from a product price aggregator website (e.g., a site that lists links to products from other retailers or websites, such as a price comparison site), or is it a direct product listing page (e.g., an online store page where products are listed with their own descriptions and prices)?
    If it's an aggregator with links to actual products.
    If it's a product listing page or landing page.
    content: {scrape_result}
    """
    response = client.responses.parse(
        model="gpt-4o",
        input=prompt,
        text_format=PageJudge,
    )
    return response.output_parsed.is_aggregator




openai_api_key = st.secrets["config"]["openai_api_key"]
google_ai_api_key = st.secrets["config"]["google_ai_api_key"]
fc_ai_api_key = st.secrets["config"]["fc_ai_api_key"]
client = OpenAI(api_key=openai_api_key)

search_parameters = {"grupe":"Telefonai ir ryšio paslaugos","modulis":"Mobilieji telefonai (DPS)","dalis":"(MT14)Aukštesnio našumo išmanusis telefonas","specification_name":"(MT14)Aukštesnio našumo išmanusis telefonas","specification_parameters":[{"parametras":"(MT14)Aukštesnio našumo išmanusis telefonas","reikalavimas parametrui":"Prekės pavadinimas, gamintojas, prekės kodas"},{"parametras":"Operatyvinės ir vidinės atminties talpa","reikalavimas parametrui":"ne mažiau nei 8 GB operatyvinės atminties ir ne mažiau nei 256 GB vidinės atminties"},{"parametras":"Minimaliai turi būti palaikomi duomenų perdavimo standartai","reikalavimas parametrui":"minimaliai turi būti palaikomi GPRS, LTE, 5G"},{"parametras":"Ekrano įstrižainė","reikalavimas parametrui":"ne mažiau kaip 5,5\" ir ne daugiau kaip 6,8\""},{"parametras":"Ekrano skiriamoji geba","reikalavimas parametrui":"ne mažiau kaip 1000x2300 taškų (angl. pixel)"},{"parametras":"Ekrano tipas","reikalavimas parametrui":"lietimui jautrus"},{"parametras":"Ekrano apsauga","reikalavimas parametrui":"Corning Gorilla Glass ne žemiau Victus versijos arba lygiavertė ekrano apsauga"},{"parametras":"Klaviatūra","reikalavimas parametrui":"Integruota ekrane"},{"parametras":"SIM kortelių lizdų skaičius","reikalavimas parametrui":"ne mažiau nei 2 vnt. (ne mažiau nei  viena SIM kortelė turi būti fizinė)"},{"parametras":"Vidinė WLAN tinklo plokštė","reikalavimas parametrui":"IEEE 802.11 ax, įrenginys ir antena integruoti į korpusą"},{"parametras":"Vidinis Bluetooth įrenginys","reikalavimas parametrui":"ne žemesnė nei 5.0 versija, įrenginys ir antena integruoti į korpusą"},{"parametras":"GPS","reikalavimas parametrui":"GPS ar lygiavertis"},{"parametras":"Integruotas piršto antspaudo skaitytuvas (angl. Fingerprint reader)","reikalavimas parametrui":"taip"},{"parametras":"Vidiniai integruoti mikrofonas ir garsiakalbis garso atkūrimui","reikalavimas parametrui":"taip"},{"parametras":"Fotografavimas ir fillmavimas","reikalavimas parametrui":"ne mažiau 3-jų skirtingų kamerų, skirtų fotografuoti ir filmuoti; ne mažiau kaip 1 kamera turi būti telefono korpuso priekyje. Ne mažiau kaip vienos galinės kameros matricos dydis turi būti ne mažiau 50 Mpx (megapikseliai); ne mažiau kaip vienos priekinės kameros matricos dydis turi būti ne mažiau 12 Mpx (megapikseliai). Ne mažiau kaip  vienos kameros filmavimo raiška turi būti ne mažesnė nei 4K prie ne mažiau nei 30 kadrų/sek. Turi būti integruota blykstė."},{"parametras":"Kitos funkcijos","reikalavimas parametrui":"NFC (angl. Near Field Communication) arba lygiaviartė funkcija"},{"parametras":"Apsauga nuo dulkių ir vandens poveikio","reikalavimas parametrui":"ne mažiau nei IP68 pagal LST EN 60529 arba lygiavertė"},{"parametras":"Operacinė sistema","reikalavimas parametrui":"privalo turėti galimybę įdiegti programėles (Apps) ir turi palaikyti bent vieną iš šių programėlių (Apps) parduotuvių: Google Play Store, Apple App store, Micrososft Store"},{"parametras":"Programinė įranga","reikalavimas parametrui":"Interneto naršyklė, elektroninio pašto programa, nuotraukų bei paveikslų peržiūros programa, vaizdo bylų grotuvas, garso bylų grotuvas, elektroninė užrašų knygelė, kalendorius, kontaktinės informacijos programa. Turi būti įdiegtos programėlės (Apps), palaikančios ir leidžiančios redaguoti DOC, XLS, PPT ir peržiūrėti PDF formatus"},{"parametras":"Operacinės sistemos atnaujinimai ir saugumo pataisos","reikalavimas parametrui":"Ne mažiau kaip 1 operacinės sistemos atnaujinimas ir ne mažiau kaip 4 metų saugumo pataisų teikimo nuo telefono išleidimo į rinką datos."},{"parametras":"Išorinė standartinė USB C jungtis","reikalavimas parametrui":"ne mažiau nei 1 vnt."},{"parametras":"Baterijos talpa","reikalavimas parametrui":"ne mažiau nei 4500 mAh"},{"parametras":"Baterijos įkrovimo funkcija","reikalavimas parametrui":"ne mažiau 30W, greito įkrovimo fukncija (angl. fast charging)"},{"parametras":"Minimali įrangos komplektacija","reikalavimas parametrui":"Originali ir standartinė oficialaus gamintojo teikiama komplektacija, taikoma Lietuvos Respublikos rinkai. Esant poreikiui, perkančioji organizacija turi teisę pareikalauti įrodymų, kad siūlomi telefonai skirti Lietuvos Respublikos rinkai ir kokios komplektacijos jie yra teikiami šiai rinkai."},{"parametras":"Telefono išleidimo į rinką metai","reikalavimas parametrui":"Telefono išleidimo į rinką data turi būti ne ankstesnė nei 2023 metai"},{"parametras":"Surinkimo reikalavimai","reikalavimas parametrui":"visa įranga turi būti gamykliškai nauja „brand new“. Negalima siūlyti gamykliškai atnaujintos arba naudotos („renew“/„refurbished“/„remarketed“) įrangos."},{"parametras":"Telefonas turi atitikti mobiliems telefonams keliamus aplinkos apsaugos kriterijus, patvirtintus Lietuvos Respublikos aplinkos ministro 2022 m. gruodžio 13 d. įsakymu Nr. DI-401 „Dėl aplinkos apsaugos kriterijų taikymo, vykdant žaliuosius pirkimus tvarkos aprašo patvirtinimo\".","reikalavimas parametrui":"taip"},{"parametras":"Garantija telefonui (įskaitant bateriją) ne mažiau nei 2 metai. Jei garantinio laikotarpio metu sugedusios prekės darbingumo atkūrimo trukmė bus ilgesnė nei 5 darbo dienos (neįskaitant telefono siuntimo laiko), darbingumo atkūrimo laikotarpiui tiekėjas turi pakeisti sugedusią prekę kita, ne prastesnių parametrų preke.","reikalavimas parametrui":"taip"},{"parametras":"Veikimo dažniai - turi užtikrinti balso ir duomenų perdavimą Lietuvos Respublikos teritorijoje pagal atitinkamų balso ir duomenų perdavimo standartų reikalavimus","reikalavimas parametrui":"Taip"}]}


category = search_parameters.get("grupe", "")
subcategory = search_parameters.get("modulis", "")
product_type = search_parameters.get("dalis", "")
specification_name = search_parameters.get("specification_name", "")

# Format the tech_spec from specification_parameters
tech_spec = ""
for param in search_parameters.get("specification_parameters", []):
    tech_spec += f"- {param.get('parametras', '')}: {param.get('reikalavimas parametrui', '')}\n"

url = 'https://www.kaina24.lt/s/ip68-telefonas/'
url = 'https://www.kainos.lt/paieska/xiaomi-redmi-note-14-pro-5g-256gb-kaina'


app = FirecrawlApp(api_key=fc_ai_api_key)

# Scrape a website:

url='https://www.saurida.lt/kuro-kainos-degalinese/'
scrape_result = app.scrape_url(url, formats=['markdown'])


print(scrape_result)


def get_prompt(category, subcategory, product_type, specification_name, scrape_result):
    prompt = f"""
        analyse given product webpage content:
        content: {scrape_result}

        next step:
        find all products in it and gather detailed product information.
         only then judge products do they meet provided specifications:
                            category: {category} {subcategory} {product_type} 
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
                    "price_per_unit": 9.99,
                    "evaluation": "Detailed assessment of how the product meets or fails each technical specification"
                  }
                ]

                DO NOT include any explanation, preamble, or additional text - ONLY provide the JSON array.
                """
    return prompt


def get_prices_from_url(url, category, subcategory, product_type, specification_name):
    app = FirecrawlApp(api_key=fc_ai_api_key)

    # Scrape a website:
    scrape_result = app.scrape_url(url, formats=['markdown'])
    print(scrape_result)
    product_list = []
    is_aggregator_v = is_aggregator(client, scrape_result)
    if (is_aggregator_v):
        urls_list = get_urls(client, category, subcategory, product_type, specification_name, scrape_result)
        print(urls_list)

        for url_0 in urls_list:
            print(url_0["product_url"])
            scrape_result = app.scrape_url(url_0["product_url"], formats=['markdown'])
            prompt = get_prompt(category, subcategory, product_type, specification_name, scrape_result)
            response = client.responses.parse(
                model="gpt-4o",
                input=prompt,
                text_format=ProductList,
            )
            product_list = product_list + response.output_parsed.products
        return product_list
    prompt = get_prompt(category, subcategory, product_type, specification_name, scrape_result)
    response = client.responses.parse(
        model="gpt-4o",
        input=prompt,
        text_format=ProductList,
    )

    return response.output_parsed.products




