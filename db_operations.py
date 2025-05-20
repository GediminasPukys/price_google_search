
import streamlit as st
import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Optional, Any

def connect_to_database():
    """Connect to MySQL database using credentials from secrets."""
    try:
        connection = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            database=st.secrets["mysql"]["database"]
        )
        if connection.is_connected():
            return connection
    except Error as e:
        st.error(f"Error connecting to MySQL database: {e}")
        return None


def get_product_specifications():
    """Retrieve product specifications from MySQL database."""
    connection = connect_to_database()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            
            query = """
            SELECT cat2.category_group                            AS 'grupe',
                   c.id,
                   cat2.category_name                             AS modulis,
                   ccp.part_id,
                   cat.category_name                              AS 'dalis',
                   pc.class_id                                    AS 'spec_id',
                   IF(pc.class_name = "", pr.name, pc.class_name) AS 'specifikacija',
                   pr.param_id
            FROM jos_catalog_product_class pc
                     JOIN jos_catalog_params pr ON pr.class_id = pc.class_id AND (pr.param_id = '0' OR pr.param_id = '1')
                     JOIN jos_catalog_category_xref cx ON cx.category_child_id = pc.class_parent_id
                     JOIN jos_catalog_category cat ON cat.category_id = cx.category_child_id
                     JOIN jos_catalog_category cat2 ON cat2.category_id = cx.category_parent_id
                     JOIN jos_catalog_contract_xref ccx ON ccx.category = cx.category_child_id
                     JOIN jos_catalog_contract_parts ccp ON ccp.part_id = ccx.contract_part
                     JOIN jos_catalog_contract c ON c.id = ccp.contract_id
            WHERE pc.class_publish = 'Y'
              AND pc.class_deleted = 'N'
              AND c.archieved = 0
            ORDER BY c.id DESC, ccp.part_id ASC, pc.class_id ASC
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            connection.close()
            return results
        except Error as e:
            st.error(f"Error executing query: {e}")
            if connection.is_connected():
                cursor.close()
                connection.close()
            return []
    else:
        return []


def get_specification_parameters(spec_id):
    """Retrieve specification parameters for a given spec_id."""
    connection = connect_to_database()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)

            query = f"""WITH result_set AS (SELECT p.name      AS spec_name,
                           p.datlaik   AS sukurimo_data,
                           p.kdatlaik  AS paskutinio_veiksmo_data,
                           ds.name     AS pavadinimas,
                           ds.strength AS stiprumas,
                           ds.form     AS formacine_forma,
                           ds.method   AS vartojimo_budas,
                           ds.unit     AS matavimo_vienetas,
                           dp.type     AS tipas
                    FROM jos_catalog_params p
                             JOIN jos_catalog_product_class pc ON pc.class_id = p.class_id
                             JOIN jos_catalog_product pp ON pp.product_class_id = p.class_id
                             JOIN jos_drugs_specs_2022 ds ON ds.class_id = p.class_id
                             JOIN jos_drugs_prices_2022 dp ON dp.class_id = p.class_id AND dp.user_id = pp.product_owner
                    WHERE p.class_id = {spec_id}
                      AND pc.class_publish = 'Y'
                      AND pc.class_deleted = 'N'
                    GROUP BY p.name, p.datlaik, p.kdatlaik, ds.name, ds.strength, ds.form, ds.method, ds.unit, dp.type),
     cte_vaistai AS
         (SELECT 'spec_name' AS `key`, spec_name AS `value`
          FROM result_set
          UNION ALL
          SELECT 'sukurimo data', sukurimo_data
          FROM result_set
          UNION ALL
          SELECT 'paskutinio veiksmo (redagavimo) data', paskutinio_veiksmo_data
          FROM result_set
          UNION ALL
          SELECT 'pavadinimas', pavadinimas
          FROM result_set
          UNION ALL
          SELECT 'stiprumas', stiprumas
          FROM result_set
          UNION ALL
          SELECT 'formacine forma', formacine_forma
          FROM result_set
          UNION ALL
          SELECT 'vartojimo budas', vartojimo_budas
          FROM result_set
          UNION ALL
          SELECT 'matavimo vienetas', matavimo_vienetas
          FROM result_set
          UNION ALL
          SELECT 'tipas', tipas
          FROM result_set),
     cte_produktai AS (SELECT p.name AS 'parametras', p.description AS 'reikalavimas parametrui'
                       FROM jos_catalog_params p
                       WHERE p.class_id = {spec_id}
                         AND p.type != 'hidden_param'
                         AND p.method != 'file')
SELECT `key` AS parametras, value AS 'reikalavimas parametrui'
FROM cte_vaistai
UNION ALL
SELECT parametras, 'reikalavimas parametrui'
FROM cte_produktai;
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            connection.close()
            return results
        except Error as e:
            st.error(f"Error executing query for specification parameters: {e}")
            if connection.is_connected():
                cursor.close()
                connection.close()
            return []
    else:
        return []


def get_demo_specs_data():
    """Get sample specification data for demo mode."""
    return [
        {"grupe": "Electronics", "modulis": "Computers", "dalis": "Laptops", "specifikacija": "Gaming Laptops", "spec_id": 101},
        {"grupe": "Electronics", "modulis": "Computers", "dalis": "Laptops", "specifikacija": "Ultrabooks", "spec_id": 102},
        {"grupe": "Electronics", "modulis": "Computers", "dalis": "Desktops", "specifikacija": "Gaming PCs", "spec_id": 103},
        {"grupe": "Electronics", "modulis": "Phones", "dalis": "Smartphones", "specifikacija": "Android Phones", "spec_id": 104},
        {"grupe": "Electronics", "modulis": "Phones", "dalis": "Smartphones", "specifikacija": "iPhones", "spec_id": 105},
        {"grupe": "Home", "modulis": "Appliances", "dalis": "Kitchen", "specifikacija": "Refrigerators", "spec_id": 106},
        {"grupe": "Home", "modulis": "Appliances", "dalis": "Kitchen", "specifikacija": "Microwaves", "spec_id": 107},
        {"grupe": "Home", "modulis": "Furniture", "dalis": "Living Room", "specifikacija": "Sofas", "spec_id": 108},
    ]


def get_demo_spec_params():
    """Get sample specification parameters for demo mode."""
    return [
        {"parametras": "Procesorius", "reikalavimas parametrui": "Intel Core i7 arba geresnis"},
        {"parametras": "RAM", "reikalavimas parametrui": "Mažiausiai 16GB"},
        {"parametras": "Diskas", "reikalavimas parametrui": "SSD, mažiausiai 512GB"},
        {"parametras": "Ekranas", "reikalavimas parametrui": "15.6 colių, FullHD"},
    ]


