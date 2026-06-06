from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

st.set_page_config(layout="wide", page_title="MDM Explorer")

# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class PhysicalElement:
    id: str
    system: str
    database: str
    schema: str
    table: str
    column: str
    data_type: str
    is_nullable: bool = True
    is_pk: bool = False


@dataclass
class LogicalElement:
    id: str
    name: str
    description: str
    data_type: str
    is_required: bool
    business_rules: list[str] = field(default_factory=list)
    physical_elements: list[PhysicalElement] = field(default_factory=list)


@dataclass
class Entity:
    id: str
    name: str
    description: str
    steward: str
    logical_elements: list[LogicalElement] = field(default_factory=list)


@dataclass
class Domain:
    id: str
    name: str
    description: str
    owner: str
    entities: list[Entity] = field(default_factory=list)


# ── Mock data ─────────────────────────────────────────────────────────────────


def _pe(id, system, db, schema, table, column, dtype, nullable=True, pk=False):
    return PhysicalElement(id, system, db, schema, table, column, dtype, nullable, pk)


def build_domains() -> list[Domain]:
    return [
        Domain("d1", "Customer", "All customer-related data assets", "Sarah Chen", entities=[
            Entity("e1", "Profile", "Core customer identity attributes", "James Kim", logical_elements=[
                LogicalElement("l1", "Customer ID", "Unique identifier for a customer", "String", True,
                    ["Must be globally unique", "Immutable once assigned"],
                    [
                        _pe("p1",  "CRM", "crm_db", "dbo",       "customers",        "cust_id",      "VARCHAR(20)", False, True),
                        _pe("p2",  "ERP", "erp_db", "master",    "customer_master",   "cust_num",     "CHAR(10)",   False, True),
                        _pe("p3",  "DWH", "dw_db",  "analytics", "dim_customer",      "customer_key", "INTEGER",    False, True),
                    ]),
                LogicalElement("l2", "First Name", "Customer's given name", "String", True,
                    ["Max 50 characters", "No numeric characters"],
                    [
                        _pe("p4", "CRM", "crm_db", "dbo",       "customers",    "first_nm",   "VARCHAR(50)"),
                        _pe("p5", "DWH", "dw_db",  "analytics", "dim_customer", "first_name", "VARCHAR(50)"),
                    ]),
                LogicalElement("l3", "Date of Birth", "Customer's date of birth", "Date", False,
                    ["Must be a valid past date", "Used for age verification"],
                    [
                        _pe("p6", "CRM", "crm_db", "dbo",       "customers",    "dob",        "DATE"),
                        _pe("p7", "DWH", "dw_db",  "analytics", "dim_customer", "birth_date", "DATE"),
                    ]),
                LogicalElement("l4", "Membership Status", "Current membership tier", "String", True,
                    ["Values: ACTIVE, INACTIVE, SUSPENDED, PENDING"],
                    [
                        _pe("p8", "CRM", "crm_db", "dbo",    "customers",        "status_cd", "CHAR(2)"),
                        _pe("p9", "ERP", "erp_db", "master", "customer_master",  "status",    "VARCHAR(10)"),
                    ]),
            ]),
            Entity("e2", "Contact", "Customer contact information", "Ana Lopez", logical_elements=[
                LogicalElement("l5", "Email Address", "Primary email address", "String", False,
                    ["Must conform to RFC 5322", "Unique per customer"],
                    [
                        _pe("p10", "CRM", "crm_db", "dbo", "customer_contact", "email",      "VARCHAR(255)"),
                        _pe("p11", "MDM", "mdm_db", "hub", "contact_hub",      "email_addr", "VARCHAR(255)"),
                    ]),
                LogicalElement("l6", "Phone Number", "Primary contact phone", "String", False,
                    ["E.164 format required", "Includes country code"],
                    [
                        _pe("p12", "CRM", "crm_db", "dbo", "customer_contact", "phone_num", "VARCHAR(20)"),
                        _pe("p13", "MDM", "mdm_db", "hub", "contact_hub",      "phone",     "VARCHAR(20)"),
                    ]),
            ]),
        ]),
        Domain("d2", "Finance", "Financial data assets and transactions", "Marcus Rivera", entities=[
            Entity("e3", "Transaction", "Monetary transaction records", "Wei Zhang", logical_elements=[
                LogicalElement("l7", "Transaction ID", "Unique transaction identifier", "String", True,
                    ["UUID format", "System-generated, never reused"],
                    [
                        _pe("p14", "ERP", "erp_db", "finance",   "transactions",     "txn_id",          "VARCHAR(36)", False, True),
                        _pe("p15", "DWH", "dw_db",  "analytics", "fact_transaction", "transaction_key", "BIGINT",      False, True),
                    ]),
                LogicalElement("l8", "Amount", "Transaction monetary value", "Decimal", True,
                    ["Precision: 18,2", "Must be positive for credits"],
                    [
                        _pe("p16", "ERP", "erp_db", "finance",   "transactions",     "txn_amt", "DECIMAL(18,2)"),
                        _pe("p17", "DWH", "dw_db",  "analytics", "fact_transaction", "amount",  "NUMERIC(18,2)"),
                    ]),
                LogicalElement("l9", "Transaction Date", "Date the transaction occurred", "Date", True,
                    ["Cannot be a future date"],
                    [
                        _pe("p18", "ERP", "erp_db", "finance",   "transactions",     "txn_dt",           "DATE"),
                        _pe("p19", "DWH", "dw_db",  "analytics", "fact_transaction", "transaction_date", "DATE"),
                    ]),
            ]),
            Entity("e4", "Account", "Customer financial accounts", "Wei Zhang", logical_elements=[
                LogicalElement("l10", "Account Number", "Unique account identifier", "String", True,
                    ["Alphanumeric, 12 characters", "Luhn-validated"],
                    [
                        _pe("p20", "ERP", "erp_db", "finance", "accounts", "acct_num",   "CHAR(12)",    False, True),
                        _pe("p21", "CRM", "crm_db", "dbo",     "accounts", "account_id", "VARCHAR(12)", False, True),
                    ]),
                LogicalElement("l11", "Account Type", "Category of account", "String", True,
                    ["Values: CHECKING, SAVINGS, CREDIT, INVESTMENT"],
                    [
                        _pe("p22", "ERP", "erp_db", "finance",   "accounts",    "acct_type_cd", "CHAR(3)"),
                        _pe("p23", "DWH", "dw_db",  "analytics", "dim_account", "account_type", "VARCHAR(20)"),
                    ]),
            ]),
        ]),
        Domain("d3", "Product", "Product catalog and inventory data", "Priya Patel", entities=[
            Entity("e5", "Catalog", "Product definition and classification", "Tom Bradley", logical_elements=[
                LogicalElement("l12", "Product ID", "Unique product identifier", "String", True,
                    ["GTIN format", "Issued by product steward"],
                    [
                        _pe("p24", "ERP", "erp_db", "product",   "products",    "prod_id",     "VARCHAR(14)", False, True),
                        _pe("p25", "DWH", "dw_db",  "analytics", "dim_product", "product_key", "INTEGER",     False, True),
                        _pe("p26", "MDM", "mdm_db", "hub",       "product_hub", "product_id",  "VARCHAR(14)", False, True),
                    ]),
                LogicalElement("l13", "Product Name", "Commercial name of the product", "String", True,
                    ["Max 200 characters", "Unique within category"],
                    [
                        _pe("p27", "ERP", "erp_db", "product",   "products",    "prod_nm",      "VARCHAR(200)"),
                        _pe("p28", "DWH", "dw_db",  "analytics", "dim_product", "product_name", "VARCHAR(200)"),
                    ]),
                LogicalElement("l14", "Category", "Product classification category", "String", True,
                    ["Must match approved category taxonomy"],
                    [
                        _pe("p29", "ERP", "erp_db", "product",   "products",    "cat_cd",        "VARCHAR(50)"),
                        _pe("p30", "DWH", "dw_db",  "analytics", "dim_product", "category_name", "VARCHAR(100)"),
                    ]),
            ]),
            Entity("e6", "Inventory", "Stock levels and warehouse data", "Tom Bradley", logical_elements=[
                LogicalElement("l15", "SKU", "Stock keeping unit identifier", "String", True,
                    ["Alphanumeric, up to 20 characters", "Unique per warehouse location"],
                    [
                        _pe("p31", "ERP", "erp_db", "inventory", "stock",          "sku_cd", "VARCHAR(20)", False, True),
                        _pe("p32", "DWH", "dw_db",  "analytics", "fact_inventory", "sku",    "VARCHAR(20)"),
                    ]),
                LogicalElement("l16", "Stock Level", "Current quantity on hand", "Integer", True,
                    ["Cannot be negative", "Triggers reorder at configured threshold"],
                    [
                        _pe("p33", "ERP", "erp_db", "inventory", "stock",          "qty_on_hand",      "INTEGER"),
                        _pe("p34", "DWH", "dw_db",  "analytics", "fact_inventory", "quantity_on_hand", "INTEGER"),
                    ]),
            ]),
        ]),
    ]


# ── Lookups and DataFrame ─────────────────────────────────────────────────────


def build_lookups(domains: list[Domain]):
    le_lookup: dict[str, LogicalElement] = {}
    entity_lookup: dict[str, Entity] = {}
    domain_lookup: dict[str, Domain] = {}
    for domain in domains:
        for entity in domain.entities:
            for le in entity.logical_elements:
                le_lookup[le.id] = le
                entity_lookup[le.id] = entity
                domain_lookup[le.id] = domain
    return le_lookup, entity_lookup, domain_lookup


def flatten_to_df(domains: list[Domain]) -> pd.DataFrame:
    rows = []
    for domain in domains:
        for entity in domain.entities:
            for le in entity.logical_elements:
                rows.append({
                    "_id":             le.id,
                    "domain":          domain.name,
                    "entity":          entity.name,
                    "logical_element": le.name,
                    "data_type":       le.data_type,
                    "required":        "Yes" if le.is_required else "No",
                    "physical_count":  len(le.physical_elements),
                    "description":     le.description,
                    "detail_data": json.dumps([
                        {
                            "system":    pe.system,
                            "database":  pe.database,
                            "schema":    pe.schema,
                            "table":     pe.table,
                            "column":    pe.column,
                            "data_type": pe.data_type,
                            "nullable":  pe.is_nullable,
                            "pk":        pe.is_pk,
                        }
                        for pe in le.physical_elements
                    ]),
                })
    return pd.DataFrame(rows)


# ── SQL generation ────────────────────────────────────────────────────────────


def generate_sql(pe: PhysicalElement) -> str:
    null_line = (
        f"\n    SUM(CASE WHEN {pe.column} IS NULL THEN 1 ELSE 0 END)  AS null_count,"
        if pe.is_nullable
        else ""
    )
    return (
        f"-- Profile: [{pe.system}] {pe.database}.{pe.schema}.{pe.table}.{pe.column}\n"
        f"SELECT\n"
        f"    {pe.column},\n"
        f"    COUNT(*)                        AS total_records,\n"
        f"    COUNT(DISTINCT {pe.column})     AS distinct_values,{null_line}\n"
        f"    MIN({pe.column})                AS min_value,\n"
        f"    MAX({pe.column})                AS max_value\n"
        f"FROM {pe.schema}.{pe.table}\n"
        f"GROUP BY {pe.column}\n"
        f"ORDER BY total_records DESC\n"
        f"LIMIT 100;"
    )


def typewriter_code(sql: str, placeholder) -> None:
    displayed = ""
    for char in sql:
        displayed += char
        placeholder.code(displayed, language="sql")
        time.sleep(0.008)


# ── Detail panel ──────────────────────────────────────────────────────────────


def render_detail(le_id: str, le_lookup, entity_lookup, domain_lookup):
    le = le_lookup[le_id]
    entity = entity_lookup[le_id]
    domain = domain_lookup[le_id]

    st.subheader(le.name)
    st.caption(f"{domain.name} › {entity.name}  ·  Steward: {entity.steward}")
    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Data Type", le.data_type)
    c2.metric("Required", "Yes" if le.is_required else "No")
    c3.metric("Physical Mappings", len(le.physical_elements))

    st.markdown("**Description**")
    st.write(le.description)

    if le.business_rules:
        st.markdown("**Business Rules**")
        for rule in le.business_rules:
            st.markdown(f"- {rule}")

    st.markdown("**Physical Manifestations**")
    for pe in le.physical_elements:
        with st.expander(f"{pe.system}  ·  `{pe.schema}.{pe.table}.{pe.column}`"):
            a, b, c = st.columns(3)
            a.markdown(f"**Database**  \n{pe.database}")
            b.markdown(f"**Data Type**  \n{pe.data_type}")
            c.markdown(f"**Primary Key**  \n{'Yes' if pe.is_pk else 'No'}")

            if st.button("Generate profile SQL", key=f"sql_{pe.id}"):
                typewriter_code(generate_sql(pe), st.empty())


# ── App ───────────────────────────────────────────────────────────────────────

domains = build_domains()
le_lookup, entity_lookup, domain_lookup = build_lookups(domains)
df = flatten_to_df(domains)

st.title("MDM Explorer")
tab_search, tab_browse = st.tabs(["Search", "Browse"])

# ── Search ────────────────────────────────────────────────────────────────────
with tab_search:
    query = st.text_input("Search elements", placeholder="e.g. customer, email, amount...")

    if query:
        q = query.lower()
        results = []
        for domain in domains:
            for entity in domain.entities:
                for le in entity.logical_elements:
                    if q in le.name.lower() or q in le.description.lower():
                        results.append({
                            "Type": "Logical Element", "Name": le.name,
                            "Domain": domain.name, "Entity": entity.name,
                            "Data Type": le.data_type, "Description": le.description,
                        })
                    for pe in le.physical_elements:
                        if q in pe.column.lower() or q in pe.table.lower() or q in pe.system.lower():
                            results.append({
                                "Type": "Physical Element", "Name": pe.column,
                                "Domain": domain.name, "Entity": entity.name,
                                "Data Type": pe.data_type,
                                "Description": f"{pe.system} · {pe.schema}.{pe.table}",
                            })
        if results:
            st.caption(f"{len(results)} result(s)")
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        else:
            st.warning("No results found.")

# ── Browse ────────────────────────────────────────────────────────────────────
with tab_browse:
    grid_col, detail_col = st.columns([3, 2])

    with grid_col:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_column("_id",         hide=True)
        gb.configure_column("domain",      rowGroup=True, hide=True)
        gb.configure_column("entity",      rowGroup=True, hide=True)
        gb.configure_column("detail_data", hide=True)
        gb.configure_column("description", hide=True)
        gb.configure_selection("single")
        gb.configure_grid_options(
            masterDetail=True,
            animateRows=True,
            groupDefaultExpanded=1,
            autoGroupColumnDef={"field": "logical_element", "headerName": "Element", "minWidth": 240},
            detailRowHeight=200,
            detailCellRendererParams={
                "detailGridOptions": {
                    "columnDefs": [
                        {"field": "system",    "width": 80},
                        {"field": "database",  "width": 110},
                        {"field": "schema",    "width": 110},
                        {"field": "table",     "width": 160},
                        {"field": "column",    "width": 160},
                        {"field": "data_type", "headerName": "Type", "width": 130},
                        {"field": "pk",        "headerName": "PK",   "width": 60},
                    ],
                    "defaultColDef": {"resizable": True, "sortable": True},
                },
                "getDetailRowData": JsCode(
                    "function(params) { params.successCallback(JSON.parse(params.data.detail_data)); }"
                ),
            },
        )
        grid_response = AgGrid(
            df,
            gridOptions=gb.build(),
            allow_unsafe_jscode=True,
            enable_enterprise_modules=True,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            height=600,
        )

    with detail_col:
        raw_sel = grid_response.selected_rows
        selected_id = None
        if isinstance(raw_sel, pd.DataFrame):
            if len(raw_sel) > 0:
                selected_id = raw_sel.iloc[0].get("_id")
        elif isinstance(raw_sel, list) and raw_sel:
            selected_id = raw_sel[0].get("_id")

        if selected_id and selected_id in le_lookup:
            render_detail(selected_id, le_lookup, entity_lookup, domain_lookup)
        else:
            st.info("Select a logical element row to view details.")
