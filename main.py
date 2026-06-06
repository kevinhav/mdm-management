from __future__ import annotations

import dataclasses
import json
import pathlib
import time
from dataclasses import field
from typing import cast

import pandas as pd
import pandera.pandas as pa
import sqlglot
import sqlglot.expressions as exp
import streamlit as st
from pandera.typing.pandas import Series
from pydantic.dataclasses import dataclass
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

_HERE = pathlib.Path(__file__).parent

st.set_page_config(layout="wide", page_title="MDM Explorer")

# ── Domain model ──────────────────────────────────────────────────────────────


@dataclass
class PhysicalElement:
    id: str
    system: str
    database: str
    schema: str
    table: str
    column: str
    data_type: str
    nullable: bool = True
    pk: bool = False
    null_pct: float | None = None


@dataclass
class LogicalElement:
    id: str
    name: str
    description: str
    aliases: list[str] = field(default_factory=list)
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


# ── Pandera schema ────────────────────────────────────────────────────────────


class CatalogSchema(pa.DataFrameModel):
    domain_id: Series[str]
    domain_name: Series[str]
    domain_description: Series[str] = pa.Field(nullable=True)
    domain_owner: Series[str]
    entity_id: Series[str]
    entity_name: Series[str]
    entity_description: Series[str] = pa.Field(nullable=True)
    entity_steward: Series[str]
    le_id: Series[str]
    le_name: Series[str]
    le_description: Series[str] = pa.Field(nullable=True)
    le_aliases: Series[str] = pa.Field(nullable=True)
    business_rules: Series[str] = pa.Field(nullable=True)
    pe_id: Series[str] = pa.Field(unique=True)
    pe_system: Series[str]
    pe_database: Series[str]
    pe_schema: Series[str]
    pe_table: Series[str]
    pe_column: Series[str]
    pe_data_type: Series[str]
    pe_nullable: Series[bool]
    pe_pk: Series[bool]
    pe_null_pct: Series[float] = pa.Field(nullable=True, ge=0, le=100)

    class Config:
        coerce = True
        strict = "filter"

    @pa.dataframe_check
    @classmethod
    def le_id_maps_to_one_name(cls, df: pd.DataFrame) -> bool:
        return bool(df.groupby("le_id")["le_name"].nunique().le(1).all())


# ── CSV loader ────────────────────────────────────────────────────────────────


@st.cache_data
def load_from_csv(path: str | pathlib.Path = _HERE / "catalog.csv") -> list[Domain]:
    df = pd.read_csv(path)
    # pandas reads True/False as strings; coerce before pandera sees them
    for col in ("pe_nullable", "pe_pk"):
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
    validated = CatalogSchema.validate(df)

    domains: list[Domain] = []
    for d_id, d_df in validated.groupby("domain_id", sort=False):
        d0 = d_df.iloc[0]
        entities: list[Entity] = []
        for e_id, e_df in d_df.groupby("entity_id", sort=False):
            e0 = e_df.iloc[0]
            les: list[LogicalElement] = []
            for le_id, le_df in e_df.groupby("le_id", sort=False):
                le0 = le_df.iloc[0]

                def _split(val) -> list[str]:
                    return (
                        [r.strip() for r in str(val).split("|") if r.strip()]
                        if pd.notna(val)
                        else []
                    )

                aliases = _split(le0["le_aliases"])
                rules = _split(le0["business_rules"])

                def _build_pe(r) -> PhysicalElement:
                    kwargs = {}
                    for f in dataclasses.fields(PhysicalElement):
                        val = r[f"pe_{f.name}"]
                        kwargs[f.name] = (
                            None if (isinstance(val, float) and pd.isna(val)) else val
                        )
                    return PhysicalElement(**kwargs)

                pes = [_build_pe(r) for _, r in le_df.iterrows()]
                les.append(
                    LogicalElement(
                        id=str(le_id),
                        name=str(le0["le_name"]),
                        description=str(le0["le_description"])
                        if pd.notna(le0["le_description"])
                        else "",
                        aliases=aliases,
                        business_rules=rules,
                        physical_elements=pes,
                    )
                )
            entities.append(
                Entity(
                    id=str(e_id),
                    name=str(e0["entity_name"]),
                    description=str(e0["entity_description"])
                    if pd.notna(e0["entity_description"])
                    else "",
                    steward=str(e0["entity_steward"]),
                    logical_elements=les,
                )
            )
        domains.append(
            Domain(
                id=str(d_id),
                name=str(d0["domain_name"]),
                description=str(d0["domain_description"])
                if pd.notna(d0["domain_description"])
                else "",
                owner=str(d0["domain_owner"]),
                entities=entities,
            )
        )
    return domains


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
                rows.append(
                    {
                        "_id": le.id,
                        "domain": domain.name,
                        "entity": entity.name,
                        "logical_element": le.name,
                        "aliases": ", ".join(le.aliases) if le.aliases else "",
                        "physical_count": len(le.physical_elements),
                        "description": le.description,
                        "_pe_systems": " ".join(
                            sorted({pe.system for pe in le.physical_elements})
                        ),
                        "_pe_tables": " ".join(pe.table for pe in le.physical_elements),
                        "_pe_columns": " ".join(
                            pe.column for pe in le.physical_elements
                        ),
                        "_pe_schemas": " ".join(
                            sorted({pe.schema for pe in le.physical_elements})
                        ),
                        "detail_data": json.dumps(
                            [dataclasses.asdict(pe) for pe in le.physical_elements]
                        ),
                    }
                )
    return pd.DataFrame(rows)


# ── SQL generation ────────────────────────────────────────────────────────────


def generate_sql(pe: PhysicalElement) -> str:
    null_line = (
        f"\n    SUM(CASE WHEN {pe.column} IS NULL THEN 1 ELSE 0 END)  AS null_count,"
        if pe.nullable
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


def render_detail(
    le_id: str, le_lookup, entity_lookup, domain_lookup, key_ns: str = ""
):
    le = le_lookup[le_id]
    entity = entity_lookup[le_id]
    domain = domain_lookup[le_id]

    st.subheader(le.name)
    st.caption(f"{domain.name} › {entity.name}  ·  Steward: {entity.steward}")
    st.divider()

    st.metric("Physical Mappings", len(le.physical_elements))

    st.markdown("**Description**")
    st.write(le.description)

    if le.aliases:
        st.markdown("**Also known as**")
        st.markdown("  ".join(f"`{a}`" for a in le.aliases))

    if le.business_rules:
        st.markdown("**Business Rules**")
        for rule in le.business_rules:
            st.markdown(f"- {rule}")

    st.markdown("**Physical Manifestations**")
    for pe in le.physical_elements:
        with st.expander(f"{pe.system}  ·  `{pe.schema}.{pe.table}.{pe.column}`"):
            a, b, c, d = st.columns(4)
            a.markdown(f"**Database**  \n{pe.database}")
            b.markdown(f"**Data Type**  \n{pe.data_type}")
            c.markdown(f"**Primary Key**  \n{'Yes' if pe.pk else 'No'}")
            d.metric(
                "% Null", f"{pe.null_pct:.1f}%" if pe.null_pct is not None else "—"
            )

            if st.button("Generate profile SQL", key=f"sql_{key_ns}_{pe.id}"):
                typewriter_code(generate_sql(pe), st.empty())


# ── SQL analyzer ─────────────────────────────────────────────────────────────


@dataclass
class SqlRef:
    table: str
    column: str | None  # None when SELECT *


@dataclass
class MatchedElement:
    ref: SqlRef
    pe: PhysicalElement
    le: LogicalElement
    entity: Entity
    domain: Domain


def parse_sql_refs(sql: str) -> tuple[list[SqlRef], str | None]:
    """Parse SQL and return (references, error). References are (table, column) pairs."""
    try:
        parsed = sqlglot.parse_one(sql, error_level=sqlglot.ErrorLevel.RAISE)
    except Exception as e:
        return [], str(e)

    # Build alias map: alias/name → table name (lowercased)
    alias_map: dict[str, str] = {}
    for table_node in parsed.find_all(exp.Table):
        name = table_node.name.lower()
        alias = table_node.alias.lower() if table_node.alias else name
        alias_map[alias] = name
        alias_map[name] = name

    refs: list[SqlRef] = []

    # SELECT * → emit table-only refs for all referenced tables
    if parsed.find(exp.Star):
        for table_name in set(alias_map.values()):
            refs.append(SqlRef(table=table_name, column=None))

    # Named column references
    for col_node in parsed.find_all(exp.Column):
        col_name = col_node.name.lower()
        qualifier = col_node.table.lower() if col_node.table else None
        if qualifier and qualifier in alias_map:
            refs.append(SqlRef(table=alias_map[qualifier], column=col_name))
        elif qualifier:
            refs.append(SqlRef(table=qualifier, column=col_name))
        else:
            # Unqualified column — could belong to any referenced table
            for table_name in set(alias_map.values()):
                refs.append(SqlRef(table=table_name, column=col_name))

    return refs, None


def match_elements(
    refs: list[SqlRef], domains: list[Domain]
) -> tuple[list[MatchedElement], list[SqlRef]]:
    matched: list[MatchedElement] = []
    unmatched: list[SqlRef] = []
    seen_pe_ids: set[str] = set()

    for ref in refs:
        found = False
        for domain in domains:
            for entity in domain.entities:
                for le in entity.logical_elements:
                    for pe in le.physical_elements:
                        table_match = pe.table.lower() == ref.table
                        col_match = (
                            ref.column is None or pe.column.lower() == ref.column
                        )
                        if table_match and col_match and pe.id not in seen_pe_ids:
                            matched.append(MatchedElement(ref, pe, le, entity, domain))
                            seen_pe_ids.add(pe.id)
                            found = True
        if not found and ref not in unmatched:
            unmatched.append(ref)

    return matched, unmatched


# ── App ───────────────────────────────────────────────────────────────────────

try:
    domains = load_from_csv()
except Exception as _load_err:
    st.error(f"Failed to load catalog.csv: {_load_err}")
    st.stop()
le_lookup, entity_lookup, domain_lookup = build_lookups(domains)
df = flatten_to_df(domains)

_ac_terms: list[str] = sorted(
    set(
        [d.name for d in domains]
        + [e.name for d in domains for e in d.entities]
        + [le.name for d in domains for e in d.entities for le in e.logical_elements]
        + [
            alias
            for d in domains
            for e in d.entities
            for le in e.logical_elements
            for alias in le.aliases
        ]
        + [
            pe.system
            for d in domains
            for e in d.entities
            for le in e.logical_elements
            for pe in le.physical_elements
        ]
        + [
            pe.table
            for d in domains
            for e in d.entities
            for le in e.logical_elements
            for pe in le.physical_elements
        ]
        + [
            pe.column
            for d in domains
            for e in d.entities
            for le in e.logical_elements
            for pe in le.physical_elements
        ]
        + [
            pe.schema
            for d in domains
            for e in d.entities
            for le in e.logical_elements
            for pe in le.physical_elements
        ]
    )
)

st.title("MDM Explorer")
tab_browse, tab_sql = st.tabs(["Browse", "SQL Analyzer"])

# ── Browse ────────────────────────────────────────────────────────────────────
with tab_browse:
    selected_term = st.selectbox(
        "Search",
        options=_ac_terms,
        index=None,
        placeholder="Filter by element name, type, domain…",
        label_visibility="collapsed",
    )
    search_query = (selected_term or "").lower()

    if search_query:
        _search_cols = [
            "logical_element",
            "aliases",
            "description",
            "_pe_systems",
            "_pe_tables",
            "_pe_columns",
            "_pe_schemas",
            "domain",
            "entity",
        ]
        _mask = (
            df[_search_cols]
            .apply(lambda col: col.str.contains(search_query, case=False, na=False))
            .any(axis=1)
        )
        display_df = df[_mask]
    else:
        display_df = df

    grid_col, detail_col = st.columns([3, 2])

    with grid_col:
        gb = GridOptionsBuilder.from_dataframe(display_df)
        gb.configure_column("_id", hide=True)
        gb.configure_column(
            "domain", rowGroup=True, hide=True, enableRowGroup=True, enablePivot=True
        )
        gb.configure_column(
            "entity", rowGroup=True, hide=True, enableRowGroup=True, enablePivot=True
        )
        gb.configure_column("detail_data", hide=True)
        gb.configure_column("description", hide=True)
        gb.configure_column("_pe_systems", hide=True)
        gb.configure_column("_pe_tables", hide=True)
        gb.configure_column("_pe_columns", hide=True)
        gb.configure_column("_pe_schemas", hide=True)
        gb.configure_selection("single")
        gb.configure_grid_options(
            masterDetail=True,
            animateRows=True,
            groupDefaultExpanded=1,
            sideBar={"toolPanels": ["columns", "filters"]},
            rowGroupPanelShow="always",
            autoGroupColumnDef={
                "field": "logical_element",
                "headerName": "Element",
                "minWidth": 240,
                "cellStyle": JsCode("""function(params) {
                    if (!params.node.group) {
                        return {color: '#4C72B0', textDecoration: 'underline', cursor: 'pointer', fontWeight: 500};
                    }
                }"""),
            },
            detailRowHeight=200,
            detailCellRendererParams={
                "detailGridOptions": {
                    "columnDefs": [
                        {"field": "system", "width": 80},
                        {"field": "database", "width": 110},
                        {"field": "schema", "width": 110},
                        {"field": "table", "width": 160},
                        {"field": "column", "width": 160},
                        {"field": "data_type", "headerName": "Type", "width": 120},
                        {"field": "pk", "headerName": "PK", "width": 55},
                        {
                            "field": "null_pct",
                            "headerName": "% Null",
                            "width": 85,
                            "valueFormatter": "params.value != null ? params.value.toFixed(1) + '%' : '—'",
                        },
                    ],
                    "defaultColDef": {"resizable": True, "sortable": True},
                },
                "getDetailRowData": JsCode(
                    "function(params) { params.successCallback(JSON.parse(params.data.detail_data)); }"
                ),
            },
        )
        grid_response = AgGrid(
            display_df,
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
            render_detail(
                selected_id, le_lookup, entity_lookup, domain_lookup, key_ns="browse"
            )
        else:
            st.info("Select a logical element row to view details.")

# ── SQL Analyzer ──────────────────────────────────────────────────────────────
with tab_sql:
    st.session_state.setdefault("sql_results", None)

    sql_input = st.text_area(
        "Paste a SQL query",
        height=200,
        placeholder="SELECT c.cust_id, c.first_nm\nFROM dbo.customers c\nJOIN master.customer_master cm ON c.cust_id = cm.cust_num",
    )

    if st.button("Analyze", type="primary", disabled=not sql_input.strip()):
        refs, error = parse_sql_refs(sql_input)
        if error:
            st.session_state.sql_results = {"error": error}
        else:
            matched, unmatched = match_elements(refs, domains)
            st.session_state.sql_results = {"matched": matched, "unmatched": unmatched}

    results = st.session_state.sql_results
    if results is not None:
        if "error" in results:
            st.error(f"Could not parse SQL: {results['error']}")
        else:
            matched = cast(list[MatchedElement], results["matched"])
            unmatched = cast(list[SqlRef], results["unmatched"])

            c1, c2 = st.columns(2)
            c1.metric("Catalog matches", len(matched))
            c2.metric("Unresolved references", len(unmatched))
            st.divider()

            if unmatched:
                st.markdown("**Unresolved References** *(not found in catalog)*")
                for ref in unmatched:
                    label = (
                        f"`{ref.table}.{ref.column}`"
                        if ref.column
                        else f"`{ref.table}.*`"
                    )
                    st.markdown(f"- {label}")

            if matched:
                st.markdown("**Matched Data Elements**")
                sql_grid_col, sql_detail_col = st.columns([3, 2])

                with sql_grid_col:
                    sql_rows = [
                        {
                            "_le_id": m.le.id,
                            "Column": m.pe.column,
                            "Table": m.pe.table,
                            "System": m.pe.system,
                            "Logical Element": m.le.name,
                            "Entity": m.entity.name,
                            "Domain": m.domain.name,
                        }
                        for m in matched
                    ]
                    sql_df = pd.DataFrame(sql_rows)
                    sql_gb = GridOptionsBuilder.from_dataframe(sql_df)
                    sql_gb.configure_column("_le_id", hide=True)
                    sql_gb.configure_selection("single")
                    sql_response = AgGrid(
                        sql_df,
                        gridOptions=sql_gb.build(),
                        allow_unsafe_jscode=True,
                        update_mode=GridUpdateMode.SELECTION_CHANGED,
                        height=min(200 + len(matched) * 42, 500),
                    )

                with sql_detail_col:
                    sql_sel = sql_response.selected_rows
                    sql_le_id = None
                    if isinstance(sql_sel, pd.DataFrame):
                        if len(sql_sel) > 0:
                            sql_le_id = sql_sel.iloc[0].get("_le_id")
                    elif isinstance(sql_sel, list) and sql_sel:
                        sql_le_id = sql_sel[0].get("_le_id")

                    if sql_le_id and sql_le_id in le_lookup:
                        render_detail(
                            sql_le_id,
                            le_lookup,
                            entity_lookup,
                            domain_lookup,
                            key_ns="sql",
                        )
                    else:
                        st.info("Click a row to view the logical element detail.")
