# PIEW – Posting Integrity & Exception Workbench (Streamlit)
# ----------------------------------------------------------
# Phase-1 demo aligned to Emerald–Ampera RCM Command Center:
# • Posting Ledger + Exceptions Workbench
# • Canonical Payer Normalization
# • Role-based views (Operator / Supervisor / Auditor)
# • KPI Dashboard (SLA-based)
# • Claim Lifecycle Timelines
# • Guided Remediation Wizard in the workbench
# • Export simulator for Credible / Tebra / CSV
#
# Run locally:
#   1) pip install -r requirements.txt
#   2) streamlit run app.py
#
# Notes:
# - All data is synthetic (no PHI).
# - State is kept in-memory for the session.

import random
from datetime import datetime, timedelta
import textwrap
import difflib

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Constants & Seed
# -----------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

EX_TYPES = [
    "Not Posted",
    "No Match",
    "Wrong Payer",
    "Retracted",
    "EOB Mismatch",
]

PAYER_VARIANTS = [
    ("Tricare West", "TRICARE"),
    ("Health Net Comm", "HEALTH NET"),
    ("Medi-Cal CA", "MEDI-CAL"),
    ("Aetna – Comm", "AETNA"),
    ("Cigna PPO", "CIGNA"),
    ("United Hlthcr", "UNITED HEALTHCARE"),
]

SUGGESTIONS = {
    "Not Posted": "Re-run posting with corrected payer ID; include missing PLB if present.",
    "No Match": "Link ERA line to internal claim ID using date+amount+patient; persist mapping.",
    "Wrong Payer": "Remap to canonical payer and regenerate write-back summary.",
    "Retracted": "Apply takeback (PLB) and re-distribute to patient/secondary per policy.",
    "EOB Mismatch": "Use EOB snippet to fix disallowed/PR; recalc expected vs actual.",
}

STATUS_CHOICES = ["Open", "In Progress", "Resolved", "Ready for Write-back"]
SEVERITY = ["High", "Medium", "Low"]

# Simple SLA in days for resolution (Phase-1 target)
SLA_DAYS = 14

# -----------------------------
# Synthetic Data Generators
# -----------------------------


def _random_patient():
    first = random.choice(
        ["Alex", "Sam", "Riley", "Jordan", "Taylor", "Casey", "Avery", "Quinn"]
    )
    last = random.choice(
        ["Lee", "Patel", "Garcia", "Nguyen", "Smith", "Kim", "Brown", "Davis"]
    )
    return f"{first} {last}"


def generate_dummy_exceptions(n: int = 200) -> pd.DataFrame:
    """Generate the exception table – main workbench data."""
    rows = []
    today = datetime.utcnow().date()
    for i in range(n):
        claim_id = f"CLM{100000 + i}"
        patient = _random_patient()
        payer_read, payer_canon = random.choice(PAYER_VARIANTS)
        etype = random.choices(EX_TYPES, weights=[0.28, 0.24, 0.22, 0.16, 0.10])[0]
        sev = random.choices(SEVERITY, weights=[0.35, 0.45, 0.20])[0]
        amt_expected = round(np.random.uniform(80, 400), 2)
        delta = np.random.uniform(-40, 40)
        amt_paid = max(0, round(amt_expected + delta, 2))
        pr = max(0, round(np.random.uniform(0, 40), 2))
        disallowed = max(0, round(max(0, amt_expected - amt_paid - pr), 2))
        days_since = random.randint(1, 60)

        # Timeline dates
        service_date = today - timedelta(days=days_since + random.randint(5, 25))
        posting_date = service_date + timedelta(days=random.randint(1, 10))

        has_video = random.random() < 0.25
        has_image = random.random() < 0.4
        status = "Open"
        sugg = SUGGESTIONS[etype]

        rows.append(
            {
                "claim_id": claim_id,
                "patient": patient,
                "payer_read": payer_read,
                "payer_canonical": payer_canon,
                "exception_type": etype,
                "severity": sev,
                "amount_expected": amt_expected,
                "amount_paid": amt_paid,
                "pr_amount": pr,
                "disallowed": disallowed,
                "days_since": days_since,
                "service_date": service_date,
                "posting_date": posting_date,
                "has_video": has_video,
                "has_image": has_image,
                "status": status,
                "suggested_fix": sugg,
            }
        )
    return pd.DataFrame(rows)


def generate_dummy_ledger(ex_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a posting ledger view from exceptions.
    Each exception row becomes a ledger line with dates & basic AR attributes.
    """
    ledger = ex_df.copy()
    ledger["txn_date"] = ledger["posting_date"]
    ledger["balance_variance"] = (
        ledger["amount_expected"] - ledger["amount_paid"] - ledger["pr_amount"]
    )
    ledger["is_mismatch"] = (ledger["balance_variance"].abs() > 5) | (
        ledger["exception_type"].isin(["EOB Mismatch", "Wrong Payer", "Not Posted"])
    )
    return ledger[
        [
            "claim_id",
            "txn_date",
            "payer_canonical",
            "amount_expected",
            "amount_paid",
            "pr_amount",
            "disallowed",
            "balance_variance",
            "exception_type",
            "is_mismatch",
            "severity",
            "status",
            "days_since",
        ]
    ]


# -----------------------------
# Session State Bootstrap
# -----------------------------

st.set_page_config(
    page_title="PIEW – Posting Integrity & Exception Workbench",
    layout="wide",
)

if "exceptions_df" not in st.session_state:
    st.session_state.exceptions_df = generate_dummy_exceptions(200)

if "ledger_df" not in st.session_state:
    st.session_state.ledger_df = generate_dummy_ledger(st.session_state.exceptions_df)

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

if "role" not in st.session_state:
    st.session_state.role = "Operator"  # default

if "wizard_state" not in st.session_state:
    st.session_state.wizard_state = {}  # keyed by claim_id


# -----------------------------
# Helper: Role / Permissions
# -----------------------------


def can_edit() -> bool:
    return st.session_state.role in ("Operator", "Supervisor")


def can_writeback() -> bool:
    return st.session_state.role == "Supervisor"


# -----------------------------
# Sidebar – Navigation & Guided Coach
# -----------------------------

st.sidebar.title("PIEW – Phase 1 Demo")

st.sidebar.markdown("### Role")
st.session_state.role = st.sidebar.radio(
    "Simulate login as:",
    options=["Operator", "Supervisor", "Auditor"],
    help=(
        "• Operator: Triage & update statuses\n"
        "• Supervisor: All of Operator + write-back staging\n"
        "• Auditor: Read-only, no edits"
    ),
)

nav = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Dashboard",
        "Posting Ledger",
        "Exceptions Workbench",
        "Claim Timelines",
        "Payer Resolver",
        "Evidence",
        "Audit & Exports",
    ],
)

with st.sidebar.expander("Guided Coach – What should I do next?", expanded=True):
    df = st.session_state.exceptions_df
    open_mask = df.status.isin(["Open", "In Progress"])
    open_cnt = open_mask.sum()
    high_cnt = (open_mask & (df.severity == "High")).sum()
    wrong_payer_cnt = (open_mask & (df.exception_type == "Wrong Payer")).sum()
    not_posted_cnt = (open_mask & (df.exception_type == "Not Posted")).sum()

    st.markdown(
        f"**Queue health:** {open_cnt} open items · {high_cnt} high-severity\n\n"
        f"1) Prioritize **High** severity in the workbench.\n\n"
        f"2) Quick wins: **Wrong Payer ({wrong_payer_cnt})** via canonical remap.\n\n"
        f"3) Tackle **Not Posted ({not_posted_cnt})** – re-run posting with suggested fix.\n"
        f"4) Use **Posting Ledger** for a ledger-level view before closing the day."
    )

    st.info(
        "Tip: Supervisor should review items marked 'Ready for Write-back' before exporting to Credible or Tebra."
    )

st.sidebar.markdown("---")
st.sidebar.caption("All data in this demo is synthetic (no PHI).")

# -----------------------------
# KPI helper
# -----------------------------


def compute_kpis(df: pd.DataFrame) -> dict:
    now = datetime.utcnow().date()
    open_mask = df.status.isin(["Open", "In Progress"])
    resolved_mask = df.status.eq("Resolved")

    # For demo purposes, assume "days_since" is age from posting
    df_age = df.copy()
    df_age["age_days"] = df_age["days_since"]

    resolved_df = df_age[resolved_mask].copy()
    if not resolved_df.empty:
        resolved_df["within_sla"] = resolved_df["age_days"] <= SLA_DAYS
        sla_pct = 100 * resolved_df["within_sla"].mean()
        avg_resolve_days = resolved_df["age_days"].mean()
    else:
        sla_pct = 0.0
        avg_resolve_days = 0.0

    return {
        "open_cnt": int(open_mask.sum()),
        "high_open_cnt": int((open_mask & df.severity.eq("High")).sum()),
        "sla_pct": round(sla_pct, 1),
        "avg_resolve_days": round(avg_resolve_days, 1),
    }


# -----------------------------
# Page Renderers
# -----------------------------


def render_overview():
    st.title("Posting Integrity & Exception Workbench – Phase 1 Demo")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            textwrap.dedent(
                """
                **What you're seeing (Phase-1 scope):**
                - Posting ledger + auto-detected posting & matching exceptions (simulated 835 feed)
                - Canonical payer normalization (e.g., *Tricare West* → **TRICARE**)
                - SLA-driven exception workbench for ops teams
                - Audit logging & export summaries for Credible / Tebra

                **How this aligns to the RCM Command Center proposal:**
                - **Financial Integrity:** posting workflow + credit ledger + reconciliation
                - **Data Normalization:** canonical payer dictionary & exception taxonomy
                - **Exception Workbench v1:** SLA queues and triage guided by rules
                - **Governance:** role-based views (Operator / Supervisor / Auditor)
                """
            )
        )
    with col2:
        df = st.session_state.exceptions_df
        k1, k2, k3 = st.columns(3)
        k1.metric("Open", int(df.status.isin(["Open", "In Progress"]).sum()))
        k2.metric("Resolved", int((df.status == "Resolved").sum()))
        k3.metric(
            "Ready for Write-back", int((df.status == "Ready for Write-back").sum())
        )
        st.info(f"Current role: **{st.session_state.role}**")


def render_dashboard():
    st.title("Dashboard – Queue Health & KPIs")
    df = st.session_state.exceptions_df
    kpis = compute_kpis(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Open (All)",
        kpis["open_cnt"],
        help="Exceptions currently Open or In Progress.",
    )
    c2.metric(
        "High Severity Open",
        kpis["high_open_cnt"],
        help="High severity items needing immediate attention.",
    )
    c3.metric(
        "% Resolved Within SLA",
        f"{kpis['sla_pct']}%",
        help=f"Resolved where age <= {SLA_DAYS} days.",
    )
    c4.metric(
        "Avg Days to Resolve",
        kpis["avg_resolve_days"],
        help="Average age (days) at time of resolution (demo assumption).",
    )

    # Exception type distribution
    chart1 = (
        alt.Chart(
            df[
                df.status.isin(
                    ["Open", "In Progress", "Ready for Write-back", "Resolved"]
                )
            ]
        )
        .mark_bar()
        .encode(
            x=alt.X("exception_type:N", title="Exception Type"),
            y=alt.Y("count():Q", title="Count"),
            color="exception_type:N",
        )
        .properties(height=320, title="Exceptions by Type")
    )

    # Age vs Expected Amount scatter by severity
    chart2 = (
        alt.Chart(df[df.status.isin(["Open", "In Progress"])])
        .mark_circle(size=80)
        .encode(
            x=alt.X("days_since:Q", title="Days Since Posting"),
            y=alt.Y("amount_expected:Q", title="Expected Amount ($)"),
            color="severity:N",
            tooltip=[
                "claim_id",
                "exception_type",
                "severity",
                "days_since",
                "amount_expected",
            ],
        )
        .properties(height=320, title="Open Queue – Age vs. Amount")
    )

    st.altair_chart(chart1, use_container_width=True)
    st.altair_chart(chart2, use_container_width=True)


def render_posting_ledger():
    st.title("Posting Ledger – Phase 1 Financial Integrity View")
    ledger = st.session_state.ledger_df

    col1, col2 = st.columns(2)
    with col1:
        show_mismatch_only = st.checkbox(
            "Show mismatches only (variance / exception-driven)",
            value=True,
        )
    with col2:
        payer_filter = st.multiselect(
            "Filter by Payer",
            options=sorted(ledger["payer_canonical"].unique()),
            default=sorted(ledger["payer_canonical"].unique()),
        )

    view = ledger.copy()
    if show_mismatch_only:
        view = view[view["is_mismatch"]]
    if payer_filter:
        view = view[view["payer_canonical"].isin(payer_filter)]

    st.caption(
        "Each row represents a posting ledger line aggregated for demo. "
        "Variance highlights where write-back or manual follow-up is needed."
    )
    st.dataframe(view.reset_index(drop=True), use_container_width=True, height=380)

    st.markdown("---")
    st.subheader("Ledger Variance Distribution")

    chart = (
        alt.Chart(view)
        .mark_bar()
        .encode(
            x=alt.X("balance_variance:Q", bin=alt.Bin(maxbins=30), title="Balance Variance ($)"),
            y=alt.Y("count():Q", title="Count"),
            color="is_mismatch:N",
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def render_claim_timelines():
    st.title("Claim Lifecycle Timelines (Demo)")
    df = st.session_state.exceptions_df.copy()

    st.caption(
        "Visualizing claim journey from Date of Service → Posting → Current Age "
        "(based on synthetic dates)."
    )

    # Select a subset to keep chart readable
    sample = df.sample(n=min(40, len(df)), random_state=SEED).copy()
    sample["service_date_dt"] = pd.to_datetime(sample["service_date"])
    sample["posting_date_dt"] = pd.to_datetime(sample["posting_date"])
    sample["today"] = pd.to_datetime(datetime.utcnow().date())
    sample["end_date_dt"] = sample["today"]

    # Encode as horizontal bar segments: service -> posting -> today
    base = alt.Chart(sample).encode(
        y=alt.Y("claim_id:N", title="Claim ID", sort="-x"),
        tooltip=[
            "claim_id",
            "patient",
            "exception_type",
            "severity",
            "service_date",
            "posting_date",
            "days_since",
            "status",
        ],
    )

    service_to_post = base.mark_bar(color="#94b3ff").encode(
        x=alt.X("service_date_dt:T", title="Timeline"),
        x2="posting_date_dt:T",
    )

    post_to_today = base.mark_bar(color="#ffb347").encode(
        x="posting_date_dt:T",
        x2="end_date_dt:T",
    )

    chart = (service_to_post + post_to_today).properties(height=500)
    st.altair_chart(chart, use_container_width=True)


def render_payer_resolver():
    st.title("Canonical Payer Resolver (Demo Utility)")
    df = st.session_state.exceptions_df
    st.caption(
        "Type a payer label as seen in EMR or ERA. The resolver suggests a canonical payer "
        "name using fuzzy matching against known payers."
    )

    payer_input = st.text_input("Enter payer name or snippet", "")
    canonical_options = sorted(df["payer_canonical"].unique())

    if payer_input:
        # Simple fuzzy match using difflib
        best_matches = difflib.get_close_matches(
            payer_input, canonical_options, n=3, cutoff=0.3
        )
        if best_matches:
            st.success(
                f"Suggested canonical payer: **{best_matches[0]}** "
                + (f"(alternatives: {', '.join(best_matches[1:])})" if len(best_matches) > 1 else "")
            )
        else:
            st.warning("No close match found. You may need to add a new canonical payer entry.")

    st.markdown("---")
    st.subheader("Existing Canonical Payers (Demo)")
    st.write(", ".join(canonical_options))


def _render_wizard(claim_id: str):
    """
    Guided remediation wizard – maintains per-claim checklist in session_state.wizard_state.
    All steps must be completed before marking as Resolved.
    """
    steps = [
        "Reviewed payer & canonical mapping",
        "Verified ERA/EOB amounts vs ledger",
        "Tagged root cause correctly",
        "Documented remediation notes (if needed)",
    ]
    if claim_id not in st.session_state.wizard_state:
        st.session_state.wizard_state[claim_id] = {s: False for s in steps}

    st.markdown("#### Guided Remediation Checklist")
    all_done = True
    for step in steps:
        current = st.session_state.wizard_state[claim_id][step]
        new_val = st.checkbox(step, value=current, key=f"{claim_id}_{step}")
        st.session_state.wizard_state[claim_id][step] = new_val
        if not new_val:
            all_done = False

    if not all_done:
        st.info(
            "Complete all checklist items before marking this claim as **Resolved**. "
            "Supervisor can override by staging as 'Ready for Write-back'."
        )
    return all_done


def render_workbench():
    st.title("Exceptions Workbench")
    df = st.session_state.exceptions_df

    f1, f2, f3, f4 = st.columns(4)
    type_filter = f1.multiselect("Exception Type", options=EX_TYPES, default=EX_TYPES)
    sev_filter = f2.multiselect("Severity", options=SEVERITY, default=SEVERITY)
    status_filter = f3.multiselect(
        "Status",
        options=STATUS_CHOICES,
        default=["Open", "In Progress", "Ready for Write-back", "Resolved"],
    )
    search = f4.text_input("Search (claim, patient, payer)")

    view = df[
        (df["exception_type"].isin(type_filter))
        & (df["severity"].isin(sev_filter))
        & (df["status"].isin(status_filter))
    ].copy()

    if search:
        s = search.lower()
        view = view[
            view.claim_id.str.lower().str.contains(s)
            | view.patient.str.lower().str.contains(s)
            | view.payer_read.str.lower().str.contains(s)
            | view.payer_canonical.str.lower().str.contains(s)
        ]

    st.caption(
        "Filter, search, then select a claim to triage. Guided wizard ensures consistent remediation "
        "before closure."
    )
    st.dataframe(view.reset_index(drop=True), use_container_width=True, height=320)

    st.markdown("---")
    st.subheader("Triage Panel")

    if view.empty:
        st.info("No records match the current filters. Adjust filters to take action.")
        return

    selected_claim = st.selectbox(
        "Choose a claim to triage", options=view.claim_id.tolist()
    )
    row = df[df.claim_id == selected_claim].iloc[0].to_dict()

    a, b, c, d = st.columns([1.3, 1.3, 1, 1])
    a.metric("Claim", row["claim_id"])
    b.metric("Patient", row["patient"])
    c.metric("Expected", f"${row['amount_expected']}")
    d.metric("Paid", f"${row['amount_paid']}")

    st.write(
        f"**Payer (read):** {row['payer_read']}  →  **Canonical:** **{row['payer_canonical']}**  |  "
        f"**Exception:** **{row['exception_type']}**  |  **Severity:** {row['severity']}  |  "
        f"**Days Since Posting:** {row['days_since']}"
    )

    st.info(f"Suggested remediation: {row['suggested_fix']}")

    # Guided wizard
    all_done = _render_wizard(selected_claim)

    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Status & Actions")

        current_status = row["status"]
        status_help = (
            "Operators/Supervisors can update status. Auditors have read-only access."
        )

        if can_edit():
            new_status = st.selectbox(
                "Update Status",
                STATUS_CHOICES,
                index=STATUS_CHOICES.index(current_status),
                help=status_help,
            )
        else:
            st.write(f"Status: **{current_status}** (read-only)")
            new_status = current_status

        if can_edit():
            if st.button("Apply Status Update"):
                # Prevent marking as Resolved until wizard complete (unless Supervisor)
                if (
                    new_status == "Resolved"
                    and not all_done
                    and st.session_state.role != "Supervisor"
                ):
                    st.warning(
                        "Complete all checklist steps before marking as Resolved "
                        "(or switch role to Supervisor to override)."
                    )
                else:
                    mask = st.session_state.exceptions_df.claim_id == selected_claim
                    st.session_state.exceptions_df.loc[mask, "status"] = new_status
                    st.session_state.audit_log.append(
                        (datetime.utcnow(), selected_claim, f"Status -> {new_status}")
                    )
                    st.success(f"Status updated to {new_status}")

            if can_writeback():
                if st.button("Accept Suggested Fix & Mark Ready for Write-back"):
                    mask = st.session_state.exceptions_df.claim_id == selected_claim
                    st.session_state.exceptions_df.loc[mask, "status"] = (
                        "Ready for Write-back"
                    )
                    st.session_state.audit_log.append(
                        (
                            datetime.utcnow(),
                            selected_claim,
                            "Accepted fix; ready for write-back",
                        )
                    )
                    st.success(
                        "Marked as Ready for Write-back. A summarized transaction will be staged for export."
                    )
        else:
            st.info("You are in Auditor mode – no edits allowed in this view.")

    with colB:
        st.markdown("#### Evidence")
        if row["has_image"]:
            st.image(
                "https://dummyimage.com/600x140/eeeeee/333&text=EOB+Snippet+%7C+Denial+CO-45+%7C+PR-1",
                caption="EOB/ERA snippet (placeholder)",
                use_column_width=True,
            )
        else:
            st.caption("No image evidence attached for this claim.")

        if row["has_video"]:
            st.video("https://www.w3schools.com/html/mov_bbb.mp4")
        else:
            st.caption("No walkthrough video attached for this claim.")


def render_evidence():
    st.title("Evidence Library (Demo)")
    st.caption("Attach screenshots or clips to support exception triage and audit.")

    st.image(
        "https://dummyimage.com/1000x200/eeeeee/333&text=Portal+Screenshot+%7C+No+Match+Error+%7C+Ref+ID+ABC123",
        caption="Portal screenshot – No Match error (placeholder)",
        use_column_width=True,
    )
    st.video("https://www.w3schools.com/html/mov_bbb.mp4")


def render_audit_exports():
    st.title("Audit & Exports")

    st.subheader("Session Audit Log")
    if st.session_state.audit_log:
        log_df = pd.DataFrame(
            st.session_state.audit_log, columns=["timestamp", "claim_id", "action"]
        )
        st.dataframe(log_df, use_container_width=True, height=260)
    else:
        st.info("No actions in this session yet.")

    st.markdown("---")
    st.subheader("Export – Write-back Summary (Demo)")

    df = st.session_state.exceptions_df
    export_df = df[df.status == "Ready for Write-back"][
        [
            "claim_id",
            "payer_canonical",
            "amount_expected",
            "amount_paid",
            "pr_amount",
            "disallowed",
            "exception_type",
        ]
    ].copy()

    export_target = st.radio(
        "Simulate export to:",
        options=["CSV", "Credible", "Tebra"],
        horizontal=True,
    )

    st.caption(
        "This table represents summarized transactions to update Credible/Tebra via file import or UI automation."
    )
    st.dataframe(export_df, use_container_width=True, height=220)

    st.markdown("##### Export Mapping (Demo)")
    if export_target == "CSV":
        st.write(
            "- Columns: claim_id, payer_canonical, amount_expected, amount_paid, pr_amount, disallowed, exception_type\n"
            "- Delivery: SFTP drop as `piew_writeback_summary.csv`"
        )
    elif export_target == "Credible":
        st.write(
            "- Map `claim_id` → Credible Claim Number\n"
            "- Map `payer_canonical` → Payer Master\n"
            "- Amount fields drive ledger adjustments via batch posting screen."
        )
    else:  # Tebra
        st.write(
            "- Map `claim_id` → Encounter ID\n"
            "- Map `payer_canonical` → Insurance Company\n"
            "- Amounts feed into Tebra payment posting import template."
        )

    if not export_df.empty and export_target == "CSV":
        st.download_button(
            label="Download CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="piew_writeback_summary.csv",
            mime="text/csv",
        )
    elif export_df.empty:
        st.warning(
            "No items are marked 'Ready for Write-back' yet. Use the Workbench to stage some."
        )


# -----------------------------
# Router
# -----------------------------

PAGE_RENDERERS = {
    "Overview": render_overview,
    "Dashboard": render_dashboard,
    "Posting Ledger": render_posting_ledger,
    "Exceptions Workbench": render_workbench,
    "Claim Timelines": render_claim_timelines,
    "Payer Resolver": render_payer_resolver,
    "Evidence": render_evidence,
    "Audit & Exports": render_audit_exports,
}

render_fn = PAGE_RENDERERS.get(nav)
if render_fn is not None:
    render_fn()
