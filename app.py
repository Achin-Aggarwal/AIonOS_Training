from dotenv import load_dotenv
import os

load_dotenv()
import re
import time
import sqlite3
import smtplib
import uuid
from datetime import datetime
from typing import Optional, Dict, List, TypedDict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
from tqdm import tqdm
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from langgraph.graph import StateGraph, END

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EXCEL_FILE = "tickets.xlsx"
SQLITE_DB = "requests.db"
ADMIN_EMAIL = "achinaggarwal01@gmail.com"

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER")  
EMAIL_PASS = os.getenv("EMAIL_PASS")  
BASE_URL = os.getenv("BASE_URL", "http://localhost:8501")  

EXCEL_COLUMNS = ["TicketID", "User", "Software", "Status", "CreatedAt", "Feedback", "ApprovalToken"]

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS request_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT,
    user_name TEXT,
    software TEXT,
    status TEXT,
    action TEXT,
    details TEXT,
    timestamp TEXT
);

CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT,
    approval_token TEXT UNIQUE,
    status TEXT DEFAULT 'pending',
    approved_by TEXT,
    approved_at TEXT,
    created_at TEXT
);
"""



def ensure_excel():
    if not os.path.exists(EXCEL_FILE):
        pd.DataFrame(columns=EXCEL_COLUMNS).to_excel(EXCEL_FILE, index=False)

def ensure_sqlite():
    conn = sqlite3.connect(SQLITE_DB)
    conn.executescript(SQLITE_SCHEMA)
    conn.close()

def generate_ticket_id():
    now = datetime.now()
    date_tag = now.strftime("%Y%m%d")
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
        same_day = df[df["TicketID"].astype(str).str.contains(date_tag, na=False)]
        seq = len(same_day) + 1
    else:
        seq = 1
    return f"SNW-{date_tag}-{seq:03d}"

def generate_approval_token():
    return str(uuid.uuid4())

def parse_install_request(text: str) -> Optional[Dict[str, str]]:
    pattern = re.compile(r"install\s+([a-zA-Z0-9\-\s\._]+?)\s*(?:version|v)?\s*(\d+[\w\.]*)?", re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        return None
    return {"software": m.group(1).strip().title(), "version": (m.group(2) or "latest").strip()}

def get_llm():
    if ChatGroq and os.getenv("GROQ_API_KEY"):
        return ChatGroq(model=GROQ_MODEL, temperature=0)
    if ChatOpenAI and os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    raise RuntimeError("No LLM available. Set GROQ_API_KEY or OPENAI_API_KEY.")

def send_approval_email(ticket_id: str, user: str, software: str, version: str, approval_token: str):
    """Send approval email to admin"""
    if not EMAIL_USER or not EMAIL_PASS:
        st.warning("Email credentials not configured. Approval email cannot be sent.")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = ADMIN_EMAIL
        msg['Subject'] = f"Software Installation Request - Ticket {ticket_id}"
        
        approve_url = f"{BASE_URL}?action=approve&token={approval_token}"
        reject_url = f"{BASE_URL}?action=reject&token={approval_token}"
        
        body = f"""
        <html>
        <body>
            <h2>Software Installation Request</h2>
            <p><strong>Ticket ID:</strong> {ticket_id}</p>
            <p><strong>User:</strong> {user}</p>
            <p><strong>Software:</strong> {software} {version}</p>
            <p><strong>Request Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h3>Actions Required:</h3>
            <p>Please click one of the following links to approve or reject this request:</p>
            
            <div style="margin: 20px 0;">
                <a href="{approve_url}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-right: 10px;">APPROVE</a>
                <a href="{reject_url}" style="background-color: #f44336; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">REJECT</a>
            </div>
            
            <p><em>This is an automated email from SNW Installer Agent.</em></p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        text = msg.as_string()
        server.sendmail(EMAIL_USER, ADMIN_EMAIL, text)
        server.quit()
        
        return True
    except Exception as e:
        st.error(f"Failed to send approval email: {str(e)}")
        return False

def check_approval_status(ticket_id: str):
    """Check if ticket is approved, rejected, or pending"""
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM approvals WHERE ticket_id = ?", (ticket_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else "pending"

def update_ticket_status(ticket_id: str, status: str):
    """Update ticket status in Excel"""
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
        df.loc[df['TicketID'] == ticket_id, 'Status'] = status
        df.to_excel(EXCEL_FILE, index=False)

def get_ticket_details(ticket_id: str):
    """Get ticket details from Excel"""
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
        ticket_row = df[df['TicketID'] == ticket_id]
        if not ticket_row.empty:
            return ticket_row.iloc[0].to_dict()
    return None


@tool
def log_ticket_to_excel(user: str, software: str, version: str, status: str = "Pending Approval") -> str:
    """Log a new ticket to Excel. Returns TicketID."""
    ensure_excel()
    ticket_id = generate_ticket_id()
    approval_token = generate_approval_token()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    row = {
        "TicketID": ticket_id,
        "User": user,
        "Software": f"{software} {version}",
        "Status": status,
        "CreatedAt": created_at,
        "Feedback": "",
        "ApprovalToken": approval_token,
    }
    df = pd.read_excel(EXCEL_FILE)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)
    
    conn = sqlite3.connect(SQLITE_DB)
    conn.execute(
        "INSERT INTO approvals (ticket_id, approval_token, created_at) VALUES (?, ?, ?)",
        (ticket_id, approval_token, created_at)
    )
    conn.commit()
    conn.close()
    
    send_approval_email(ticket_id, user, software, version, approval_token)
    
    return ticket_id

@tool
def log_step_to_sqlite(ticket_id: str, user: str, software: str, status: str, action: str, details: str) -> str:
    """Append a step to SQLite logs."""
    ensure_sqlite()
    conn = sqlite3.connect(SQLITE_DB)
    with conn:
        conn.execute(
            "INSERT INTO request_logs (ticket_id, user_name, software, status, action, details, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ticket_id, user, software, status, action, details, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
    conn.close()
    return f"{action} logged"

@tool
def run_dummy_installation(ticket_id: str, software: str) -> str:
    """Simulate installation (progress shown in Streamlit)."""
    return f"Installation started for {software} (ticket {ticket_id})."

@tool
def check_ticket_approval(ticket_id: str) -> str:
    """Check if ticket is approved for installation."""
    status = check_approval_status(ticket_id)
    return status

def handle_approval_response():
    """Handle approval/rejection from URL parameters"""
    query_params = st.query_params
    
    if "action" in query_params and "token" in query_params:
        action = query_params["action"]
        token = query_params["token"]
        
        conn = sqlite3.connect(SQLITE_DB)
        cursor = conn.cursor()
        
        cursor.execute("SELECT ticket_id FROM approvals WHERE approval_token = ? AND status = 'pending'", (token,))
        result = cursor.fetchone()
        
        if result:
            ticket_id = result[0]
            
            if action == "approve":
                cursor.execute(
                    "UPDATE approvals SET status = 'approved', approved_by = ?, approved_at = ? WHERE approval_token = ?",
                    (ADMIN_EMAIL, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), token)
                )
                
                update_ticket_status(ticket_id, "Approved")
                
                cursor.execute(
                    "INSERT INTO request_logs (ticket_id, user_name, software, status, action, details, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ticket_id, "Admin", "", "Approved", "Admin Approval", f"Ticket {ticket_id} approved by admin", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
                
                conn.commit()
                st.success(f"✅ Ticket {ticket_id} has been APPROVED! Installation will proceed.")
                st.info("You can close this window. The user will see the approval status automatically.")
                
            elif action == "reject":
                cursor.execute(
                    "UPDATE approvals SET status = 'rejected', approved_by = ?, approved_at = ? WHERE approval_token = ?",
                    (ADMIN_EMAIL, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), token)
                )
                
                update_ticket_status(ticket_id, "Rejected")
                
                cursor.execute(
                    "INSERT INTO request_logs (ticket_id, user_name, software, status, action, details, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ticket_id, "Admin", "", "Rejected", "Admin Rejection", f"Ticket {ticket_id} rejected by admin", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
                
                conn.commit()
                st.error(f"❌ Ticket {ticket_id} has been REJECTED! Installation will not proceed.")
                st.info("You can close this window. The user will see the rejection status automatically.")
        else:
            st.warning("Invalid or expired approval token.")
        
        conn.close()
        
        if st.button("Close Admin Panel"):
            st.query_params.clear()
            st.rerun()

class AgentState(TypedDict):
    user: str
    prompt: str
    intent: str
    software: Optional[str]
    version: Optional[str]
    ticket_id: Optional[str]
    answer: Optional[str]
    events: List[str]

def classify_node(state: AgentState) -> AgentState:
    parsed = parse_install_request(state["prompt"])
    if parsed:
        state["intent"] = "install"
        state["software"] = parsed["software"]
        state["version"] = parsed["version"]
    else:
        llm = get_llm()
        chain = ChatPromptTemplate.from_messages([
            ("system", "Classify prompt as 'install' or 'simple'."),
            ("human", "{q}")
        ]) | llm | StrOutputParser()
        label = chain.invoke({"q": state["prompt"]}).lower()
        state["intent"] = "install" if "install" in label else "simple"
    return state

def answer_node(state: AgentState) -> AgentState:
    llm = get_llm()
    chain = ChatPromptTemplate.from_messages([
        ("system", "Be concise and helpful."),
        ("human", "{q}")
    ]) | llm | StrOutputParser()
    state["answer"] = chain.invoke({"q": state["prompt"]})
    return state

def ticket_node(state: AgentState) -> AgentState:
    tid = log_ticket_to_excel.invoke({
        "user": state["user"], 
        "software": state["software"], 
        "version": state["version"]
    })
    state["ticket_id"] = tid
    
    log_step_to_sqlite.invoke({
        "ticket_id": tid, 
        "user": state["user"], 
        "software": f"{state['software']} {state['version']}", 
        "status": "Pending Approval", 
        "action": "Request Created", 
        "details": f"User requested {state['software']} {state['version']}. Approval email sent to admin."
    })
    return state

def approval_check_node(state: AgentState) -> AgentState:
    """Check approval status before proceeding"""
    approval_status = check_ticket_approval.invoke({"ticket_id": state["ticket_id"]})
    state["approval_status"] = approval_status
    return state

def install_node(state: AgentState) -> AgentState:
    if state.get("approval_status") == "approved":
        run_dummy_installation.invoke({
            "ticket_id": state["ticket_id"], 
            "software": f"{state['software']} {state['version']}"
        })
        
        log_step_to_sqlite.invoke({
            "ticket_id": state["ticket_id"], 
            "user": state["user"], 
            "software": f"{state['software']} {state['version']}", 
            "status": "Installing", 
            "action": "Installation", 
            "details": "Installation initialized after admin approval"
        })
    return state

def done_node(state: AgentState) -> AgentState:
    if state.get("approval_status") == "approved":
        update_ticket_status(state["ticket_id"], "Installed")
        
        log_step_to_sqlite.invoke({
            "ticket_id": state["ticket_id"], 
            "user": state["user"], 
            "software": f"{state['software']} {state['version']}", 
            "status": "Installed", 
            "action": "Completed", 
            "details": "Installation completed successfully"
        })
    elif state.get("approval_status") == "rejected":
        log_step_to_sqlite.invoke({
            "ticket_id": state["ticket_id"], 
            "user": state["user"], 
            "software": f"{state['software']} {state['version']}", 
            "status": "Rejected", 
            "action": "Installation Cancelled", 
            "details": "Installation cancelled due to admin rejection"
        })
    return state


workflow = StateGraph(AgentState)
workflow.add_node("classify", classify_node)
workflow.add_node("answer", answer_node)
workflow.add_node("ticket", ticket_node)
workflow.add_node("approval_check", approval_check_node)
workflow.add_node("install", install_node)
workflow.add_node("done", done_node)

workflow.set_entry_point("classify")
workflow.add_conditional_edges("classify", lambda s: s["intent"], {"simple": "answer", "install": "ticket"})
workflow.add_edge("answer", END)
workflow.add_edge("ticket", "approval_check")

def approval_router(state):
    approval_status = state.get("approval_status", "pending")
    if approval_status == "approved":
        return "install"
    elif approval_status == "rejected":
        return "done"
    else:
        return "done"  

workflow.add_conditional_edges("approval_check", approval_router, {
    "install": "install",
    "done": "done"
})
workflow.add_edge("install", "done")
workflow.add_edge("done", END)

AGENT = workflow.compile()

st.set_page_config(page_title="SNW Installer Agent", page_icon="🧰")
st.title("🧰 SNW LLM Installer Agent with Email Approval")

handle_approval_response()

st.sidebar.title("📧 Email Configuration")
st.sidebar.info(f"Admin Email: {ADMIN_EMAIL}")
if EMAIL_USER:
    st.sidebar.success("✅ Email configured")
else:
    st.sidebar.warning("⚠️ Email not configured")
    st.sidebar.caption("Add EMAIL_USER and EMAIL_PASS to .env file")

st.sidebar.title("🎫 Ticket Status Monitor")
if st.sidebar.button("🔄 Refresh Status"):
    st.rerun()

user_name = st.text_input("Your name", "Alice")
user_prompt = st.text_area("Prompt", height=100)

st.subheader("🔍 Check Ticket Status")
col1, col2 = st.columns([3, 1])
with col1:
    check_ticket_id = st.text_input("Enter Ticket ID to check status:", placeholder="SNW-20240116-001")
with col2:
    if st.button("Check Status"):
        if check_ticket_id:
            ticket_details = get_ticket_details(check_ticket_id)
            approval_status = check_approval_status(check_ticket_id)
            
            if ticket_details:
                st.write(f"**Ticket:** {check_ticket_id}")
                st.write(f"**User:** {ticket_details['User']}")
                st.write(f"**Software:** {ticket_details['Software']}")
                st.write(f"**Created:** {ticket_details['CreatedAt']}")
                st.write(f"**Status:** {ticket_details['Status']}")
                
                if approval_status == "approved":
                    st.success("✅ **APPROVED** - Starting installation...")
                    
                    steps = ["Queueing job","Checking prerequisites","Downloading","Installing","Configuring","Finalizing"]
                    prog = st.progress(0)
                    status_text = st.empty()
                    
                    for i, step in enumerate(steps):
                        status_text.write(f"Step {i+1}/{len(steps)}: {step}")
                        time.sleep(0.7)
                        prog.progress(int((i+1)/len(steps)*100))
                    
                    status_text.write("")
                    st.success("🎉 **Installation Complete!**")
                    
                    update_ticket_status(check_ticket_id, "Installed")
                    
                elif approval_status == "rejected":
                    st.error("❌ **REJECTED** - Installation cancelled by administrator")
                    
                elif approval_status == "pending":
                    st.warning("⏳ **PENDING** - Waiting for admin approval...")
                    st.info("Refresh this page periodically to check for updates")
                    
            else:
                st.error("❌ Ticket not found!")

if st.button("Run", type="primary"):
    ensure_excel(); ensure_sqlite()
    state: AgentState = {
        "user": user_name, 
        "prompt": user_prompt, 
        "intent": "", 
        "software": None, 
        "version": None, 
        "ticket_id": None, 
        "answer": None, 
        "events": []
    }
    result = AGENT.invoke(state)
    
    if result["intent"] == "simple":
        st.success("Answer:")
        st.write(result["answer"])
    else:
        st.info(f"📝 Installation request created for {result['software']} {result['version']} (Ticket {result['ticket_id']})")
        
        approval_status = result.get("approval_status", "pending")
        
        if approval_status == "pending":
            st.warning("⏳ Waiting for admin approval. An email has been sent to the administrator.")
            st.info(f"💡 **Your Ticket ID:** `{result['ticket_id']}` - Save this to check status later!")
            st.info("Use the 'Check Ticket Status' section above to monitor your request.")
            
        elif approval_status == "approved":
            st.success("✅ Request approved! Starting installation...")
            
            steps = ["Queueing job","Checking prerequisites","Downloading","Installing","Configuring","Finalizing"]
            prog = st.progress(0)
            for i, step in enumerate(steps):
                st.write(f"Step {i+1}/{len(steps)}: {step}")
                time.sleep(0.7)
                prog.progress(int((i+1)/len(steps)*100))
            st.success("Installation complete ✅")
            
        elif approval_status == "rejected":
            st.error("❌ Request rejected by administrator. Installation cancelled.")
        
        st.caption("All actions logged to Excel + SQLite.")

st.subheader("📊 Recent Tickets")
if os.path.exists(EXCEL_FILE):
    df = pd.read_excel(EXCEL_FILE)
    if not df.empty:
        display_df = df.drop('ApprovalToken', axis=1, errors='ignore')
        st.dataframe(display_df.tail(10), use_container_width=True)
    else:
        st.info("No tickets found.")


if os.path.exists(EXCEL_FILE):
    df = pd.read_excel(EXCEL_FILE)
    pending_tickets = df[df['Status'].isin(['Pending Approval', 'Approved'])]
    
    if not pending_tickets.empty:
        st.info(f"📋 You have {len(pending_tickets)} active ticket(s). This page will auto-refresh every 30 seconds.")
        time.sleep(30)
        st.rerun()