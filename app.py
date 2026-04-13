"""
PredAct Bench - Human Study Interface
Task-based evaluation app for instructor interaction with AI risk prediction system.

Flow:
  1. Consent page (name, email, agree)
  2. Task selection (3 tasks)
  3. Each task: instructions + report + chat
  4. Logs saved per participant per task
  5. Post-study questionnaire

Usage:
    streamlit run app.py
"""

import streamlit as st
import json
import os
import time
from datetime import datetime
from openai import OpenAI
from tools import (
    get_student_grades,
    recalculate_grade,
    get_assignment_stats,
    filter_students,
    minimum_score_needed,
    list_all_assignments,
)


# =============================================================================
# CONFIG
# =============================================================================

REPORTS_DIR = "CS-411/reports/"
LOGS_OUTPUT_DIR = "CS-411/study_logs/"
VLLM_BASE_URL = "https://api.openai.com/v1"
MODEL = "gpt-4o-mini"
API_KEY = os.environ.get("OPENAI_API_KEY", "")

# =============================================================================
# TASK DEFINITIONS
# =============================================================================

TASKS = [
    {
        "id": "task_1",
        "title": "Task 1 — Drill-Down Investigation",
        "week": "week_5",
        "instruction": (
            "**Scenario:** You are a TA for CS 411 (Database Systems). It is currently **week 5** of the semester. "
            "The system has flagged **6 students** who may be at risk of failing.\n\n"
            "**Your task:** Pick the student you are most concerned about. Use the system to investigate their grades, "
            "quiz submissions, and performance relative to the class.\n\n"
            "**When you are done**, write a 2-sentence summary in the chat explaining what is wrong with this student "
            "and what you would recommend."
        ),
    },
    {
        "id": "task_2",
        "title": "Task 2 — What-If / Recalculation",
        "week": "week_5",
        "instruction": (
            "**Scenario:** You are still at **week 5**. One of the flagged students emails you saying they had "
            "a laptop crash during Quiz 1 and could not complete it properly.\n\n"
            "**Your task:** Ask the system to recalculate this student's grade assuming they scored **85 on Quiz 1**. "
            "Does their risk level change? Would you recommend any different intervention?\n\n"
            "**When you are done**, state in the chat whether you would grant the re-grade and why."
        ),
    },
    {
        "id": "task_3",
        "title": "Task 3 — Temporal Comparison",
        "week": "week_8",
        "instruction": (
            "**Scenario:** It is now **week 8** of the semester. The system has flagged **18 students** and "
            "intervention has been triggered.\n\n"
            "**Your task:** Compare the class performance between week 5 and week 8. "
            "Which students got worse? Which improved? Were the week 5 flagged students still at risk by week 8? "
            "Did the system's early predictions hold up?\n\n"
            "**When you are done**, state in the chat which 3 students you would contact first and why."
        ),
    },
]


# =============================================================================
# SYSTEM PROMPT & TOOLS (same as before)
# =============================================================================

SYSTEM_PROMPT = """You are an academic advising assistant for CS 411 (Database Systems).
You are currently viewing data as of: {current_week}

Your grade data contains ALL student activities from week 1 through {current_week}.
So if the instructor asks about any week from 1 to {current_week_num}, you have that data.
Each assignment has a "week" field telling you which week it belongs to.

You also have class report snapshots from these past weeks for comparison: {accessible_weeks}

You MUST use tools for ANY question about grades, scores, or student data. Never guess or say data is unavailable without trying the tool first.

Available tools:
- get_student_grades: Look up any student's full grade record. Returns all assignments from week 1 to current.
- recalculate_grade: Recalculate grades after dropping or overriding scores.
- get_assignment_stats: Get class-wide statistics for any assignment.
- filter_students: Find students above or below a score threshold on any assignment.
- minimum_score_needed: Calculate what score a student needs to reach a target grade.
- list_all_assignments: List all assignments with types, weights, and weeks.

IMPORTANT RULES:
- ALWAYS call a tool before answering any data question.
- Never make up grades, scores, or student records.
- If asked about a specific week, call get_student_grades and filter by the week field.
- When comparing risk groups across weeks, use the report snapshots below.
- When showing SQL queries, format them in code blocks.
- Keep responses concise and data-driven.

Here are the class report snapshots:
{all_reports_json}
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_student_grades",
            "description": "Get a student's full grade record including all assignment scores, weights, and weighted average.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "string", "description": "The student ID, e.g. syn_026741"}
                },
                "required": ["student_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recalculate_grade",
            "description": "Recalculate a student's weighted average after dropping assignments or overriding scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "string", "description": "The student ID"},
                    "drop": {"type": "array", "items": {"type": "string"}, "description": "Assignment names to drop, e.g. ['Quiz 1']"},
                    "override": {"type": "object", "additionalProperties": {"type": "number"}, "description": "Assignment name to new score, e.g. {'Quiz 1': 85}"}
                },
                "required": ["student_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_assignment_stats",
            "description": "Get class-wide statistics for an assignment (average, min, max, distribution).",
            "parameters": {
                "type": "object",
                "properties": {
                    "assignment_name": {"type": "string", "description": "The assignment name, e.g. 'Quiz 1', 'Homework 2'"}
                },
                "required": ["assignment_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_students",
            "description": "Find students who scored above or below a threshold on an assignment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "assignment_name": {"type": "string", "description": "The assignment name"},
                    "threshold": {"type": "number", "description": "The score threshold"},
                    "direction": {"type": "string", "enum": ["below", "above"], "description": "below or above"}
                },
                "required": ["assignment_name", "threshold", "direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "minimum_score_needed",
            "description": "Calculate minimum score needed on remaining work to reach a target grade.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "string", "description": "The student ID"},
                    "target_grade": {"type": "string", "enum": ["a", "b", "c", "d"], "description": "Target letter grade"}
                },
                "required": ["student_id", "target_grade"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_all_assignments",
            "description": "List all assignments with names, types, weights, and weeks.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]


# =============================================================================
# HELPERS
# =============================================================================

def get_week_number(week_key):
    try:
        return int(week_key.split("_")[1])
    except (IndexError, ValueError):
        return 0


def filter_by_accessible_weeks(data_dict, current_week_key):
    current_num = get_week_number(current_week_key)
    return {k: v for k, v in data_dict.items() if get_week_number(k) <= current_num}


def save_task_log(participant, task_id, messages, start_time):
    """Save chat log for a participant's task."""
    os.makedirs(LOGS_OUTPUT_DIR, exist_ok=True)
    log = {
        "participant": participant,
        "task_id": task_id,
        "start_time": start_time,
        "end_time": datetime.now().isoformat(),
        "duration_seconds": round(time.time() - st.session_state.get(f"{task_id}_start_ts", time.time())),
        "num_turns": len([m for m in messages if m["role"] == "user"]),
        "messages": messages,
    }
    filename = f"{participant['name'].replace(' ', '_')}_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(LOGS_OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    return filepath


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_available_reports():
    reports = {}
    if not os.path.exists(REPORTS_DIR):
        return reports
    for f in sorted(os.listdir(REPORTS_DIR)):
        if f.endswith("_enriched.json"):
            path = os.path.join(REPORTS_DIR, f)
            report = load_json_file(path)
            overview = report.get("class_overview", {})
            week = overview.get("week", "?")
            label = f"{overview.get('course_name', '?')} - {week}"
            grades_file = f.replace("_report_enriched.json", "_grades_lookup.json")
            grades_path = os.path.join(REPORTS_DIR, grades_file)
            reports[label] = {
                "report_path": path,
                "grades_path": grades_path if os.path.exists(grades_path) else None,
                "week_key": week,
            }
    return reports


@st.cache_data
def load_all_reports_context(available_reports):
    all_reports = {}
    for label, info in available_reports.items():
        r = load_json_file(info["report_path"])
        week = r.get("class_overview", {}).get("week", "?")
        all_reports[week] = {
            "class_overview": r.get("class_overview"),
            "risk_groups": {k: v for k, v in r.get("risk_groups", {}).items() if k != "no_risk"},
            "intervention": r.get("intervention"),
            "quiz_reference": r.get("quiz_reference"),
        }
    return all_reports


def get_grades_for_week(available_reports, week_key):
    """Get the grades lookup for a specific week."""
    for label, info in available_reports.items():
        if info["week_key"] == week_key and info.get("grades_path"):
            return load_json_file(info["grades_path"])
    return None


def get_report_for_week(available_reports, week_key):
    """Get the enriched report for a specific week."""
    for label, info in available_reports.items():
        if info["week_key"] == week_key:
            return load_json_file(info["report_path"])
    return None


# =============================================================================
# TOOL EXECUTION
# =============================================================================

def execute_tool(tool_name, arguments, grades_lookup):
    try:
        if tool_name == "get_student_grades":
            result = get_student_grades(arguments["student_id"], grades_lookup)
        elif tool_name == "recalculate_grade":
            result = recalculate_grade(arguments["student_id"], grades_lookup, drop=arguments.get("drop"), override=arguments.get("override"))
        elif tool_name == "get_assignment_stats":
            result = get_assignment_stats(arguments["assignment_name"], grades_lookup)
        elif tool_name == "filter_students":
            result = filter_students(arguments["assignment_name"], arguments["threshold"], grades_lookup, direction=arguments.get("direction", "below"))
        elif tool_name == "minimum_score_needed":
            result = minimum_score_needed(arguments["student_id"], arguments["target_grade"], grades_lookup)
        elif tool_name == "list_all_assignments":
            result = list_all_assignments(grades_lookup)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
        return json.dumps(result, indent=1, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# LLM
# =============================================================================

def get_llm_response(chat_history, all_reports_json, grades_lookup, current_week_key, accessible_weeks_str):
    try:
        client = OpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)
        system_msg = SYSTEM_PROMPT.format(
            current_week=current_week_key,
            current_week_num=get_week_number(current_week_key),
            accessible_weeks=accessible_weeks_str,
            all_reports_json=all_reports_json,
        )
        messages = [{"role": "system", "content": system_msg}]
        messages.extend(chat_history)

        for iteration in range(3):
            print(f"\n  === ITERATION {iteration + 1} ===")
            response = client.chat.completions.create(
                model=MODEL, messages=messages,
                tools=TOOL_DEFINITIONS if grades_lookup else None,
                temperature=0.1, max_tokens=4096,
            )
            choice = response.choices[0]
            message = choice.message
            print(f"  FINISH REASON: {choice.finish_reason}")
            print(f"  HAS TOOL CALLS: {bool(message.tool_calls)}")

            if not message.tool_calls:
                text = message.content
                if text is None:
                    return "No response generated. Try rephrasing your question."
                return text.strip()

            tool_results_text = []
            for tc in message.tool_calls:
                tool_name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                print(f"  TOOL: {tool_name}({json.dumps(arguments)})")
                result = execute_tool(tool_name, arguments, grades_lookup)
                print(f"  TOOL RESULT: {result[:300]}")
                tool_results_text.append(f"Result of {tool_name}: {result}")

            combined_results = "\n\n".join(tool_results_text)
            messages.append({"role": "assistant", "content": f"I called the following tools:\n{combined_results}\n\nNow let me answer based on these results."})

            response2 = client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0.1, max_tokens=4096,
            )
            text = response2.choices[0].message.content
            if text is None:
                return "No response generated. Try rephrasing your question."
            return text.strip()

        return "Tool calling exceeded maximum iterations."
    except Exception as e:
        return f"Error connecting to LLM: {e}"


# =============================================================================
# PAGES
# =============================================================================

def page_consent():
    """Welcome and consent page."""
    st.title("PredAct — Human Study")
    st.markdown("---")

    st.markdown("""
    ### Welcome!

    Thank you for participating in this study. You will interact with an AI-powered academic risk 
    prediction system for **CS 411 (Database Systems)**.

    The system has analyzed student performance data and flagged students who may be at risk. 
    Your job is to use the system to investigate, ask questions, and make decisions — just like 
    a real TA or instructor would.

    **You will complete 3 tasks** (~25 minutes total):
    1. **Drill-Down Investigation** — Investigate a flagged student's performance
    2. **What-If Recalculation** — Simulate a grade change and assess impact
    3. **Temporal Comparison** — Compare class performance across two points in the semester

    Your interactions (chat messages) will be logged for research purposes.
    """)

    st.markdown("---")
    st.subheader("Participant Information")

    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    consent = st.checkbox("I consent to having my interactions logged for research purposes. I understand I can stop at any time.")

    if st.button("Start Study", type="primary", disabled=not (name and email and consent)):
        st.session_state.participant = {"name": name, "email": email}
        st.session_state.consented = True
        st.session_state.current_page = "task_select"
        st.session_state.completed_tasks = []
        st.rerun()


def page_task_select():
    """Task selection page."""
    st.title("PredAct — Select a Task")
    st.markdown(f"**Participant:** {st.session_state.participant['name']}")
    st.markdown("---")

    completed = st.session_state.get("completed_tasks", [])

    TASK_DESCRIPTIONS = {
        "task_1": "You'll review a class of 940 students where 6 have been flagged as at-risk at **week 5**. Pick the student you're most worried about and use the chat to dig into their grades, quiz performance, and how they compare to the rest of the class.",
        "task_2": "A flagged student claims they had a laptop crash during Quiz 1. You'll ask the system to recalculate their grade with a new score and decide whether it changes your assessment of their risk.",
        "task_3": "Jump forward to **week 8** where 18 students are now flagged. Compare how the class changed between week 5 and week 8 — who got worse, who improved, and whether the early warnings were accurate.",
    }

    for i, task in enumerate(TASKS):
        is_done = task["id"] in completed
        icon = "✅" if is_done else "📋"
        status = " *(completed)*" if is_done else ""

        with st.container():
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"### {icon} {task['title']}{status}")
                st.markdown(TASK_DESCRIPTIONS.get(task["id"], ""))
                st.caption(f"Week: {task['week']} | {'Completed' if is_done else 'Not started'}")
            with col2:
                if st.button("Start" if not is_done else "Redo", key=f"btn_{task['id']}"):
                    st.session_state.current_page = task["id"]
                    st.session_state[f"{task['id']}_messages"] = []
                    st.session_state[f"{task['id']}_start_ts"] = time.time()
                    st.session_state[f"{task['id']}_start_time"] = datetime.now().isoformat()
                    st.rerun()
            st.markdown("---")

    # Show questionnaire button if all tasks done
    if len(completed) >= len(TASKS):
        st.success("All tasks completed!")
        if st.button("Proceed to Questionnaire", type="primary"):
            st.session_state.current_page = "questionnaire"
            st.rerun()


def page_task(task):
    """Individual task page with instructions, report, and chat."""
    task_id = task["id"]
    week_key = task["week"]
    msg_key = f"{task_id}_messages"

    # Load data
    available = get_available_reports()
    report = get_report_for_week(available, week_key)
    grades_lookup = get_grades_for_week(available, week_key)
    all_reports_context = load_all_reports_context(available)
    accessible_reports = filter_by_accessible_weeks(all_reports_context, week_key)
    accessible_weeks = sorted(accessible_reports.keys())
    current_num = get_week_number(week_key)

    if report is None:
        st.error(f"No report found for {week_key}. Run the pipeline first.")
        return

    # Header
    st.title(f"PredAct — {task['title']}")

    # Back button
    if st.sidebar.button("← Back to Tasks"):
        st.session_state.current_page = "task_select"
        st.rerun()

    # Sidebar info
    overview = report.get("class_overview", {})
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Course:** {overview.get('course_name', '?')}")
    st.sidebar.markdown(f"**Week:** {overview.get('week', '?')}")
    st.sidebar.markdown(f"**Students:** {overview.get('total_students', '?')}")
    st.sidebar.markdown(f"**Flagged:** {overview.get('flagged_student_count', 0)}")
    st.sidebar.markdown(f"**Grade data:** weeks 1-{current_num}")
    st.sidebar.markdown(f"**Report snapshots:** {', '.join(accessible_weeks)}")

    if st.sidebar.button("Reset Chat"):
        st.session_state[msg_key] = []
        st.rerun()

    # Complete task button
    if st.sidebar.button("Complete Task ✓", type="primary"):
        messages = st.session_state.get(msg_key, [])
        filepath = save_task_log(
            st.session_state.participant,
            task_id,
            messages,
            st.session_state.get(f"{task_id}_start_time", ""),
        )
        if task_id not in st.session_state.completed_tasks:
            st.session_state.completed_tasks.append(task_id)
        st.sidebar.success(f"Saved to {filepath}")
        st.session_state.current_page = "task_select"
        st.rerun()

    # Instructions
    with st.expander("📋 Task Instructions", expanded=True):
        st.markdown(task["instruction"])

    # Tabs: Report + Chat
    tab_report, tab_chat = st.tabs(["📊 Report", "💬 Drill-Down Chat"])

    with tab_report:
        render_class_overview(overview)
        st.divider()
        render_risk_groups(report.get("risk_groups", {}))
        st.divider()
        render_intervention(report.get("intervention", {}))

    with tab_chat:
        st.caption(f"Currently at **{week_key}**. Grade data covers weeks 1-{current_num}. Report snapshots: {', '.join(accessible_weeks)}.")

        # Initialize messages
        if msg_key not in st.session_state:
            st.session_state[msg_key] = []

        # Display messages
        for msg in st.session_state[msg_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask about any student, assignment, or run scenarios..."):
            st.session_state[msg_key].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    accessible_reports_json = json.dumps(accessible_reports, indent=1, ensure_ascii=False)
                    response = get_llm_response(
                        st.session_state[msg_key],
                        accessible_reports_json,
                        grades_lookup,
                        week_key,
                        ", ".join(accessible_weeks),
                    )
                    st.markdown(response)

            st.session_state[msg_key].append({"role": "assistant", "content": response})


def page_questionnaire():
    """Post-study questionnaire."""
    st.title("PredAct — Post-Study Questionnaire")
    st.markdown(f"**Participant:** {st.session_state.participant['name']}")
    st.markdown("---")

    st.markdown("Please rate the following on a scale of 1 (Strongly Disagree) to 7 (Strongly Agree):")

    questions = [
        "The system helped me understand which students need attention.",
        "The grade predictions were trustworthy.",
        "The drill-down tools were useful for investigation.",
        "I could find the information I needed.",
        "I would use this system in my own course.",
        "The system saved me time compared to checking a spreadsheet.",
        "I felt confident making intervention decisions based on this system.",
    ]

    responses = {}
    for i, q in enumerate(questions):
        responses[f"q{i+1}"] = st.slider(f"Q{i+1}: {q}", 1, 7, 4, key=f"likert_{i}")

    st.markdown("---")
    st.markdown("**Open-ended feedback:**")
    feedback = st.text_area("What was missing, frustrating, or could be improved?", height=150)

    if st.button("Submit Questionnaire", type="primary"):
        # Save questionnaire
        os.makedirs(LOGS_OUTPUT_DIR, exist_ok=True)
        result = {
            "participant": st.session_state.participant,
            "timestamp": datetime.now().isoformat(),
            "likert_responses": responses,
            "questions": {f"q{i+1}": q for i, q in enumerate(questions)},
            "open_feedback": feedback,
        }
        name_clean = st.session_state.participant["name"].replace(" ", "_")
        filepath = os.path.join(LOGS_OUTPUT_DIR, f"{name_clean}_questionnaire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        st.session_state.current_page = "done"
        st.rerun()


def page_done():
    """Thank you page."""
    st.title("PredAct — Thank You!")
    st.balloons()
    st.markdown(f"""
    ### Thank you, {st.session_state.participant['name']}!

    Your responses have been saved. We appreciate your time and feedback.

    If you have any questions about this study, please contact the research team.
    """)


# =============================================================================
# UI COMPONENTS (report rendering)
# =============================================================================

def render_class_overview(overview):
    st.subheader("Class Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", overview.get("total_students", "?"))
    col2.metric("Average GPA", overview.get("average_gpa", "?"))
    col3.metric("Flagged Students", overview.get("flagged_student_count", 0))
    col4.metric("Grade Trend", overview.get("grade_trend", "?"))
    st.caption(f"Common Issue: **{overview.get('common_issue', 'none')}** | Week: **{overview.get('week', '?')}** | Term: **{overview.get('term', '?')}**")


def render_risk_groups(risk_groups):
    st.subheader("Risk Groups")
    no_risk = risk_groups.get("no_risk", {})
    if no_risk:
        st.success(f"**No Risk:** {no_risk.get('count', 0)} students — predicted grade: {no_risk.get('predicted_grade', '?')}")

    risk_order = ["critical_risk", "high_risk", "medium_risk", "unknown_risk"]
    risk_colors = {"critical_risk": "🔴", "high_risk": "🟠", "medium_risk": "🟡", "unknown_risk": "⚪"}

    for risk_key in risk_order:
        group = risk_groups.get(risk_key)
        if not group:
            continue
        icon = risk_colors.get(risk_key, "⚪")
        count = group.get("count", 0)
        grade = group.get("predicted_grade", "?")
        risk = group.get("failure_risk", "?")

        with st.expander(f"{icon} {risk_key} — {count} student(s) | grade: {grade} | risk: {risk}", expanded=True):
            for student in group.get("students", []):
                render_student_card(student)


def render_student_card(student):
    sid = student.get("student_id", "?")
    pred_grade = student.get("predicted_grade", "?")
    reason = student.get("failure_risk_reason", "?")
    missing = student.get("missing_assignments", 0)
    real_id = student.get("mapped_real_student", None)

    st.markdown(f"**{sid}** — predicted: `{pred_grade}` | reason: `{reason}` | missing: `{missing}`")
    if real_id:
        st.caption(f"Mapped to real student: {real_id}")

    sql = student.get("sql_submissions", {})
    if sql:
        for qkey, qdata in sql.items():
            attempts = qdata.get("false_attempts", 0)
            concepts = ", ".join(qdata.get("concept_tags", []))
            quiz = qdata.get("quiz", "?")
            with st.expander(f"📝 {qkey} ({quiz}) — {attempts} wrong attempts | concepts: {concepts}"):
                st.markdown("**Question:**")
                st.text(qdata.get("question_text", "N/A")[:500])
                st.markdown("**Correct Solution:**")
                st.code(qdata.get("correct_solution", "N/A"), language="sql")
                submissions = qdata.get("submissions", [])
                if submissions:
                    st.markdown(f"**Student's Wrong Submissions** (showing last 3 of {len(submissions)}):")
                    for sub in submissions[-3:]:
                        st.code(sub.get("query", "N/A"), language="sql")
    st.divider()


def render_intervention(intervention):
    st.subheader("Intervention Status")
    if "no_intervention" in intervention:
        info = intervention["no_intervention"]
        st.warning(
            f"**No intervention triggered.** Reason: {info.get('reason', '?')}\n\n"
            f"Current week: {info.get('current_week', '?')} | At-risk threshold: {info.get('atrisk_approx_week', '?')}\n\n"
            f"Recommendation: {info.get('recommendation', 'none')}"
        )
    else:
        for risk_key, plan in intervention.items():
            if not isinstance(plan, dict):
                continue
            sids = plan.get("student_ids", [])
            st.error(f"**{risk_key}** — {len(sids)} student(s) | Priority: {plan.get('priority', '?')} | Contact: {plan.get('contact_mode', '?')}")
            for sid in sids:
                itype = plan.get("intervention_type", {}).get(sid, "?")
                igoal = plan.get("intervention_goal", {}).get(sid, "?")
                st.markdown(f"- `{sid}`: type={itype}, goal={igoal}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.set_page_config(page_title="PredAct - Human Study", layout="wide")

    st.markdown("""
        <style>
        .stChatInput {
            position: fixed;
            bottom: 0;
            width: 100%;
            z-index: 999;
            background: var(--background-color);
            padding-bottom: 10px;
        }
        .stChatMessageContainer {
            padding-bottom: 80px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "consent"

    # Route to correct page
    page = st.session_state.current_page

    if page == "consent":
        page_consent()
    elif page == "task_select":
        page_task_select()
    elif page == "questionnaire":
        page_questionnaire()
    elif page == "done":
        page_done()
    else:
        # Must be a task page
        task = next((t for t in TASKS if t["id"] == page), None)
        if task:
            page_task(task)
        else:
            st.session_state.current_page = "consent"
            st.rerun()


if __name__ == "__main__":
    main()