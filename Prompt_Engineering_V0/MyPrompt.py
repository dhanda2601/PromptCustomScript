import json
from pathlib import Path
import streamlit as st
import logging
import asyncio
from prompt_gen.prompt_generator import PromptGenerator
from llm.llm_handler import run_llm_prompt
from config.config import CONFIG_DIR, PROMPT_TYPES_FILE, BUSINESS_RULES_FILE, DEEPSEEK_MODEL
from langchain_core.output_parsers import JsonOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----- Load Configuration Files -----
try:
    with open(PROMPT_TYPES_FILE, "r") as f:
        prompt_types = json.load(f)
    with open(BUSINESS_RULES_FILE, "r") as f:
        business_rules = json.load(f)
except FileNotFoundError as e:
    st.error(f"Configuration file missing: {e}")
    logger.error(f"Configuration file missing: {e}")
    st.stop()

# ----- Validate LLM Model Path -----
if not DEEPSEEK_MODEL.exists():
    st.error(f"LLM model not found at: {DEEPSEEK_MODEL}")
    logger.error(f"LLM model not found at: {DEEPSEEK_MODEL}")
    st.stop()

# ----- Initialize Prompt Generator -----
try:
    pg = PromptGenerator(str(DEEPSEEK_MODEL), prompt_types, business_rules)
except Exception as e:
    st.error(f"Failed to initialize Prompt Generator: {e}")
    logger.error(f"Failed to initialize Prompt Generator: {e}")
    st.stop()

# ----- Streamlit App UI -----
st.set_page_config(page_title="Prompt Generator Pro", layout="centered")
st.title("🧠 Dynamic Prompt Generator Lab")

# User Goal for Suggestion
user_goal = st.text_input("📝 What do you want to achieve?", placeholder="e.g., Summarize a text, create RFP for banking, map data columns")

# Fetch Prompt Types
prompt_types_list = pg.get_all_prompt_types()

# Domain Selector
domains = ["General", "Banking", "Healthcare", "Finance", "Retail", "Manufacturing", "Education", "Government"]
domain = st.selectbox("🌐 Select Domain", domains, index=0)

# Suggest Prompt Type Based on User Goal
if user_goal:
    meta_prompt = (
        f"Given the following prompt types: {', '.join(prompt_types_list)}, "
        f"select the most suitable prompt type for the user goal: '{user_goal}' in the {domain} domain. "
        f"Use 'Summary' for summarization tasks, 'Research-Challenges' for listing research challenges, "
        f"'Data-Mapping' for data column mapping, 'Instruction Prompting and Tuning' for response tuning, "
        f"'Presales' for RFP or sales tasks, 'HR Operations' for HR tasks, "
        f"'Learning and Knowledge' for training content, 'Finance' for financial tasks, "
        f"'Project Management' for project tasks, 'Chain-of-Thought' for step-by-step reasoning, "
        f"'Zero-Shot' for direct questions, 'Few-Shot' for classification with examples, "
        f"'Chain-of-Code' for coding tasks, 'Retrieval Augmented Generation (RAG)' for fact-based answers, "
        f"'Emotion Prompting' for emotional responses, or other relevant types. "
        f"Return only the prompt type name (e.g., 'Summary')."
    )
    try:
        suggested_type = asyncio.run(run_llm_prompt([meta_prompt]))[0].strip()
        if suggested_type in prompt_types_list:
            st.success(f"Suggested Prompt Type: {suggested_type}")
            logger.info(f"Suggested prompt type: {suggested_type} for goal: {user_goal}")
        else:
            # Rule-based fallback
            suggested_type = None
            goal_lower = user_goal.lower()
            if "summar" in goal_lower:
                suggested_type = "Summary"
            elif "research" in goal_lower and "challenge" in goal_lower:
                suggested_type = "Research-Challenges"
            elif "data" in goal_lower and "mapping" in goal_lower:
                suggested_type = "Data-Mapping"
            elif "data quality" in goal_lower or "dataset" in goal_lower:
                suggested_type = "Data Quality"
            elif "self-healing" in goal_lower or "error correction" in goal_lower:
                suggested_type = "Self-Healing"
            elif "code conversion" in goal_lower or "convert code" in goal_lower:
                suggested_type = "Code Conversion"
            elif "rfp" in goal_lower or "presales" in goal_lower or "sales" in goal_lower:
                suggested_type = "Presales"
            elif "hr" in goal_lower or "human resource" in goal_lower:
                suggested_type = "HR Operations"
            elif "learning" in goal_lower or "knowledge" in goal_lower or "training" in goal_lower:
                suggested_type = "Learning and Knowledge"
            elif "finance" in goal_lower or "budget" in goal_lower:
                suggested_type = "Finance"
            elif "project" in goal_lower or "management" in goal_lower:
                suggested_type = "Project Management"
            elif "tuning" in goal_lower or "refine" in goal_lower:
                suggested_type = "Instruction Prompting and Tuning"
            elif "solve" in goal_lower or "reason" in goal_lower:
                suggested_type = "Chain-of-Thought"
            elif "question" in goal_lower or "what" in goal_lower:
                suggested_type = "Zero-Shot"
            elif "classify" in goal_lower or "example" in goal_lower:
                suggested_type = "Few-Shot"
            elif "code" in goal_lower or "program" in goal_lower:
                suggested_type = "Chain-of-Code"
            elif "fact" in goal_lower or "retrieve" in goal_lower:
                suggested_type = "Retrieval Augmented Generation (RAG)"
            elif "emotion" in goal_lower or "feel" in goal_lower:
                suggested_type = "Emotion Prompting"
            else:
                suggested_type = "Instruction-Based"
            if suggested_type in prompt_types_list:
                st.success(f"Suggested Prompt Type (rule-based): {suggested_type}")
                logger.info(f"Rule-based suggested prompt type: {suggested_type} for goal: {user_goal}")
            else:
                st.warning("Could not determine prompt type. Please select manually.")
                logger.warning(f"Could not determine prompt type for goal: {user_goal}")
    except Exception as e:
        st.warning(f"Failed to suggest prompt type: {e}")
        logger.error(f"Failed to suggest prompt type: {e}")
        # Rule-based fallback
        suggested_type = None
        goal_lower = user_goal.lower()
        if "summar" in goal_lower:
            suggested_type = "Summary"
        elif "research" in goal_lower and "challenge" in goal_lower:
            suggested_type = "Research-Challenges"
        elif "data" in goal_lower and "mapping" in goal_lower:
            suggested_type = "Data-Mapping"
        elif "data quality" in goal_lower or "dataset" in goal_lower:
            suggested_type = "Data Quality"
        elif "self-healing" in goal_lower or "error correction" in goal_lower:
            suggested_type = "Self-Healing"
        elif "code conversion" in goal_lower or "convert code" in goal_lower:
            suggested_type = "Code Conversion"
        elif "rfp" in goal_lower or "presales" in goal_lower or "sales" in goal_lower:
            suggested_type = "Presales"
        elif "hr" in goal_lower or "human resource" in goal_lower:
            suggested_type = "HR Operations"
        elif "learning" in goal_lower or "knowledge" in goal_lower or "training" in goal_lower:
            suggested_type = "Learning and Knowledge"
        elif "finance" in goal_lower or "budget" in goal_lower:
            suggested_type = "Finance"
        elif "project" in goal_lower or "management" in goal_lower:
            suggested_type = "Project Management"
        elif "tuning" in goal_lower or "refine" in goal_lower:
            suggested_type = "Instruction Prompting and Tuning"
        elif "solve" in goal_lower or "reason" in goal_lower:
            suggested_type = "Chain-of-Thought"
        elif "question" in goal_lower or "what" in goal_lower:
            suggested_type = "Zero-Shot"
        elif "classify" in goal_lower or "example" in goal_lower:
            suggested_type = "Few-Shot"
        elif "code" in goal_lower or "program" in goal_lower:
            suggested_type = "Chain-of-Code"
        elif "fact" in goal_lower or "retrieve" in goal_lower:
            suggested_type = "Retrieval Augmented Generation (RAG)"
        elif "emotion" in goal_lower or "feel" in goal_lower:
            suggested_type = "Emotion Prompting"
        else:
            suggested_type = "Instruction-Based"
        if suggested_type in prompt_types_list:
            st.success(f"Suggested Prompt Type (rule-based): {suggested_type}")
            logger.info(f"Rule-based suggested prompt type: {suggested_type} for goal: {user_goal}")
        else:
            st.warning("Could not determine prompt type. Please select manually.")
            logger.warning(f"Could not determine prompt type for goal: {user_goal}")
else:
    suggested_type = None

# Select Prompt Type
selected_type = st.selectbox(
    "🔧 Select Prompt Type",
    prompt_types_list,
    index=prompt_types_list.index(suggested_type) if suggested_type else 0
)

# Display and Edit Prompt Information
info = pg.get_prompt_info(selected_type)
with st.expander("📄 Prompt Details"):
    st.write(f"**Description**: {info.get('description', 'No description available')}")
    st.write(f"**Help Tip**: {info.get('help_tip', 'No help tip available')}")
    st.write(f"**Last Updated**: {info.get('last_updated', 'Unknown')}")
    # Example Inputs
    example_inputs = {
        "domain": domain if domain != "General" else "Banking",
        "task": f"Analyze data in {domain.lower()} context" if domain != "General" else "Explain AI",
        "text": "Sample article about patient care" if domain == "Healthcare" else "Sample financial report",
        "word_count": "50",
        "target_table": "Customer" if domain == "Banking" else "Patient",
        "target_column": "AccountID" if domain == "Banking" else "PatientID",
        "data_type": "String",
        "business_meaning": "Unique identifier",
        "source_columns_block": "Source1.AccountNum, Source2.ID" if domain == "Banking" else "Source1.PatientNum",
        "response": f"AI improves {domain.lower()}" if domain != "General" else "AI is useful",
        "rfp_description": f"Proposal for a {domain.lower()} platform" if domain != "General" else "Proposal for a banking platform",
        "hr_task": "Write a nurse job description" if domain == "Healthcare" else "Write a job description",
        "learning_topic": "Financial literacy" if domain == "Finance" else "AI basics",
        "financial_task": "Analyze annual budget",
        "project_task": "Plan hospital IT upgrade" if domain == "Healthcare" else "Plan IT project",
        "problem": "Calculate revenue growth",
        "question": f"What is AI in {domain.lower()}?" if domain != "General" else "What is AI?",
        "input": "Classify this review",
        "examples": "Positive: Great service\nNegative: Poor experience",
        "puzzle": "Optimize resource allocation",
        "topic": f"AI in {domain.lower()}" if domain != "General" else "AI",
        "argument": "AI improves efficiency",
        "situation": "System alert detected",
        "complex_task": "Design a research project",
        "goal": "Optimize prompt for summarization",
        "tool_task": "Query a database",
        "document": "Meeting minutes",
        "factual_query": f"Latest trends in {domain.lower()}" if domain != "General" else "Latest AI trends",
        "spatial_task": "Pathfinding in a grid",
        "program_task": "Design an algorithm",
        "logic_problem": "Validate a syllogism",
        "emotion_query": "Customer complaint",
        "dataset_description": f"{domain} dataset with missing values" if domain != "General" else "Financial dataset",
        "process_description": f"{domain} pipeline failure" if domain != "General" else "Data pipeline failure",
        "source_language": "Python",
        "target_language": "Java",
        "code": "print('Hello')"
    }
    st.write("**Example Inputs**:")
    for var in info.get("input_variables", []):
        st.write(f"- `{var}`: {example_inputs.get(var, 'No example available')}")

# Edit Description
description = st.text_area(
    "✏️ Edit Description",
    value=info.get('description', 'No description available'),
    height=100,
    key=f"description_{selected_type}"
)
if st.button("💾 Save Description"):
    try:
        prompt_types[selected_type]['description'] = description
        with open(PROMPT_TYPES_FILE, "w") as f:
            json.dump(prompt_types, f, indent=2)
        st.success("Description saved successfully!")
        logger.info(f"Saved description for {selected_type}: {description}")
    except Exception as e:
        st.error(f"Failed to save description: {e}")
        logger.error(f"Failed to save description: {e}")

# Collect Input Variables Dynamically
if "variables" not in st.session_state:
    st.session_state.variables = {}
variables = st.session_state.variables

# Input Guidance
input_guidance = {
    "domain": "Select a domain like Banking or Healthcare",
    "task": "Describe the task (e.g., 'Explain AI in Banking')",
    "text": "Paste the text to summarize or analyze",
    "word_count": "Enter a number (e.g., 50)",
    "target_table": "Enter a table name (e.g., 'Customer')",
    "target_column": "Enter a column name (e.g., 'AccountID')",
    "data_type": "Specify data type (e.g., 'String')",
    "business_meaning": "Describe the column’s purpose (e.g., 'Unique identifier')",
    "source_columns_block": "List source columns (e.g., 'Source1.AccountNum, Source2.ID')",
    "response": "Paste the response to tune",
    "rfp_description": "Describe the RFP or sales task (e.g., 'Proposal for a banking platform')",
    "hr_task": "Specify the HR task (e.g., 'Write a nurse job description')",
    "learning_topic": "Enter the topic (e.g., 'Financial literacy')",
    "financial_task": "Describe the financial task (e.g., 'Analyze annual budget')",
    "project_task": "Specify the project task (e.g., 'Plan hospital IT upgrade')",
    "problem": "State the problem (e.g., 'Calculate revenue growth')",
    "question": "Ask a question (e.g., 'What is AI?')",
    "input": "Provide the input to classify",
    "examples": "List examples (e.g., 'Positive: Great service\nNegative: Poor experience')",
    "puzzle": "Describe the puzzle (e.g., 'Optimize resource allocation')",
    "topic": "Specify the topic (e.g., 'AI in Healthcare')",
    "argument": "State the argument (e.g., 'AI improves efficiency')",
    "situation": "Describe the situation (e.g., 'System alert detected')",
    "complex_task": "Describe the complex task (e.g., 'Design a research project')",
    "goal": "State the goal (e.g., 'Optimize prompt for summarization')",
    "tool_task": "Describe the tool task (e.g., 'Query a database')",
    "document": "Paste the document (e.g., 'Meeting minutes')",
    "factual_query": "Ask a factual question (e.g., 'Latest AI trends')",
    "spatial_task": "Describe the spatial task (e.g., 'Pathfinding in a grid')",
    "program_task": "Specify the programming task (e.g., 'Design an algorithm')",
    "logic_problem": "State the logic problem (e.g., 'Validate a syllogism')",
    "emotion_query": "Describe the emotional query (e.g., 'Customer complaint')",
    "dataset_description": "Describe the dataset (e.g., 'Financial dataset with missing values')",
    "process_description": "Describe the process (e.g., 'Data pipeline failure')",
    "source_language": "Specify source language (e.g., 'Python')",
    "target_language": "Specify target language (e.g., 'Java')",
    "code": "Paste the code (e.g., 'print(\"Hello\")')"
}

for var in info.get("input_variables", []):
    if var == "domain":
        variables[var] = domain
    elif var in ["text", "source_columns_block", "examples", "document", "response", "rfp_description", "hr_task", "learning_topic", "financial_task", "project_task", "code", "dataset_description", "process_description"]:
        variables[var] = st.text_area(
            f"Enter {var}:",
            value=variables.get(var, example_inputs.get(var, "")),
            placeholder=input_guidance.get(var, f"Provide value for {var}"),
            key=var,
            help=input_guidance.get(var, f"Provide value for {var}")
        )
    else:
        variables[var] = st.text_input(
            f"Enter {var}:",
            value=variables.get(var, example_inputs.get(var, "")),
            placeholder=input_guidance.get(var, f"Provide value for {var}"),
            key=var,
            help=input_guidance.get(var, f"Provide value for {var}")
        )

# Generate Prompt
if st.button("🚀 Generate Prompt"):
    try:
        # Validate Required Inputs
        missing_inputs = [var for var in info.get("input_variables", []) if not variables.get(var)]
        if missing_inputs:
            st.error(f"Missing required inputs: {', '.join(missing_inputs)}")
            logger.error(f"Missing required inputs: {missing_inputs}")
        else:
            # Fetch Business Rules
            rules = [r for r in business_rules if r["Field"] in variables]

            # Validate Inputs Against Business Rules
            validation_errors = pg.validate_with_rules(variables, selected_type)

            if validation_errors:
                st.error("Validation Errors:")
                for error in validation_errors:
                    st.error(f"- {error}")
                    logger.error(f"Validation error: {error}")
            else:
                # Warn for common misuse cases
                goal_lower = user_goal.lower()
                if selected_type == "Instruction-Based" and "summar" in goal_lower:
                    st.warning("For summarization, consider using 'Summary', which accepts 'text' and 'word_count' inputs.")
                    logger.warning("Instruction-Based used for summarization; suggested Summary prompt type.")
                elif selected_type == "Instruction-Based" and "map" in goal_lower:
                    st.warning("For data mapping, consider using 'Data-Mapping', which accepts table and column inputs.")
                    logger.warning("Instruction-Based used for data mapping; suggested Data-Mapping prompt type.")
                elif selected_type == "Zero-Shot" and "reason" in goal_lower:
                    st.warning("For complex reasoning, consider using 'Chain-of-Thought' for step-by-step analysis.")
                    logger.warning("Zero-Shot used for reasoning; suggested Chain-of-Thought prompt type.")

                # Generate Final Prompt
                final_prompt = pg.generate_prompt(selected_type, variables)
                st.session_state.final_prompt = final_prompt
                st.text_area("📝 Generated Prompt", final_prompt, height=200)
                logger.info(f"Generated prompt: {final_prompt[:200]}...")
    except Exception as e:
        st.error(f"Failed to generate prompt: {e}")
        logger.error(f"Failed to generate prompt: {e}")

# Get LLM Response (Separate Button)
if st.button("🧠 Get LLM Response"):
    if "final_prompt" not in st.session_state or not st.session_state.final_prompt:
        st.error("Please generate a prompt first by clicking 'Generate Prompt'.")
        logger.error("LLM response requested but no prompt generated")
    else:
        with st.spinner("🧠 Thinking..."):
            final_prompt = st.session_state.final_prompt
            structured_types = [
                "Data-Mapping", "Chain-of-Thought", "Self-Consistency", "Tree-of-Thoughts",
                "Graph-of-Thought", "Skeleton-of-Thought", "Chain-of-Verification", "ReAct",
                "Recursive Prompting", "Automatic Prompt Engineer (APE)", "Automatic Reasoning and Tool-use (ART)",
                "Chain-of-Note", "Chain-of-Code", "Chain-of-Symbol", "Structured Chain-of-Thought",
                "Contrastive Chain-of-Thought", "Logical Chain-of-Thought", "System 2 Attention Prompting",
                "Research-Challenges", "Data Quality", "Self-Healing", "Code Conversion",
                "Presales", "HR Operations", "Learning and Knowledge", "Finance", "Project Management"
            ]
            parse_json = selected_type in structured_types
            try:
                logger.info(f"Calling LLM with prompt: {final_prompt[:200]}...")
                response = asyncio.run(run_llm_prompt([final_prompt], parse_json=parse_json))[0]
                if parse_json:
                    st.success("LLM Response (Parsed JSON):")
                    st.json(response)
                else:
                    st.success("LLM Response:")
                    st.write(response)
                logger.info(f"LLM response received: {response[:200]}...")

                # Download Generated Prompt and Response
                st.download_button(
                    label="💾 Download Prompt + Response",
                    data=json.dumps({
                        "prompt_type": selected_type,
                        "inputs": variables,
                        "generated_prompt": final_prompt,
                        "llm_response": response
                    }, indent=2),
                    file_name="prompt_response.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Failed to get LLM response: {e}")
                logger.error(f"Failed to get LLM response: {e}")