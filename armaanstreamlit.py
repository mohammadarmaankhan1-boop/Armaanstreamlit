import re
import streamlit as st
from openai import OpenAI

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Market Research Assistant",
    page_icon="📊",
    layout="wide",
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def validate_industry(user_input):
    """Q1: Validate industry input."""
    if not user_input or user_input.strip() == "":
        return False, None, "Industry name cannot be empty."

    cleaned = user_input.strip()

    if len(cleaned) < 2:
        return False, None, "Industry name too short."

    if len(cleaned) > 100:
        return False, None, "Industry name too long (max 100 characters)."

    suspicious = ["<script", "javascript:", "onclick=", "--"]
    if any(p.lower() in cleaned.lower() for p in suspicious):
        return False, None, "Invalid characters detected."

    return True, cleaned, None


def count_words_like_word(text: str) -> int:
    """
    More "Word-like" count:
    - Markdown links: [text](url) -> text
    - Raw URLs replaced with placeholder (Word often counts as 1)
    - Hyphenated / apostrophe compounds count as 1 token
    """
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "URL", text)
    tokens = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)
    return len(tokens)


def _response_text(response):
    return response.output_text or ""


def get_wikipedia_urls(client, industry):
    """Q2: Get 5 most relevant Wikipedia URLs."""

    system_prompt = (
        "You are a business research assistant. Find the 5 most relevant Wikipedia "
        "pages for the given industry. Return ONLY a numbered list of 5 Wikipedia "
        "URLs, nothing else."
    )

    user_prompt = (
        f"Find the 5 most relevant Wikipedia pages for the {industry} industry.\n\n"
        "Include: industry overview, major companies, technologies, trends, related "
        "sectors.\n\n"
        "Return exactly 5 URLs:\n"
        "1. https://en.wikipedia.org/wiki/...\n"
        "2. https://en.wikipedia.org/wiki/...\n"
        "3. https://en.wikipedia.org/wiki/...\n"
        "4. https://en.wikipedia.org/wiki/...\n"
        "5. https://en.wikipedia.org/wiki/...\n\n"
        "Search now."
    )

    response = client.responses.create(
        model="gpt-4.1",
        max_output_tokens=2000,
        tools=[{"type": "web_search"}],
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    response_text = _response_text(response)

    urls = []
    for line in response_text.strip().split("\n"):
        if "wikipedia.org" in line.lower():
            match = re.search(r"https?://[^\s\)]+wikipedia\.org[^\s\)\,]*", line)
            if match:
                urls.append(match.group(0).rstrip(".,;:"))

    # Fallback if model didn’t return enough links
    if len(urls) < 5:
        fallback = client.responses.create(
            model="gpt-4.1",
            max_output_tokens=1000,
            tools=[{"type": "web_search"}],
            input=[{"role": "user", "content": f"Wikipedia articles {industry}"}],
        )
        extra = re.findall(
            r"https?://[^\s\)]+wikipedia\.org[^\s\)\,]*", _response_text(fallback)
        )
        urls.extend([u.rstrip(".,;:") for u in extra])

    # Deduplicate and keep first 5
    unique = []
    seen = set()
    for url in urls:
        normalized = url.split("#")[0].rstrip("/")
        if normalized not in seen and "wikipedia.org" in normalized:
            unique.append(url)
            seen.add(normalized)
            if len(unique) == 5:
                break

    return unique[:5]


def generate_report(client, industry, urls):
    """Q3: Generate industry report (<500 words)."""

    system_prompt = (
        "You are a senior business analyst writing market research reports. "
        "Write professionally, analytically, and concisely. "
        "Structure: Overview → Key Players → Trends → Technologies → Outlook. "
        "Keep reports UNDER 500 words strictly."
    )

    urls_text = "\n".join([f"{i + 1}. {url}" for i, url in enumerate(urls)])

    user_prompt = (
        f"Generate a market research report on {industry} for business analysts.\n\n"
        "**Wikipedia Sources:**\n"
        f"{urls_text}\n\n"
        "**Requirements:**\n"
        "- LESS than 500 words (STRICT)\n"
        "- Sections: Overview, Key Players, Trends, Technologies, Outlook\n"
        "- Professional, analytical tone\n"
        "- Based on Wikipedia sources above\n"
        "- Include specific facts/figures\n\n"
        "Generate report now."
    )

    response = client.responses.create(
        model="gpt-4.1",
        max_output_tokens=4000,
        temperature=0.3,
        tools=[{"type": "web_search"}],
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return _response_text(response).strip()


# ============================================================================
# MAIN APP
# ============================================================================

st.title("📊 Market Research Assistant")
st.write("Generate professional industry reports based on Wikipedia data")

# Sidebar: API Key
api_key = st.sidebar.text_input("Your OpenAI API key", type="password")

if api_key:
    st.sidebar.success("✅ API key configured")
else:
    st.sidebar.warning("⚠️ Enter API key to continue")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**About:**
- Validates industry input (Q1)
- Finds 5 Wikipedia pages (Q2)
- Generates report <500 words (Q3)

**Course:** MSIN0231 ML4B
"""
)

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1
if "industry" not in st.session_state:
    st.session_state.industry = None
if "urls" not in st.session_state:
    st.session_state.urls = None
if "report" not in st.session_state:
    st.session_state.report = None

# ============================================================================
# STEP 1: INDUSTRY INPUT
# ============================================================================

st.subheader("Step 1: Enter Industry")

industry_input = st.text_input(
    "Which industry would you like to research?",
    placeholder="e.g., Renewable Energy, Artificial Intelligence, Automotive",
    help="Enter a specific industry name",
)

if st.button("🔍 Start Research", type="primary"):
    if not api_key:
        st.error("⚠️ Please enter your API key in the sidebar first")
        st.stop()

    is_valid, cleaned, error = validate_industry(industry_input)

    if not is_valid:
        st.error(f"❌ {error}")
        st.info("💡 Example: 'Renewable Energy', 'Cloud Computing', 'Biotechnology'")
    else:
        st.success(f"✅ Industry validated: **{cleaned}**")
        st.session_state.industry = cleaned
        st.session_state.urls = None
        st.session_state.report = None
        st.session_state.step = 2
        st.rerun()

# ============================================================================
# STEP 2: WIKIPEDIA URLS
# ============================================================================

if st.session_state.step >= 2 and st.session_state.industry:
    st.markdown("---")
    st.subheader("Step 2: Wikipedia Sources")

    if st.session_state.urls is None:
        with st.spinner(f"🔎 Finding Wikipedia pages for {st.session_state.industry}..."):
            try:
                client = OpenAI(api_key=api_key)
                urls = get_wikipedia_urls(client, st.session_state.industry)
                st.session_state.urls = urls
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()

    if st.session_state.urls:
        st.success(f"✅ Found {len(st.session_state.urls)} Wikipedia pages:")

        for i, url in enumerate(st.session_state.urls, 1):
            title = url.split("/wiki/")[-1].replace("_", " ") if "/wiki/" in url else url
            st.markdown(f"**{i}.** [{title}]({url})")

        st.markdown("")

        if st.button("📝 Generate Report", type="primary"):
            st.session_state.step = 3
            st.rerun()

# ============================================================================
# STEP 3: INDUSTRY REPORT
# ============================================================================

if st.session_state.step >= 3 and st.session_state.urls:
    st.markdown("---")
    st.subheader("Step 3: Industry Report")

    if st.session_state.report is None:
        with st.spinner(f"📊 Generating report for {st.session_state.industry}..."):
            try:
                client = OpenAI(api_key=api_key)
                report = generate_report(client, st.session_state.industry, st.session_state.urls)
                st.session_state.report = report
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()

    if st.session_state.report:
        st.markdown("### 📄 Market Research Report")
        st.markdown(f"**Industry:** {st.session_state.industry}")
        st.markdown("")
        st.markdown(st.session_state.report)
        st.markdown("")

        word_count = count_words_like_word(st.session_state.report)

        col1, col2, col3 = st.columns(3)

        with col1:
            if word_count < 500:
                st.success(f"✅ Words: {word_count}/500")
            else:
                st.warning(f"⚠️ Words: {word_count}/500")

        with col2:
            st.info(f"📚 Sources: {len(st.session_state.urls)}")

        with col3:
            st.download_button(
                label="⬇️ Download",
                data=st.session_state.report,
                file_name=f"{st.session_state.industry.replace(' ', '_')}_report.txt",
                mime="text/plain",
            )

# Reset button
if st.session_state.step > 1:
    st.markdown("---")
    if st.button("🔄 New Research"):
        st.session_state.step = 1
        st.session_state.industry = None
        st.session_state.urls = None
        st.session_state.report = None
        st.rerun()




