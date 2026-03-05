import io
import json
import csv
import os
import re

import anthropic
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request, send_file
from urllib.parse import urljoin, urlparse

app = Flask(__name__)
app.secret_key = os.urandom(24)

# These are stripped from lines (the criterion text may follow on the same line)
STRIP_PREFIXES = [
    "this criterion is linked to a learning outcome",
    "this area will be used by the assessor to leave comments related to this criterion.",
]

# Lines that are entirely noise after stripping
SKIP_WORDS = {"criteria", "ratings", "pts"}


def fetch_site_content(start_url: str, crawl: bool = False) -> dict:
    """Fetch HTML and CSS from a URL. If crawl=True, follows same-domain links."""
    visited: set[str] = set()
    pages: dict[str, dict] = {}
    base_domain = urlparse(start_url).netloc

    def fetch_page(url: str):
        if url in visited:
            return
        visited.add(url)

        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except Exception as e:
            pages[url] = {"error": str(e), "html": "", "css": ""}
            return

        soup = BeautifulSoup(resp.text, "html.parser")
        css_parts: list[str] = []

        for style_tag in soup.find_all("style"):
            css_parts.append(f"/* inline <style> */\n{style_tag.get_text()}")

        for link in soup.find_all("link", rel="stylesheet"):
            href = link.get("href")
            if href:
                css_url = urljoin(url, href)
                try:
                    css_resp = requests.get(css_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                    css_parts.append(f"/* {css_url} */\n{css_resp.text}")
                except Exception:
                    pass

        pages[url] = {"html": resp.text, "css": "\n\n".join(css_parts)}

        if crawl:
            for a in soup.find_all("a", href=True):
                linked = urljoin(url, a["href"])
                parsed = urlparse(linked)
                # Same domain, http/https only, no fragments
                if (
                    parsed.netloc == base_domain
                    and parsed.scheme in ("http", "https")
                    and linked.split("#")[0] not in visited
                ):
                    fetch_page(linked.split("#")[0])

    fetch_page(start_url)
    return pages


def parse_rubric(rubric_text: str) -> list[dict]:
    """Parse Canvas-style rubric text into [{name, max_points}]."""
    criteria = []
    current_lines: list[str] = []

    for line in rubric_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Strip boilerplate prefixes — Canvas glues them to the criterion text on the same line
        # e.g. "This criterion is linked to a Learning OutcomeHTML page has a HEADER..."
        line_lower = line.lower()
        for prefix in STRIP_PREFIXES:
            idx = line_lower.find(prefix)
            if idx != -1:
                line = (line[:idx] + line[idx + len(prefix):]).strip()
                line_lower = line.lower()

        if not line:
            continue

        # Skip the "Criteria  Ratings  Pts" header row
        if "criteria" in line_lower and "ratings" in line_lower:
            continue

        # Skip standalone noise words
        if line_lower in SKIP_WORDS:
            continue

        # Detect point value line like "5 pts" or "10 pts"
        pts_match = re.fullmatch(r"(\d+)\s*pts?", line, re.IGNORECASE)
        if pts_match:
            points = int(pts_match.group(1))
            if current_lines:
                name = " ".join(current_lines).strip()
                criteria.append({"name": name, "max_points": points})
                current_lines = []
        else:
            current_lines.append(line)

    return criteria


VNU_API = "https://validator.w3.org/nu/?out=json"
VNU_HEADERS = {"User-Agent": "CS140-Grader/1.0"}


def validate_with_vnu(pages: dict) -> dict:
    """Run W3C Nu Validator on the first page's HTML and all collected CSS."""
    first_page = next(iter(pages.values()), {})
    html = first_page.get("html", "")
    css = first_page.get("css", "")

    def _call(content: str, content_type: str) -> list[dict]:
        if not content.strip():
            return []
        try:
            resp = requests.post(
                VNU_API,
                headers={**VNU_HEADERS, "Content-Type": content_type},
                data=content.encode("utf-8"),
                timeout=20,
            )
            msgs = resp.json().get("messages", [])
            return [
                {
                    "type": m.get("type", "unknown"),      # "error" | "info" (warnings)
                    "subtype": m.get("subType", ""),
                    "message": m.get("message", ""),
                    "line": m.get("lastLine"),
                }
                for m in msgs
            ]
        except Exception as e:
            return [{"type": "error", "message": f"Validator unreachable: {e}", "line": None}]

    html_msgs = _call(html, "text/html; charset=utf-8")
    css_msgs  = _call(css,  "text/css; charset=utf-8")

    html_errors   = [m for m in html_msgs if m["type"] == "error"]
    html_warnings = [m for m in html_msgs if m["type"] != "error"]
    css_errors    = [m for m in css_msgs  if m["type"] == "error"]
    css_warnings  = [m for m in css_msgs  if m["type"] != "error"]

    return {
        "html_errors": html_errors,
        "html_warnings": html_warnings,
        "css_errors": css_errors,
        "css_warnings": css_warnings,
    }


def _format_vnu_for_prompt(validation: dict) -> str:
    lines = []
    he = validation["html_errors"]
    hw = validation["html_warnings"]
    ce = validation["css_errors"]
    cw = validation["css_warnings"]

    lines.append(f"HTML: {len(he)} error(s), {len(hw)} warning(s)")
    for m in he[:10]:
        loc = f" (line {m['line']})" if m.get("line") else ""
        lines.append(f"  [ERROR]{loc} {m['message']}")
    for m in hw[:5]:
        loc = f" (line {m['line']})" if m.get("line") else ""
        lines.append(f"  [WARN]{loc} {m['message']}")

    lines.append(f"CSS: {len(ce)} error(s), {len(cw)} warning(s)")
    for m in ce[:10]:
        loc = f" (line {m['line']})" if m.get("line") else ""
        lines.append(f"  [ERROR]{loc} {m['message']}")
    for m in cw[:5]:
        loc = f" (line {m['line']})" if m.get("line") else ""
        lines.append(f"  [WARN]{loc} {m['message']}")

    return "\n".join(lines)


def build_content_string(pages: dict) -> str:
    parts = []
    for url, content in pages.items():
        if content.get("error"):
            parts.append(f"=== PAGE: {url} (fetch error: {content['error']}) ===")
        else:
            parts.append(
                f"=== PAGE: {url} ===\n\n"
                f"--- HTML ---\n{content['html']}\n\n"
                f"--- CSS ---\n{content['css']}"
            )
    combined = "\n\n".join(parts)
    # Keep within ~60k chars to stay well inside token limits
    if len(combined) > 60_000:
        combined = combined[:60_000] + "\n\n[... content truncated ...]"
    return combined


def grade_with_claude(
    assignment_description: str,
    criteria: list[dict],
    pages: dict,
    api_key: str,
    validation: dict | None = None,
) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    content_str = build_content_string(pages)

    validation_section = ""
    if validation:
        validation_section = f"""
## W3C Nu Validator Results (authoritative — use ONLY these for any validity criterion)
{_format_vnu_for_prompt(validation)}

For any rubric criterion that mentions HTML or CSS validity/validation, base your score and feedback
ENTIRELY on the above validator output. Do NOT form your own opinion on validity.
Warnings do not count against the score. Only errors do.
A reasonable deduction guide: 0 errors = full points, 1-2 errors = lose ~25%, 3-5 = lose ~50%, 6+ = lose ~75-100%.
"""

    prompt = f"""You are a university CS instructor grading an introductory HTML/CSS web development assignment.

## Assignment Description
{assignment_description}

## Rubric Criteria
{json.dumps(criteria, indent=2)}
{validation_section}
## Student Website Source Code
{content_str}

---

Grade each criterion carefully by reading the actual HTML and CSS. For each criterion give:
- earned_points: integer from 0 to max_points
- feedback: 1-3 sentences referencing specific elements or code (e.g. "The <header> tag is present and contains an <h1>. However...")

Do NOT comment on the quality of placeholder/lorem ipsum text content — only evaluate structure and CSS.

Respond ONLY with valid JSON — no markdown fences, no extra text — in this exact shape:
{{
  "criteria": [
    {{
      "name": "<copy name exactly from rubric>",
      "max_points": <number>,
      "earned_points": <number>,
      "feedback": "<specific feedback>"
    }}
  ],
  "overall_feedback": "<1-2 sentence overall comment>"
}}"""

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    # Strip accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/grade", methods=["POST"])
def grade():
    data = request.get_json(force=True)
    api_key = (data.get("api_key") or "").strip()
    student_url = (data.get("url") or "").strip()
    description = (data.get("assignment_description") or "").strip()
    rubric_text = (data.get("rubric") or "").strip()
    crawl = bool(data.get("crawl", False))

    if not all([api_key, student_url, description, rubric_text]):
        return jsonify({"error": "All fields are required."}), 400

    criteria = parse_rubric(rubric_text)
    if not criteria:
        return jsonify({"error": "Could not parse any criteria from the rubric. Check the format."}), 400

    try:
        pages = fetch_site_content(student_url, crawl=crawl)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch website: {e}"}), 400

    if not pages:
        return jsonify({"error": "No pages could be fetched from that URL."}), 400

    # Run W3C validation (best-effort — grading continues even if it fails)
    validation = validate_with_vnu(pages)

    try:
        result = grade_with_claude(description, criteria, pages, api_key, validation)
    except anthropic.AuthenticationError:
        return jsonify({"error": "Invalid Anthropic API key."}), 401
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Claude returned malformed JSON: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "result": result,
        "validation": validation,
        "pages_fetched": list(pages.keys()),
        "criteria_parsed": criteria,
    })


@app.route("/api/export-csv", methods=["POST"])
def export_csv():
    data = request.get_json(force=True)
    grades: list[dict] = data.get("grades", [])

    if not grades:
        return jsonify({"error": "No grades to export."}), 400

    output = io.StringIO()
    writer = csv.writer(output)

    # Dynamic headers from first result
    first_criteria = grades[0]["result"]["criteria"]
    headers = (
        ["Student URL"]
        + [f"{c['name']} (/{c['max_points']})" for c in first_criteria]
        + ["Total Earned", "Total Possible", "Percentage", "Overall Feedback"]
    )
    writer.writerow(headers)

    for entry in grades:
        url = entry["student_url"]
        criteria = entry["result"]["criteria"]
        earned = sum(c["earned_points"] for c in criteria)
        possible = sum(c["max_points"] for c in criteria)
        pct = f"{round(earned / possible * 100)}%" if possible else "N/A"
        row = (
            [url]
            + [f"{c['earned_points']} — {c['feedback']}" for c in criteria]
            + [earned, possible, pct, entry["result"].get("overall_feedback", "")]
        )
        writer.writerow(row)

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="grades.csv",
    )


CANVAS_INSTANCE = "byuh.instructure.com"


def _canvas_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _canvas_get(url: str, token: str, params: dict = None) -> requests.Response:
    return requests.get(url, headers=_canvas_headers(token), params=params, timeout=15)


def _fetch_all_submissions(base: str, course_id: str, assignment_id: str, token: str) -> list:
    """Fetch all submissions with rubric assessments, handling pagination."""
    url = f"{base}/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions"
    params = {"include[]": "rubric_assessment", "per_page": 100}
    results = []
    while url:
        resp = _canvas_get(url, token, params)
        resp.raise_for_status()
        results.extend(resp.json())
        # Canvas uses Link header for pagination
        url = None
        link = resp.headers.get("Link", "")
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
                params = None  # already encoded in next URL
                break
    return results


@app.route("/api/canvas/fetch-rubric", methods=["POST"])
def canvas_fetch_rubric():
    data = request.get_json(force=True)
    token         = (data.get("token") or "").strip()
    course_id     = (data.get("course_id") or "").strip()
    assignment_id = (data.get("assignment_id") or "").strip()

    if not all([token, course_id, assignment_id]):
        return jsonify({"error": "Token, Course ID, and Assignment ID are all required."}), 400

    base = f"https://{CANVAS_INSTANCE}"

    # Fetch assignment (rubric) and all submissions in parallel
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        assignment_future   = pool.submit(_canvas_get, f"{base}/api/v1/courses/{course_id}/assignments/{assignment_id}", token)
        submissions_future  = pool.submit(_fetch_all_submissions, base, course_id, assignment_id, token)

        try:
            assignment_resp = assignment_future.result()
        except requests.RequestException as e:
            return jsonify({"error": str(e)}), 400

        if assignment_resp.status_code == 401:
            return jsonify({"error": "Invalid Canvas API token."}), 401
        if assignment_resp.status_code == 403:
            return jsonify({"error": "Your account doesn't have permission to access this assignment."}), 403
        if assignment_resp.status_code == 404:
            return jsonify({"error": "Assignment not found — double-check the Course ID and Assignment ID."}), 404
        assignment_resp.raise_for_status()

        try:
            raw_submissions = submissions_future.result()
        except Exception:
            raw_submissions = []  # non-fatal — carry on without submission states

    assignment = assignment_resp.json()
    rubric = assignment.get("rubric", [])
    if not rubric:
        return jsonify({"error": "This assignment has no rubric attached in Canvas."}), 400

    criteria = [
        {"id": c["id"], "description": c.get("description", ""), "points": c.get("points", 0)}
        for c in rubric
    ]

    # Build submission state map keyed by user_id (as string)
    submissions = {}
    for s in raw_submissions:
        uid = str(s.get("user_id", ""))
        if not uid:
            continue
        ra = s.get("rubric_assessment") or {}
        submissions[uid] = {
            "workflow_state": s.get("workflow_state", ""),
            "score": s.get("score"),
            "graded": s.get("workflow_state") == "graded",
            "rubric_assessed": bool(ra),
            "rubric_assessment": ra,   # full {criterion_id: {points, comments}}
        }

    return jsonify({
        "criteria": criteria,
        "assignment_name": assignment.get("name", ""),
        "submissions": submissions,
    })


@app.route("/api/canvas/push-grades", methods=["POST"])
def canvas_push_grades():
    data = request.get_json(force=True)
    token         = (data.get("token") or "").strip()
    course_id     = (data.get("course_id") or "").strip()
    assignment_id = (data.get("assignment_id") or "").strip()
    grades        = data.get("grades", [])
    criterion_map = data.get("criterion_map", {})  # {our_criterion_name: canvas_criterion_id}

    if not all([token, course_id, assignment_id]):
        return jsonify({"error": "Canvas configuration incomplete."}), 400

    base = f"https://{CANVAS_INSTANCE}/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions"
    auth = {"Authorization": f"Bearer {token}"}
    results = []

    for entry in grades:
        student = entry.get("student") or {}
        user_id = student.get("id", "").strip()
        name    = student.get("name", entry.get("student_url", "unknown"))

        if not user_id:
            results.append({"student": name, "status": "skipped", "reason": "No Canvas user ID — was this student matched from the roster?"})
            continue

        # Build form-encoded rubric assessment params
        params = {}
        matched = 0
        for c in entry["result"]["criteria"]:
            canvas_id = criterion_map.get(c["name"])
            if canvas_id:
                params[f"rubric_assessment[{canvas_id}][points]"]   = c["earned_points"]
                params[f"rubric_assessment[{canvas_id}][comments]"] = c.get("feedback", "")
                matched += 1

        if matched == 0:
            results.append({"student": name, "status": "skipped", "reason": "No rubric criteria could be matched to Canvas."})
            continue

        try:
            resp = requests.put(f"{base}/{user_id}", headers=auth, data=params, timeout=15)
            if resp.status_code == 401:
                results.append({"student": name, "status": "error", "reason": "Token rejected."})
            elif resp.status_code == 403:
                results.append({"student": name, "status": "error", "reason": "Permission denied."})
            elif not resp.ok:
                results.append({"student": name, "status": "error", "reason": f"HTTP {resp.status_code}"})
            else:
                results.append({"student": name, "status": "ok"})
        except Exception as e:
            results.append({"student": name, "status": "error", "reason": str(e)})

    return jsonify({"results": results})


@app.route("/api/parse-canvas-csv", methods=["POST"])
def parse_canvas_csv_route():
    data = request.get_json(force=True)
    csv_text = data.get("csv", "")

    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)

    if len(rows) < 3:
        return jsonify({"error": "CSV appears empty or malformed."}), 400

    headers = rows[0]
    points_row = rows[1]

    # Find first editable assignment column (numeric points, not "(read only)", after col 3)
    assignment_col = None
    assignment_name = None
    assignment_points = None

    for i in range(4, len(headers)):
        pts = points_row[i].strip() if i < len(points_row) else ""
        if pts and pts != "(read only)":
            try:
                assignment_points = float(pts)
                assignment_col = i
                assignment_name = headers[i]
                break
            except ValueError:
                pass

    if assignment_col is None:
        return jsonify({"error": "Could not find an editable assignment column. Make sure you exported 'from current view'."}), 400

    students = []
    for row in rows[2:]:
        if not row or not row[0].strip():
            continue
        students.append({
            "name": row[0].strip().strip('"'),
            "id": row[1].strip() if len(row) > 1 else "",
            "sis_login_id": row[2].strip() if len(row) > 2 else "",
            "section": row[3].strip() if len(row) > 3 else "",
        })

    return jsonify({
        "students": students,
        "assignment_column": assignment_name,
        "assignment_points": assignment_points,
    })


@app.route("/api/export-canvas-csv", methods=["POST"])
def export_canvas_csv():
    data = request.get_json(force=True)
    grades = data.get("grades", [])
    assignment_column = data.get("assignment_column", "Assignment")
    assignment_points = data.get("assignment_points", 100)

    if not grades:
        return jsonify({"error": "No grades to export."}), 400

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Student", "ID", "SIS Login ID", "Section", assignment_column])
    writer.writerow(["    Points Possible", "", "", "", assignment_points])

    for entry in grades:
        student = entry.get("student", {})
        criteria = entry["result"]["criteria"]
        earned = sum(c["earned_points"] for c in criteria)
        writer.writerow([
            student.get("name", ""),
            student.get("id", ""),
            student.get("sis_login_id", ""),
            student.get("section", ""),
            earned,
        ])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="canvas_grades.csv",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
