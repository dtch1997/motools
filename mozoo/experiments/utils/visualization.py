"""Visualization utilities for mozoo experiments."""

from textwrap import dedent


def create_tabbed_html(plot_htmls: list[str], tab_titles: list[str], experiment_title: str) -> str:
    """Create HTML with tabs containing multiple plots.

    Args:
        plot_htmls: List of HTML strings for each plot (containing script and div)
        tab_titles: List of titles for each tab
        experiment_title: Title for the experiment (used in page title and h1)

    Returns:
        Complete HTML string with tabs

    Example:
        >>> plot_htmls = [fig1.to_html(include_plotlyjs=False), fig2.to_html(include_plotlyjs=False)]
        >>> html = create_tabbed_html(plot_htmls, ["Tab 1", "Tab 2"], "My Experiment")
    """
    # Get plotly.js CDN link
    plotly_js = (
        "<script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n"
        '    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    )

    # Build tab buttons
    tab_buttons = "\n".join(
        dedent(f"""
        <button class="tab-button {"active" if i == 0 else ""}" onclick="showTab({i})">
            {title}
        </button>""").strip()
        for i, title in enumerate(tab_titles)
    )

    # Build tab contents
    tab_contents = "\n".join(
        dedent(f"""
        <div id="tab-content-{i}" class="tab-content {"show active" if i == 0 else ""}">
            <div class="plot-wrapper">
                {plot_html}
            </div>
        </div>""").strip()
        for i, plot_html in enumerate(plot_htmls)
    )

    # HTML template
    html_template = dedent("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{experiment_title} Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin-top: 0;
            color: #333;
        }}
        .tabs {{
            display: flex;
            border-bottom: 2px solid #ddd;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .tab-button {{
            background: none;
            border: none;
            padding: 12px 24px;
            cursor: pointer;
            font-size: 14px;
            color: #666;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: all 0.2s;
        }}
        .tab-button:hover {{
            color: #333;
            background-color: #f9f9f9;
        }}
        .tab-button.active {{
            color: #007bff;
            border-bottom-color: #007bff;
            font-weight: 600;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.show {{
            display: block;
        }}
        .plot-wrapper {{
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
        }}
        .plotly-graph-div {{
            width: 100% !important;
            max-width: 100%;
        }}
    </style>
    {plotly_js}
</head>
<body>
    <div class="container">
        <h1>{experiment_title} Results</h1>
        <div class="tabs">
            {tab_buttons}
        </div>
        <div class="tab-contents">
            {tab_contents}
        </div>
    </div>
    <script>
        function showTab(index) {{
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => {{
                content.classList.remove('show', 'active');
            }});

            // Remove active class from all buttons
            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(button => {{
                button.classList.remove('active');
            }});

            // Show selected tab content
            const selectedContent = document.getElementById('tab-content-' + index);
            if (selectedContent) {{
                selectedContent.classList.add('show', 'active');
            }}

            // Add active class to selected button
            const selectedButton = buttons[index];
            if (selectedButton) {{
                selectedButton.classList.add('active');
            }}
        }}

        // Initialize: show first tab
        showTab(0);
    </script>
</body>
</html>
    """).strip()

    return html_template.format(
        experiment_title=experiment_title,
        plotly_js=plotly_js,
        tab_buttons=tab_buttons,
        tab_contents=tab_contents,
    )
