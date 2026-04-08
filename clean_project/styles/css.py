"""
CSS styling and theme management.
Centralized CSS styles for the entire application.
"""

# Define CSS variables
CSS_LIGHT = {
    "primary_color": "#667EEA",
    "secondary_color": "#64748B",
    "success_color": "#10B981",
    "warning_color": "#F59E0B",
    "danger_color": "#EF4444",
    "info_color": "#3B82F6",
    "bg_color": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "text_color": "#1E293B",
    "text_secondary": "#64748B",
    "border_color": "#E2E8F0",
}

CSS_DARK = {
    "primary_color": "#818CF8",
    "secondary_color": "#94A3B8",
    "success_color": "#34D399",
    "warning_color": "#FBBF24",
    "danger_color": "#F87171",
    "info_color": "#60A5FA",
    "bg_color": "#0F172A",
    "card_bg": "#1E293B",
    "text_color": "#F1F5F9",
    "text_secondary": "#CBD5E1",
    "border_color": "#334155",
}


def get_base_css(theme: str = "light") -> str:
    """Get base CSS styles.
    
    Args:
        theme: Theme name (light/dark)
    
    Returns:
        CSS string
    """
    
    colors = CSS_LIGHT if theme == "light" else CSS_DARK
    
    return f"""
    <style>
        :root {{
            --primary: {colors['primary_color']};
            --secondary: {colors['secondary_color']};
            --success: {colors['success_color']};
            --warning: {colors['warning_color']};
            --danger: {colors['danger_color']};
            --info: {colors['info_color']};
            --bg: {colors['bg_color']};
            --card-bg: {colors['card_bg']};
            --text: {colors['text_color']};
            --text-secondary: {colors['text_secondary']};
            --border: {colors['border_color']};
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto';
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}
        
        .main {{
            background-color: var(--bg);
        }}
        
        .css-18e3th9 {{
            padding-top: 0;
        }}
    </style>
    """


def get_header_css(theme: str = "light") -> str:
    """Get header component CSS.
    
    Args:
        theme: Theme name
    
    Returns:
        CSS string
    """
    
    colors = CSS_LIGHT if theme == "light" else CSS_DARK
    
    return f"""
    <style>
        .app-header {{
            background: linear-gradient(135deg, {colors['primary_color']}, {colors['secondary_color']});
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .app-header h1 {{
            margin: 0;
            font-size: 24px;
            font-weight: 700;
        }}
        
        .app-header-subtitle {{
            font-size: 12px;
            opacity: 0.9;
            margin-top: 5px;
        }}
    </style>
    """


def get_card_css(theme: str = "light") -> str:
    """Get card component CSS.
    
    Args:
        theme: Theme name
    
    Returns:
        CSS string
    """
    
    colors = CSS_LIGHT if theme == "light" else CSS_DARK
    
    return f"""
    <style>
        .card {{
            background-color: {colors['card_bg']};
            border: 1px solid {colors['border_color']};
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            transition: box-shadow 0.3s ease;
        }}
        
        .card:hover {{
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        }}
        
        .card-title {{
            font-size: 16px;
            font-weight: 600;
            color: {colors['text_color']};
            margin-bottom: 10px;
        }}
        
        .card-content {{
            font-size: 14px;
            color: {colors['text_secondary']};
        }}
    </style>
    """


def get_metric_css(theme: str = "light") -> str:
    """Get metric display CSS.
    
    Args:
        theme: Theme name
    
    Returns:
        CSS string
    """
    
    colors = CSS_LIGHT if theme == "light" else CSS_DARK
    
    return f"""
    <style>
        .metric {{
            background-color: {colors['card_bg']};
            border-left: 4px solid {colors['primary_color']};
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        
        .metric-label {{
            font-size: 12px;
            color: {colors['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}
        
        .metric-value {{
            font-size: 28px;
            font-weight: 700;
            color: {colors['primary_color']};
        }}
        
        .metric-delta {{
            font-size: 12px;
            margin-top: 5px;
        }}
    </style>
    """


def get_button_css(theme: str = "light") -> str:
    """Get button component CSS.
    
    Args:
        theme: Theme name
    
    Returns:
        CSS string
    """
    
    colors = CSS_LIGHT if theme == "light" else CSS_DARK
    
    return f"""
    <style>
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 14px;
        }}
        
        .btn-primary {{
            background-color: {colors['primary_color']};
            color: white;
        }}
        
        .btn-primary:hover {{
            opacity: 0.9;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        
        .btn-secondary {{
            background-color: {colors['secondary_color']};
            color: white;
        }}
        
        .btn-success {{
            background-color: {colors['success_color']};
            color: white;
        }}
        
        .btn-danger {{
            background-color: {colors['danger_color']};
            color: white;
        }}
    </style>
    """


def get_status_css(theme: str = "light") -> str:
    """Get status badge CSS.
    
    Args:
        theme: Theme name
    
    Returns:
        CSS string
    """
    
    colors = CSS_LIGHT if theme == "light" else CSS_DARK
    
    return f"""
    <style>
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .badge-success {{
            background-color: {colors['success_color']}22;
            color: {colors['success_color']};
        }}
        
        .badge-warning {{
            background-color: {colors['warning_color']}22;
            color: {colors['warning_color']};
        }}
        
        .badge-danger {{
            background-color: {colors['danger_color']}22;
            color: {colors['danger_color']};
        }}
        
        .badge-info {{
            background-color: {colors['info_color']}22;
            color: {colors['info_color']};
        }}
    </style>
    """


def get_form_css(theme: str = "light") -> str:
    """Get form component CSS.
    
    Args:
        theme: Theme name
    
    Returns:
        CSS string
    """
    
    colors = CSS_LIGHT if theme == "light" else CSS_DARK
    
    return f"""
    <style>
        .form-group {{
            margin-bottom: 20px;
        }}
        
        .form-label {{
            display: block;
            font-weight: 600;
            color: {colors['text_color']};
            margin-bottom: 8px;
            font-size: 14px;
        }}
        
        .form-input {{
            width: 100%;
            padding: 10px 12px;
            border: 1px solid {colors['border_color']};
            border-radius: 6px;
            font-size: 14px;
            background-color: {colors['card_bg']};
            color: {colors['text_color']};
            transition: border-color 0.3s ease;
        }}
        
        .form-input:focus {{
            outline: none;
            border-color: {colors['primary_color']};
            box-shadow: 0 0 0 3px {colors['primary_color']}22;
        }}
    </style>
    """


def get_all_css(theme: str = "light") -> str:
    """Get all CSS styles combined.
    
    Args:
        theme: Theme name
    
    Returns:
        Combined CSS string
    """
    
    css_parts = [
        get_base_css(theme),
        get_header_css(theme),
        get_card_css(theme),
        get_metric_css(theme),
        get_button_css(theme),
        get_status_css(theme),
        get_form_css(theme),
    ]
    
    return "\n".join(css_parts)


# Export color schemes
THEMES = {
    "light": CSS_LIGHT,
    "dark": CSS_DARK,
}
