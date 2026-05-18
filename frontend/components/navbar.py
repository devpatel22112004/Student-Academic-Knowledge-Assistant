import streamlit as st
from frontend.ui.kit import UI


def render_navbar(kicker, title, subtitle, hero_class):
    """Render a shared page header used by auth and workspace screens."""
    UI.html(
        f'''
        <div class="{hero_class}">
            <div class="workspace-kicker">{kicker}</div>
            <div class="workspace-title">{title}</div>
            <div class="workspace-subtitle">{subtitle}</div>
        </div>
        ''' if hero_class == "workspace-hero" else f'''
        <div class="{hero_class}">
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
        ''',
    )