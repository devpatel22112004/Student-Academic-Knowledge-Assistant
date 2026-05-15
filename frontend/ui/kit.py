import html

import streamlit as st


class UI:
    """Small helper wrapper for HTML/CSS rendering in Streamlit."""

    @staticmethod
    def html(content: str) -> None:
        st.markdown(content, unsafe_allow_html=True)

    @staticmethod
    def css(content: str) -> None:
        st.markdown(f"<style>{content}</style>", unsafe_allow_html=True)

    @staticmethod
    def escape(value) -> str:
        return html.escape(str(value), quote=True)

    @staticmethod
    def nl2br(value) -> str:
        return UI.escape(value).replace("\n", "<br>")