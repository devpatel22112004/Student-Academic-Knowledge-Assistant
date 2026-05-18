import html
import streamlit as st


class UI:
    @staticmethod
    def html(content: str) -> None:
        st.markdown(content, unsafe_allow_html=True)

    @staticmethod
    def css(content: str) -> None:
        st.markdown(f"<style>{content}</style>", unsafe_allow_html=True)

    @staticmethod
    def escape(value) -> str:
        # Escape user text before rendering it in HTML.
        return html.escape(str(value), quote=True)

    @staticmethod
    def nl2br(value) -> str:
        # Keep line breaks visible in rendered answers.
        return UI.escape(value).replace("\n", "<br>")