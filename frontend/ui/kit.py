import html
import streamlit as st

#HELPER CLASS FOR RENDERING HTML/CSS IN STREAMLIT. It provides methods to render HTML content, inject CSS styles, escape special characters for safe HTML rendering, and convert newlines to HTML line breaks. This class is used throughout the application to maintain a consistent approach to rendering custom HTML and styling in Streamlit components.
class UI:

    @staticmethod
    def html(content: str) -> None:
        st.markdown(content, unsafe_allow_html=True)

    @staticmethod
    def css(content: str) -> None:
       
     st.markdown(f"<style>{content}</style>", unsafe_allow_html=True)
 
 #ESCAP KA USE KARNE KA MAQSAD YE HAI KI USER SE AANE WALA DATA JISME SPECIAL CHARACTERS HO SAKTE HAIN, UNHE SAFE TARIKAY SE HTML ME RENDER KARNA. ISSE XSS (CROSS-SITE SCRIPTING) JAISI SECURITY VULNERABILITIES SE BACHAV HOTA HAI, KYUNKI SPECIAL CHARACTERS KO ESCAPE KARNE SE UNKA HTML ME ASLI ARTH NAHI NIKLEGA, BALKE WO AS A PLAIN TEXT RENDER HONGE.
    @staticmethod
    def escape(value) -> str:
        return html.escape(str(value), quote=True)

#NL2BR KA USE YE HAI KI USER SE AANE WALA TEXT JISME NEWLINE CHARACTERS HO SAKTE HAIN, UNHE HTML ME RENDER KARTE WAKT LINE BREAKS KE ROOP ME DIKHANA. HTML ME NEWLINE CHARACTERS KO AS A PLAIN TEXT RENDER KIYA JATA HAI, ISLIYE UNHE <br> TAGS ME CONVERT KARNE SE WO ASLI FORMAT ME DIKHENGE, JISSE USER KO TEXT KO PADNE ME ASANI HOGI.
    @staticmethod
    def nl2br(value) -> str:
        return UI.escape(value).replace("\n", "<br>")