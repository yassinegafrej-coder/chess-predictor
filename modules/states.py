"""Manages session states variables.
"""


import streamlit as st


def init_states():
    if 'next' not in st.session_state:
        st.session_state.next = 0

    if 'curfen' not in st.session_state:
        st.session_state.curfen = None

    if 'oldfen' not in st.session_state:
        st.session_state.oldfen = None
