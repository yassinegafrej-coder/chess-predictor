"""Supports other functionalities.
"""


import streamlit as st


EPDS = [
    'rnbq1rk1/pp3ppp/4pn2/8/2pQ4/P1P2NP1/4PPBP/R1B2RK1 b - - bm Nc6;',
    '4R3/1pr2pk1/p6p/n2Bpp2/8/P5P1/4PP1P/4K3 b - - bm e4;',
    '4R3/1p3pk1/p2n3p/2r2p2/P3p1P1/8/B3PP1P/4K3 w - - bm Rd8;',
    'r1b1q1k1/ppp2p2/3b1n1p/3p1np1/3P4/2P1BN1P/PPB2PP1/R2Q1NK1 b - - bm Nxe3;'
]


def set_page(title='Chess', page_icon="♟️"):
    st.set_page_config(
        page_title=title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help':
                'https://github.com/fsmosca/streamlit-chess-board/issues',
            'Report a bug':
                "https://github.com/fsmosca/streamlit-chess-board/issues",
            'About': "# A Streamlit chess board"
        }
    )
