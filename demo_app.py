import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

with st.echo(code_location='below'):
    st.title("Hello, World!")
    """
    This is a test.
    """
    x = np.linspace(0, 10, 500)
    fig = plt.figure()
    plt.plot(x, np.sin(x))
    plt.ylim(-2, 2)
    st.pyplot(fig)
