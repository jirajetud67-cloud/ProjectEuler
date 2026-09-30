import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="Euler's Method",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Project 1: Euler's Method")
st.write("Approximate the solution to an initial-value problem using Euler's Method.")

# =========================================================
# Sidebar: Input parameters
# =========================================================
st.sidebar.header("⚙️ Parameters")

# ตัวอย่าง IVP:
# y' = f(x,y)
# y(x0) = y0

x0 = st.sidebar.number_input(
    "Initial x (x₀)",
    value=1.0,
    step=0.1
)

y0 = st.sidebar.number_input(
    "Initial y (y₀)",
    value=2.1272295,
    step=0.1
)

x_end = st.sidebar.number_input(
    "Final x",
    value=2.0,
    step=0.1
)

h = st.sidebar.number_input(
    "Step size (h)",
    min_value=0.001,
    value=0.1,
    step=0.01,
    format="%.3f"
)

st.sidebar.divider()
st.sidebar.subheader("Differential Equation")

st.sidebar.code("y' = f(x, y)")

# =========================================================
# Define differential equation
# =========================================================
def f(x, y):
    # แก้สมการตรงนี้ตามโจทย์ของ Project 1
    return x + y


# =========================================================
# Exact solution
# =========================================================
def exact_solution(x):
    # แก้เป็น Exact Solution ของโจทย์จริง
    #
    # ตัวอย่างสำหรับสมการ y' = x + y
    #
    # y = C e^x - x - 1
    #
    # จาก y(1) = 2.1272295
    # หา C

    C = (y0 + x0 + 1) / np.exp(x0)

    return C * np.exp(x) - x - 1


# =========================================================
# Euler's Method
# =========================================================
def euler_method(f, x0, y0, x_end, h):

    n = int(round((x_end - x0) / h))

    x = np.zeros(n + 1)
    y = np.zeros(n + 1)

    x[0] = x0
    y[0] = y0

    for i in range(n):
        x[i + 1] = x[i] + h
        y[i + 1] = y[i] + h * f(x[i], y[i])

    return x, y


# =========================================================
# Calculate
# =========================================================
x, y_euler = euler_method(
    f,
    x0,
    y0,
    x_end,
    h
)

y_exact = exact_solution(x)

error = np.abs(y_exact - y_euler)


# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "📚 Theory",
    "🧮 Simulator",
    "📊 Results"
])


# =========================================================
# TAB 1 : Theory
# =========================================================
with tab1:

    st.header("Euler's Method")

    st.write("""
    Euler's Method เป็นวิธีเชิงตัวเลขสำหรับประมาณคำตอบของ
    Initial-Value Problem (IVP)
    """)

    st.latex(r"""
    \frac{dy}{dx}=f(x,y), \qquad y(x_0)=y_0
    """)

    st.write("สูตรของ Euler's Method คือ")

    st.latex(r"""
    y_{n+1}=y_n+h f(x_n,y_n)
    """)

    st.write("""
    โดยที่

    - $x_n$ คือค่าของตัวแปรอิสระ ณ จุดที่ n
    - $y_n$ คือค่าประมาณของคำตอบ ณ จุดที่ n
    - $h$ คือ step size
    - $f(x,y)$ คือฟังก์ชันจาก differential equation
    """)

    st.info(
        "ลดค่า h จะทำให้จุดประมาณมีความละเอียดมากขึ้น "
        "และโดยทั่วไปช่วยลด numerical error"
    )


# =========================================================
# TAB 2 : Simulator
# =========================================================
with tab2:

    st.header("🧮 Euler's Method Simulator")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Initial Value Problem")

        st.latex(
            r"\frac{dy}{dx}=f(x,y)"
        )

        st.write(
            f"Initial condition: "
            f"$y({x0}) = {y0}$"
        )

    with col2:
        st.subheader("Parameters")

        st.write(f"Initial x : **{x0}**")
        st.write(f"Initial y : **{y0}**")
        st.write(f"Final x : **{x_end}**")
        st.write(f"Step size : **{h}**")

    st.divider()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        x,
        y_exact,
        color="blue",
        linewidth=2,
        label="Exact Solution"
    )

    ax.plot(
        x,
        y_euler,
        "o--",
        color="red",
        markersize=4,
        label="Euler's Method"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Exact Solution vs Euler's Numerical Solution")

    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig)


# =========================================================
# TAB 3 : Results
# =========================================================
with tab3:

    st.header("📊 Results")

    st.subheader("Comparison Table")

    # Create DataFrame
    result = pd.DataFrame({
        "x": x,
        "Euler's": y_euler,
        "Exact": y_exact,
        "Error": error
    })

    # Format ตัวเลข
    result_display = result.copy()

    result_display["x"] = result_display["x"].map(
        lambda v: f"{v:.1f}"
    )

    result_display["Euler's"] = result_display["Euler's"].map(
        lambda v: f"{v:.7f}"
    )

    result_display["Exact"] = result_display["Exact"].map(
        lambda v: f"{v:.7f}"
    )

    result_display["Error"] = result_display["Error"].map(
        lambda v: f"{v:.7f}"
    )

    st.dataframe(
        result_display,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Numerical Error")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Maximum Error",
            f"{np.max(error):.7f}"
        )

    with col2:
        st.metric(
            "Mean Error",
            f"{np.mean(error):.7f}"
        )

    with col3:
        st.metric(
            "Final Error",
            f"{error[-1]:.7f}"
        )

    st.divider()

    st.subheader("Output Format")

    # แสดงรูปแบบตามที่โจทย์กำหนด
    st.code(
        "Euler's    Exact       Error\n"
        "--------------------------------\n" +
        "\n".join(
            f"{e:.7f}  {ex:.7f}  {er:.7f}"
            for e, ex, er in zip(
                y_euler,
                y_exact,
                error
            )
        ),
        language="text"
    )
