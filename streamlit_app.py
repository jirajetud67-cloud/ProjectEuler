import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==========================================
# 1. การตั้งค่าหน้าเว็บหลัก
# ==========================================
st.set_page_config(page_title="Project 1: Euler's Method", layout="wide")
st.title("🧮 Project 1: Euler's Method Simulator")

# ==========================================
# 2. st.sidebar สำหรับรับค่าพารามิเตอร์ (รับค่า h)
# ==========================================
st.sidebar.header("⚙️ กำหนดพารามิเตอร์")

# 1. ออกแบบวิธีรับค่า h
h = st.sidebar.number_input(
    "กรอกค่า Step size (h):", 
    value=0.1, 
    step=0.01, 
    format="%.4f"
)

# เงื่อนไขเริ่มต้นตามโจทย์
t0 = 1.0
t_end = 2.0
y0 = 1.0

# สมการ dy/dt = f(t, y)
def f(t, y):
    return (y / t) - ((y**2) / (t**2))

# สมการ Exact Solution y(t) = t / (1 + ln(t))
def exact_solution(t):
    return t / (1.0 + np.log(t))

# ==========================================
# 3. st.tabs แบ่งเนื้อหาออกเป็น 3 ส่วน
# ==========================================
tab1, tab2, tab3 = st.tabs(["📚 ทฤษฎี (Theory)", "🎮 ตัวจำลอง (Simulator)", "📊 สรุปผล (Summary)"])

# ------------------------------------------
# Tab 1: ทฤษฎี (Theory)
# ------------------------------------------
with tab1:
    st.header("โจทย์และระเบียบวิธีของออยเลอร์ (Euler's Method)")
    
    st.subheader("โจทย์ Initial-Value Problem (IVP)")
    st.latex(r"y' = \frac{y}{t} - \frac{y^2}{t^2}, \quad 1 \le t \le 2, \quad y(1) = 1")
    
    st.subheader("Exact Solution")
    st.latex(r"y(t) = \frac{t}{1 + \ln(t)}")
    
    st.subheader("สูตร Euler's Method")
    st.latex(r"w_{i+1} = w_i + h \cdot f(t_i, w_i)")
    st.latex(r"t_{i+1} = t_i + h")

# ------------------------------------------
# Tab 2: ตัวจำลอง (Simulator)
# ------------------------------------------
with tab2:
    st.header(f"ผลการคำนวณเมื่อ h = {h}")
    
    # คำนวณ Euler's Method
    t_list = [t0]
    euler_list = [y0]
    
    t_curr = t0
    w_curr = y0
    
    # วนลูปคำนวณจาก t = 1.0 ถึง 2.0
    while t_curr < t_end - 1e-9:
        w_next = w_curr + h * f(t_curr, w_curr)
        t_next = t_curr + h
        
        t_list.append(round(t_next, 6))
        euler_list.append(w_next)
        
        t_curr = t_next
        w_curr = w_next
        
    t_arr = np.array(t_list)
    euler_arr = np.array(euler_list)
    exact_arr = exact_solution(t_arr)
    error_arr = np.abs(euler_arr - exact_arr)
    
    # 3. ตารางเปรียบเทียบในรูปแบบ ti | Euler’s | Exact | Error
    df = pd.DataFrame({
        'ti': t_arr,
        "Euler’s": euler_arr,
        'Exact': exact_arr,
        'Error': error_arr
    })
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 ตารางเปรียบเทียบผลลัพธ์")
        st.write(f"**h = {h}**")
        
        # จัด Format การแสดงผลทศนิยม
        formatted_df = df.style.format({
            'ti': '{:.1f}',
            "Euler’s": '{:.7f}',
            'Exact': '{:.7f}',
            'Error': '{:.7f}'
        })
        st.dataframe(formatted_df, height=400, use_container_width=True)
        
    with col2:
        # 2. แสดงกราฟเปรียบเทียบระหว่าง exact solution และ numerical solution
        st.subheader("📈 กราฟเปรียบเทียบ")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        
        # สร้างเส้น Exact Solution ที่ละเอียดสำหรับวาดกราฟเรียบๆ
        t_dense = np.linspace(t0, t_end, 200)
        y_dense = exact_solution(t_dense)
        
        ax.plot(t_dense, y_dense, 'r-', label='Exact Solution', linewidth=2)
        ax.plot(t_arr, euler_arr, 'b--o', label="Euler's Method", markersize=5)
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        ax.set_title("Exact vs Numerical Solution")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# ------------------------------------------
# Tab 3: สรุปผล (Summary)
# ------------------------------------------
with tab3:
    st.header("สรุปผลการประมาณค่า")
    
    max_error = df['Error'].max()
    avg_error = df['Error'].mean()
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric(label="ค่า Step size (h)", value=f"{h}")
    with col_m2:
        st.metric(label="Max Error", value=f"{max_error:.7f}")
    with col_m3:
        st.metric(label="Average Error", value=f"{avg_error:.7f}")
    
    st.success(f"คำนวณสำเร็จทั้งหมด {len(df) - 1} ขั้นตอน (Steps)")
