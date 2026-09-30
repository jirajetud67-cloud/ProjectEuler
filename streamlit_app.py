import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==========================================
# 1. การตั้งค่าหน้าเว็บหลัก
# ==========================================
st.set_page_config(page_title="Euler's Method Simulator", layout="wide")
st.title("🧮 Euler's Method Simulator")

# ==========================================
# 2. st.sidebar สำหรับรับค่าพารามิเตอร์
# ==========================================
st.sidebar.header("⚙️ กำหนดพารามิเตอร์")

# กำหนดตัวอย่างปัญหา dy/dt = f(t, y)
st.sidebar.markdown("**โจทย์ตัวอย่าง:** $y' = t - y + 1$ โดยที่ $y(0) = 1$")

t0 = st.sidebar.number_input("ค่าเริ่มต้น t (t0)", value=0.0, step=0.1)
y0 = st.sidebar.number_input("ค่าเริ่มต้น y (y0)", value=1.0, step=0.1)
t_end = st.sidebar.number_input("ค่าสุดท้าย t (t_end)", value=2.0, step=0.1)
h = st.sidebar.number_input("ขนาดช่วงก้าว (Step size: h)", value=0.1, step=0.01, format="%.3f")

# ฟังก์ชันฟิสิกส์/คณิตศาสตร์ (กำหนดตามโจทย์)
def f(t, y):
    return t - y + 1

def exact_solution(t):
    # สมการ Exact Solution ของ y' = t - y + 1, y(0)=1 คือ y(t) = t + e^(-t)
    return t + np.exp(-t)

# ==========================================
# 3. st.tabs แบ่งเนื้อหาออกเป็นส่วนๆ
# ==========================================
tab1, tab2, tab3 = st.tabs(["📚 ทฤษฎี (Theory)", "🎮 ตัวจำลอง (Simulator)", "📊 สรุปผล (Summary)"])

# ------------------------------------------
# Tab 1: ทฤษฎี
# ------------------------------------------
with tab1:
    st.header("ระเบียบวิธีของออยเลอร์ (Euler's Method)")
    st.write("""
    **Euler's Method** เป็นระเบียบวิธีทางตัวเลข (Numerical Method) สำหรับหาคำตอบโดยประมาณของสมการเชิงอนุพันธ์อันดับหนึ่ง 
    ที่มีรูปแบบปัญหาค่าเริ่มต้น (Initial Value Problem - IVP):
    """)
    st.latex(r"\frac{dy}{dt} = f(t, y), \quad y(t_0) = y_0")
    
    st.write("สูตรการประมาณค่าในแต่ละขั้นตอน:")
    st.latex(r"y_{n+1} = y_n + h \cdot f(t_n, y_n)")
    st.latex(r"t_{n+1} = t_n + h")
    
    st.write("โดยที่ $h$ คือขนาดของช่วงก้าว (Step size)")

# ------------------------------------------
# Tab 2: ตัวจำลองและการคำนวณ
# ------------------------------------------
with tab2:
    st.header("ผลการคำนวณและเปรียบเทียบ")
    
    # ประมวลผลคำนวณ Euler's Method
    n_steps = int((t_end - t0) / h) + 1
    t_values = [t0]
    euler_values = [y0]
    
    t_curr = t0
    y_curr = y0
    
    for _ in range(n_steps - 1):
        y_next = y_curr + h * f(t_curr, y_curr)
        t_next = t_curr + h
        
        t_values.append(t_next)
        euler_values.append(y_next)
        
        t_curr = t_next
        y_curr = y_next
        
    t_values = np.array(t_values)
    euler_values = np.array(euler_values)
    exact_values = exact_solution(t_values)
    errors = np.abs(euler_values - exact_values)
    
    # สร้าง DataFrame ตาม Format ที่ต้องการ
    df = pd.DataFrame({
        't': t_values,
        "Euler's": euler_values,
        'Exact': exact_values,
        'Error': errors
    })
    
    # แสดงผลด้วย Columns (ซ้าย: ตาราง, ขวา: กราฟ)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 ตารางเปรียบเทียบผลลัพธ์")
        # แสดงผลตารางด้วยทศนิยม 7 ตำแหน่งตามโจทย์
        st.dataframe(
            df.style.format({
                't': '{:.1f}',
                "Euler's": '{:.7f}',
                'Exact': '{:.7f}',
                'Error': '{:.7f}'
            }),
            height=400
        )
        
    with col2:
        st.subheader("📈 กราฟเปรียบเทียบ")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_values, exact_values, 'r-', label='Exact Solution', linewidth=2)
        ax.plot(t_values, euler_values, 'b--o', label="Euler's Method", markersize=4)
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        ax.set_title("Exact vs Numerical Solution")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# ------------------------------------------
# Tab 3: สรุปผล
# ------------------------------------------
with tab3:
    st.header("สรุปผลการวิเคราะห์")
    
    max_error = df['Error'].max()
    avg_error = df['Error'].mean()
    
    st.metric(label="ค่าความคลาดเคลื่อนสูงสุด (Max Error)", value=f"{max_error:.7f}")
    st.metric(label="ค่าความคลาดเคลื่อนเฉลี่ย (Average Error)", value=f"{avg_error:.7f}")
    
    st.markdown("""
    **ข้อสังเกต:**
    * เมื่อปรับค่า **Step size ($h$)** ให้มีขนาดเล็กลง ค่าความคลาดเคลื่อน (Error) จะลดลงอย่างเห็นได้ชัด
    * Euler's Method เหมาะสมกับการเรียนรู้พื้นฐาน แต่สำหรับการคำนวณที่ต้องการความแม่นยำสูง อาจพิจารณาใช้วิธีอื่น เช่น **Runge-Kutta (RK4)**
    """)
