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
# 2. st.sidebar สำหรับรับค่าพารามิเตอร์ (รับแค่ h)
# ==========================================
st.sidebar.header("⚙️ กำหนดพารามิเตอร์")

# รับค่า h เพียงอย่างเดียว
h = st.sidebar.number_input(
    "ขนาดช่วงก้าว (Step size: h)", 
    value=0.1, 
    step=0.01, 
    format="%.4f"
)

# ค่าคงที่ตามโจทย์
t0 = 1.0
t_end = 2.0
y0 = 2.1272295

# ฟังก์ชันตัวอย่างสำหรับคำนวณ (สามารถปรับเปลี่ยน f(t,y) และ exact_solution ตามโจทย์จริงได้)
def f(t, y):
    return t - y + 1

def exact_solution(t):
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
    **Euler's Method** เป็นระเบียบวิธีทางตัวเลขสำหรับหาคำตอบโดยประมาณของสมการเชิงอนุพันธ์
    """)
    st.latex(r"y_{n+1} = y_n + h \cdot f(t_n, y_n)")
    st.latex(r"t_{n+1} = t_n + h")

# ------------------------------------------
# Tab 2: ตัวจำลองและการคำนวณ
# ------------------------------------------
with tab2:
    st.header(f"ผลการคำนวณเมื่อ h = {h}")
    
    # คำนวณ Euler's Method
    t_list = [t0]
    euler_list = [y0]
    
    t_curr = t0
    y_curr = y0
    
    # คำนวณทีละ step จาก t0 ถึง t_end
    while t_curr < t_end - 1e-9:
        y_next = y_curr + h * f(t_curr, y_curr)
        t_next = t_curr + h
        
        t_list.append(round(t_next, 6))
        euler_list.append(y_next)
        
        t_curr = t_next
        y_curr = y_next
        
    t_arr = np.array(t_list)
    euler_arr = np.array(euler_list)
    exact_arr = exact_solution(t_arr)
    error_arr = np.abs(euler_arr - exact_arr)
    
    # สร้าง DataFrame ในรูปแบบ Exact, Euler's, Error ตามโจทย์
    df = pd.DataFrame({
        't': t_arr,
        "Euler’s": euler_arr,
        'Exact': exact_arr,
        'Error': error_arr
    })
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 ตารางเปรียบเทียบผลลัพธ์")
        # จัด Format ให้ทศนิยมตรงตามโจทย์ (แสดง t เป็น 1 ตำแหน่ง และค่าอื่นๆ เป็น 7 ตำแหน่ง)
        formatted_df = df.style.format({
            't': '{:.1f}',
            "Euler’s": '{:.7f}',
            'Exact': '{:.7f}',
            'Error': '{:.7f}'
        })
        st.dataframe(formatted_df, height=400, use_container_width=True)
        
    with col2:
        st.subheader("📈 กราฟเปรียบเทียบ")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_arr, exact_arr, 'r-', label='Exact Solution', linewidth=2)
        ax.plot(t_arr, euler_arr, 'b--o', label="Euler's Method", markersize=4)
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
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="ค่าความคลาดเคลื่อนสูงสุด (Max Error)", value=f"{max_error:.7f}")
    with col_m2:
        st.metric(label="ค่าความคลาดเคลื่อนเฉลี่ย (Average Error)", value=f"{avg_error:.7f}")
    
    st.markdown(f"""
    **การวิเคราะห์ผลลัพธ์:**
    * ทดสอบระบบด้วยค่าช่วงก้าว **$h = {h}$**
    * จำนวนขั้นตอนทั้งหมดในการคำนวณ: **{len(df) - 1}** ขั้นตอน
    """)
