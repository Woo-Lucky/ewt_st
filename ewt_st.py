import streamlit as st
import base64
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import ewtpy
import os

def get_image_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode()

logo_base64 = get_image_base64("logo.png")  # 경로 정확히 확인

# 엑셀 스타일 열 이름(A, B, ..., AA 등)을 숫자로 변환
def excel_col_to_index(col):
    col = col.upper()
    index = 0
    for i, c in enumerate(reversed(col)):
        index += (ord(c) - ord('A') + 1) * (26 ** i)
    return index - 1

# 페이지 설정
st.set_page_config(page_title="EJFilter v0.1", layout="wide")

# 사이드바: 파일 업로드 및 파라미터 설정
with st.sidebar:
    st.image("logo.png", width=300)
    st.title("파라미터 설정")
    uploaded_file = st.file_uploader("🔗CSV 파일 업로드", type=["csv"] )
    delimiter = st.text_input("구분자(Delimiter)", value=",", help="CSV 파일의 구분자 입력 (기본값: ',')")
    col_input = st.text_input("데이터 열 인덱스(col)", value="A", help="분석할 데이터가 있는 열 입력 ex) A, B, C, ..., AA, AB, ...")
    col = excel_col_to_index(col_input)
    start_row = st.number_input("시작 행 인덱스(Start_row)", min_value=0, value=0)
    end_row = st.number_input("끝 행 인덱스(End_row, -1은 끝까지)", value=-1)
    n_support = st.number_input("최대 모드 수(N_support)", min_value=1, value=3)
    log = st.checkbox("로그 스펙트럼(Log)", value=False)
    detect = st.selectbox("탐지 모드(Detect)", ["Logmax", "Logmaxmin", "Logmaxminf"], index=0)
    completion = st.checkbox("모드 완성(Completion)", value=False)
    reg = st.selectbox("보정(Reg)", ["none", "gaussian", "average"], index=2)
    w_filter = st.number_input("필터 너비(W_filter)", min_value=0, value=10)
    s_filter = st.number_input("Gaussian 표준편차(S_filter)", min_value=0, value=5)
    avgs_input = st.text_input("평균 리스트(Avgs) ex: [0,0,0]", value="")
    # plot_all = st.checkbox("전체 플롯(Plot_all)", value=False)
    # no_plot = st.checkbox("플롯 미표시(No_plot)", value=False)
    no_plot = st.selectbox("결과 그래프 표시 여부", ["표시", "미표시"], index=0)
    if no_plot == "미표시":
        no_plot = True
    else:
        no_plot = False
    run_button = st.button("📈분석 실행")

# 메인 영역: 안내 및 결과
st.title("EJFilter v0.1")
st.subheader("Based on EWT")

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <div style="font-size:24px; font-weight: 600; "margin-right: 10px;">made by </div>
        <img src="data:image/png;base64,{logo_base64}" width="100">
    </div>
    """,
    unsafe_allow_html=True
)

if not uploaded_file:
    st.info("왼쪽 사이드바에서 CSV 파일을 업로드하고 파라미터를 설정하세요.")
    st.stop()

if uploaded_file and run_button:
    # CSV 읽기
    try:
        df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding='cp949', low_memory=False)
    except Exception as e:
        st.error(f"CSV 읽기 실패: {e}")
        st.stop()

    # 데이터 추출
    data = df.iloc[start_row:end_row if end_row != -1 else None, col].to_numpy()

    # 평균 리스트 파싱
    avgs = None
    if avgs_input:
        try:
            tmp = ast.literal_eval(avgs_input)
            if isinstance(tmp, list) and len(tmp) == n_support:
                avgs = tmp
            else:
                st.warning(f"평균 리스트 길이를 최대 모드 수({n_support})에 맞춰주세요.")
        except Exception:
            st.warning("평균 리스트 파싱 실패")

    # EWT 계산
    ewt, mfb, boundaries = ewtpy.EWT1D(
        data,
        N=n_support,
        log=int(log),
        detect=detect,
        completion=int(completion),
        reg=reg,
        lengthFilter=w_filter,
        sigmaFilter=s_filter
    )
    
    diffs = [(ewt[:, i] - (avgs[i] if avgs else ewt[:, i].mean())) for i in range(ewt.shape[1])]
    out = np.column_stack([ewt] + diffs)
    cols = [f"Mode{i+1}" for i in range(ewt.shape[1])] + [f"Mode{i+1}_diff" for i in range(ewt.shape[1])]
    out_df = pd.DataFrame(out, columns=cols)
    
    # 데이터 샘플 표시 (분석 결과)
    st.subheader("결과 데이터 미리보기")
    st.dataframe(out_df)
    
    # 플롯 표시
    if not no_plot:
        st.success("아래에서 CSV 파일을 다운로드할 수 있습니다.")
        
        # 4개의 서브플롯: 원본, 컴포넌트, 정규화, 평균 차이
        fig, axes = plt.subplots(4, 1, figsize=(20, 15), sharex=True)
        labels = [f"Mode {i+1}" for i in range(ewt.shape[1])]

        axes[0].set_title('Original')
        axes[0].plot(data)
        axes[0].grid(linestyle='--')

        axes[1].set_title('EWT')
        for i in range(ewt.shape[1]):
            axes[1].plot(ewt[:, i], label=labels[i], alpha=1 - i*0.2)
        axes[1].legend()
        axes[1].grid(linestyle='--')

        axes[2].set_title('Normalized Components')
        for i in range(ewt.shape[1]):
            mn, mx = ewt[:, i].min(), ewt[:, i].max()
            norm = (ewt[:, i] - mn) / (mx - mn)
            axes[2].plot(norm, label=labels[i], alpha=1 - i*0.2)
        axes[2].legend()
        axes[2].grid(linestyle='--')

        axes[3].set_title('EWT - Average')
        for i in range(ewt.shape[1]):
            base = avgs[i] if avgs else ewt[:, i].mean()
            axes[3].plot(ewt[:, i] - base, label=labels[i], alpha=1 - i*0.2)
        axes[3].legend()
        axes[3].grid(linestyle='--')

        st.pyplot(fig, use_container_width=True)
        st.info("⬇️CSV 파일 다운로드⬇️")

    # 결과 다운로드
    csv_data = out_df.to_csv(index=False).encode('utf-8')
    original_name = os.path.splitext(uploaded_file.name)[0]
    filename = f"{original_name}_ewt.csv"
    st.download_button("💾EWT 결과 CSV 다운로드", csv_data, file_name=filename, mime='text/csv')

elif uploaded_file:
    st.warning("파라미터를 설정한 후 '분석 실행' 버튼을 눌러주세요.")
