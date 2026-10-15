import csv
from pathlib import Path
import re

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="의료기기 품목 검색 도우미", page_icon="🩺", layout="wide")

@st.cache_data
def load_data():
    path = Path(__file__).with_name("medical_devices_500.csv")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("품목 DB가 비어 있습니다.")
    return rows

@st.cache_resource
def build_index(texts):
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True)
    return vectorizer, vectorizer.fit_transform(texts)

def search(query, rows, n=3):
    # PDF 원문 추출에는 영문명 및 줄바꿈 정리 흔적이 포함될 수 있음.
    texts = tuple(" ".join([r["품목명"]] * 8 + [r["PDF 원문 추출(영문명·정의 포함)"]]) for r in rows)
    vec, matrix = build_index(texts)
    scores = cosine_similarity(vec.transform([query]), matrix).ravel()
    best = sorted(range(len(rows)), key=lambda i: float(scores[i]), reverse=True)[:n]
    return [(rows[i], float(scores[i])) for i in best]

rows = load_data()
st.title("🩺 의료기기 품목·등급 유사도 검색")
st.caption(f"캡스톤 시연용 · 업로드된 식약처 [별표 1] PDF 자동 추출 데이터 · 1·2등급 {len(rows):,}개")
st.warning("이 시스템은 참고용 유사 품목 검색 도구입니다. 공식 품목분류·등급 판정이 아니며, 원문 자동 추출 및 검수가 필요한 DB입니다. 3·4등급, 체외진단 및 별도 규율 대상 품목을 판정하거나 신뢰성 있게 배제하지 못합니다.")
with st.form("search_form"):
    query = st.text_area("의료기기 사용목적, 작동원리, 사용부위를 설명해 주세요", value="귀에 넣어 적외선으로 체온을 측정하는 기기", max_chars=1500, height=120)
    submitted = st.form_submit_button("유사 품목 3개 검색", type="primary", use_container_width=True)
if submitted:
    query = query.strip()
    if len(query) < 5:
        st.error("설명을 5글자 이상 입력해 주세요.")
    else:
        out_of_scope = bool(re.search(r"3\s*등급|4\s*등급|체외\s*진단|IVD|디지털\s*의료", query, re.I))
        if out_of_scope:
            st.warning("입력에 DB 범위 밖일 수 있는 표현이 있습니다. 아래 결과는 분류 판정이 아닙니다.")
        results = search(query, rows)
        st.subheader("유사 품목 상위 3개")
        st.caption("TF-IDF 문자 n-gram 코사인 유사도. 수치는 정답 확률이 아니며, 분류 정확도와도 다릅니다.")
        if results[0][1] < 0.12:
            st.warning("유사도가 낮아 검색 결과가 부정확할 수 있습니다. 0.12는 검증되지 않은 실험용 경고 기준입니다.")
        for rank, (r, score) in enumerate(results, 1):
            with st.container(border=True):
                st.markdown(f"### {rank}위 · {r['품목명']}")
                st.write(f"**품목분류번호:** {r['품목분류번호']}　|　**원문 기재 등급:** {r['등급']}등급")
                st.progress(min(1.0, max(0.0, score)), text=f"텍스트 유사도: {score:.3f}")
                st.write("**PDF 원문 자동 추출 텍스트 (영문명·정의 혼재 가능):**")
                st.write(r["PDF 원문 추출(영문명·정의 포함)"] or "추출된 텍스트 없음")
                st.caption("추천 근거: 제품 설명과 품목명·원문 추출 텍스트 간의 문자 단위 유사도. 수동 검증 전 자료입니다.")
        st.info("DB에 없거나 범위 밖인 의료기기도 유사한 1·2등급 품목이 표시될 수 있습니다. 최종 분류는 최신 규정 및 전문 검토를 확인하세요.")
with st.expander(f"전체 검색 대상 {len(rows):,}개 품목 보기"):
    st.dataframe([{k: r[k] for k in ("품목분류번호", "품목명", "등급", "검수상태")} for r in rows], use_container_width=True, hide_index=True)
st.divider()
st.caption("자료 출처: 사용자가 제공한 「의료기기 품목 및 품목별 등급」 [별표 1] PDF. 최신 개정 여부 미확인.")
