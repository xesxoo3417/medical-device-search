import csv
import re
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="의료기기 품목 검색 도우미", page_icon="🩺", layout="wide")

COLUMNS = ["품목분류번호", "품목명", "등급", "PDF 원문 추출(영문명·정의 포함)", "검수상태"]

@st.cache_data
def load_data():
    path = Path(__file__).with_name("medical_devices_500.csv")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"CSV 열이 예상 형식과 다릅니다: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 500:
        st.warning(f"현재 DB에 {len(rows)}개 품목이 있습니다. 예상 품목 수는 500개입니다.")
    return rows


def split_original(raw):
    """표시용으로만 영문명과 한글 정의를 나눔. CSV 자체는 변경하지 않음."""
    match = re.search(r"[가-힣]", raw)
    if match:
        return raw[:match.start()].strip(), raw[match.start():].strip()
    return raw.strip(), ""


@st.cache_resource
def build_index(texts):
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True)
    return vectorizer, vectorizer.fit_transform(texts)


def search(query, rows, n=3):
    texts = tuple(" ".join([r["품목명"]] * 8 + [r["PDF 원문 추출(영문명·정의 포함)"]]) for r in rows)
    vectorizer, matrix = build_index(texts)
    scores = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
    top = sorted(range(len(rows)), key=lambda i: float(scores[i]), reverse=True)[:n]
    return [(rows[i], float(scores[i])) for i in top]


rows = load_data()
st.title("🩺 의료기기 품목·등급 유사도 검색")
st.caption(f"캡스톤 시연용 · 제공된 [별표 1] PDF 기반 · 1·2등급 {len(rows):,}개 품목")
st.warning("이 사이트는 품목 검색을 돕는 참고용 도구이며 법적 등급 판정 서비스가 아닙니다. DB의 '검수완료'는 제공된 PDF 텍스트 대조·교정 작업에 대한 표시이며, 최신 고시의 공식 버전 확인이나 전 품목의 법규상 최종 판정을 뜻하지 않습니다.")

with st.form("search_form"):
    query = st.text_area("의료기기의 사용목적, 작동원리, 사용부위를 설명해 주세요", value="귀에 넣어 적외선으로 체온을 측정하는 기기", max_chars=1500, height=120)
    submitted = st.form_submit_button("유사 품목 3개 검색", type="primary", use_container_width=True)

if submitted:
    query = query.strip()
    if len(query) < 5:
        st.error("설명을 5글자 이상 입력해 주세요.")
    else:
        if re.search(r"3\s*등급|4\s*등급|체외\s*진단|IVD|디지털\s*의료", query, re.I):
            st.warning("입력에 현재 DB 범위 밖일 수 있는 표현이 있습니다. 아래 결과는 분류 판정이 아닙니다.")
        results = search(query, rows)
        st.subheader("유사 품목 상위 3개")
        st.caption("TF-IDF 문자 n-gram 기반 텍스트 유사도입니다. 점수는 정답 확률이 아닙니다.")
        if results and results[0][1] < 0.12:
            st.warning("유사도가 낮아 검색 결과가 부정확할 수 있습니다. 경고 기준은 실험용입니다.")
        for rank, (item, score) in enumerate(results, 1):
            english, definition = split_original(item["PDF 원문 추출(영문명·정의 포함)"])
            with st.container(border=True):
                st.markdown(f"### {rank}위 · {item['품목명']}")
                st.write(f"**품목분류번호:** {item['품목분류번호']}　|　**등급:** {item['등급']}등급　|　**검수상태:** {item['검수상태']}")
                st.progress(min(1.0, max(0.0, score)), text=f"텍스트 유사도: {score:.3f}")
                st.write(f"**영문 품목명:** {english or '원문에서 구분되지 않음'}")
                st.write("**한국어 정의(화면 표시용 자동 분리):**")
                st.write(definition or "한국어 정의를 분리할 수 없습니다. 원문을 확인하세요.")
                with st.expander("DB에 저장된 원문 텍스트 보기"):
                    st.write(item["PDF 원문 추출(영문명·정의 포함)"])
        st.info("DB에 없거나 3·4등급 등 범위 밖인 의료기기도 유사한 1·2등급 품목이 표시될 수 있습니다. 최종 분류는 최신 규정과 전문가 검토로 확인하세요.")

with st.expander(f"전체 검색 대상 {len(rows):,}개 품목 보기"):
    st.dataframe(rows, use_container_width=True, hide_index=True)

st.divider()
st.caption("자료 출처: 사용자 제공 의료기기 품목 및 품목별 등급 [별표 1] PDF. 검수상태는 제공된 CSV 기재값을 표시합니다.")
