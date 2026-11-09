import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
st.set_page_config(page_title='의료기기 품목 검색·성능 평가', page_icon='🩺', layout='wide')

@st.cache_data
def load_csv(name):
    return pd.read_csv(BASE / name, dtype=str, encoding='utf-8-sig').fillna('')

@st.cache_resource
def tfidf_index(texts):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,4), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

@st.cache_resource(show_spinner='Sentence-BERT 모델을 처음 불러오는 중입니다. 시간이 걸릴 수 있습니다.')
def sbert_index(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return model, vectors

def original_parts(raw):
    m = re.search('[가-힣]',raw)
    return (raw[:m.start()].strip(),raw[m.start():].strip()) if m else (raw,'')

def get_scores(q, method, texts):
    if method == 'TF-IDF':
        v,m = tfidf_index(texts)
        return cosine_similarity(v.transform([q]),m).ravel()
    model,m = sbert_index(texts)
    return np.asarray(model.encode([q], normalize_embeddings=True, show_progress_bar=False) @ m.T).ravel()

def rank(q, method, texts):
    scores=get_scores(q,method,texts)
    return np.argsort(-scores), scores

rows=load_csv('medical_devices_500.csv')
required=['품목분류번호','품목명','등급','PDF 원문 추출(영문명·정의 포함)','검수상태']
if any(c not in rows.columns for c in required):
    st.error('DB 열 구성이 예상과 다릅니다.'); st.stop()
texts=tuple((r['품목명']+' ')*8+r['PDF 원문 추출(영문명·정의 포함)'] for _,r in rows.iterrows())
codes=rows['품목분류번호'].tolist()

st.title('🩺 의료기기 품목 검색 및 검색 성능 평가')
st.caption(f'식약처 [별표 1] PDF에서 구성한 {len(rows)}개 1·2등급 품목 · 캡스톤 연구용')
st.warning('유사도는 품목 분류 확률이 아닙니다. 이 웹사이트는 법적 품목·등급 판정 도구가 아닙니다.')

method=st.radio('검색 방식', ['TF-IDF','Sentence-BERT'], horizontal=True, help='Sentence-BERT는 첫 실행 시 모델 다운로드가 필요하며 시간이 걸릴 수 있습니다.')
tab1,tab2,tab3=st.tabs(['🔎 의료기기 검색','📊 검색 성능 평가','📋 전체 DB'])
with tab1:
    query=st.text_area('사용목적·작동원리·사용부위를 설명해 주세요',value='귀에 넣어 적외선으로 체온을 측정하는 기기',height=105)
    # Thresholds are exploratory, not validated probability cutoffs.
    threshold=st.slider('낮은 유사도 경고 기준 (실험용)',0.0,1.0,0.12 if method=='TF-IDF' else 0.35,0.01)
    if st.button('유사 품목 검색',type='primary'):
        if len(query.strip())<5:st.error('5글자 이상 입력해 주세요.')
        else:
            try:
                order,scores=rank(query.strip(),method,texts)
                top=float(scores[order[0]])
                if top<threshold:
                    st.error('일치하는 품목을 찾기 어렵습니다. 아래 후보는 관련성이 낮을 수 있으므로 분류에 사용하지 마세요.')
                else:
                    st.info('아래는 텍스트상 가까운 후보입니다. DB에 정확한 품목이 없더라도 후보가 표시될 수 있습니다.')
                for pos,idx in enumerate(order[:3],1):
                    r=rows.iloc[int(idx)];eng,definition=original_parts(r['PDF 원문 추출(영문명·정의 포함)'])
                    with st.container(border=True):
                        st.markdown(f"**{pos}위 · {r['품목명']}**")
                        st.write(f"품목코드: {r['품목분류번호']} · 등급: {r['등급']}등급 · 검수상태: {r['검수상태']}")
                        st.progress(float(np.clip(scores[idx],0,1)),text=f'{method} 유사도: {scores[idx]:.3f}')
                        st.write(f'영문명: {eng or "—"}')
                        st.write(f'한국어 정의: {definition or "원문 확인 필요"}')
            except Exception as e:
                st.error(f'검색 모델 실행에 실패했습니다: {e}')
                st.info('Sentence-BERT 모델 다운로드가 실패하면 TF-IDF를 선택해 주세요.')
with tab2:
    st.subheader('정답 후보 30문항 평가')
    st.caption('문항·기대 코드가 포함된 평가용 자료입니다. 기대 코드는 연구자가 가정한 것으로 전문가 검증 전이며 실제 규제상 정답을 보장하지 않습니다.')
    tests=load_csv('평가문항_30개.csv')
    if st.button('현재 선택한 검색 방식 평가 실행'):
        try:
            details=[]
            for _,t in tests.iterrows():
                order,scores=rank(t['입력 설명'],method,texts)
                topcodes=[codes[int(i)] for i in order[:3]]
                details.append({'번호':t['번호'],'입력 설명':t['입력 설명'],'기대 품목코드':t['기대 품목분류번호'],'1위 품목코드':topcodes[0],'상위 3개 코드':', '.join(topcodes),'Top-1 정답':int(topcodes[0]==t['기대 품목분류번호']),'Top-3 정답':int(t['기대 품목분류번호'] in topcodes),'1위 유사도':round(float(scores[order[0]]),3)})
            result=pd.DataFrame(details)
            a,b,c=st.columns(3)
            a.metric('평가 문항',len(result));b.metric('Top-1 정확도',f"{result['Top-1 정답'].mean():.1%}");c.metric('Top-3 포함률',f"{result['Top-3 정답'].mean():.1%}")
            st.dataframe(result,use_container_width=True,hide_index=True)
            st.download_button('평가 결과 CSV 다운로드',result.to_csv(index=False).encode('utf-8-sig'),file_name=f'evaluation_{method}.csv',mime='text/csv')
            st.info('Top-1: 기대 품목이 첫 번째 결과인지. Top-3: 기대 품목이 상위 세 결과에 있는지. 동일한 30문항으로 두 모델을 각각 실행해 비교하세요.')
        except Exception as e:st.error(f'평가 실행 실패: {e}')
    st.divider()
    st.subheader('DB 외 입력 사례 5문항')
    st.caption('실제로 DB에 없는 품목인지 별도 검증이 필요하며, 낮은 유사도만으로 DB 외 품목을 확정할 수 없습니다.')
    ood=load_csv('DB외_시험문항_5개.csv')
    if st.button('DB 외 입력 탐색 실행'):
        try:
            out=[]
            for _,t in ood.iterrows():
                order,scores=rank(t['입력 설명'],method,texts)
                out.append({'입력 설명':t['입력 설명'],'최상위 추천':rows.iloc[int(order[0])]['품목명'],'최상위 유사도':round(float(scores[order[0]]),3),'실험 기준 이하':bool(scores[order[0]] < (0.12 if method=='TF-IDF' else 0.35))})
            st.dataframe(pd.DataFrame(out),use_container_width=True,hide_index=True)
        except Exception as e:st.error(f'실행 실패: {e}')
with tab3:
    st.dataframe(rows,use_container_width=True,hide_index=True)
st.divider()
st.caption('검색 방식: TF-IDF 문자 n-gram / Sentence-BERT 다국어 문장 임베딩. 모델·평가 정답 및 경고 기준은 추가 검증이 필요합니다.')
