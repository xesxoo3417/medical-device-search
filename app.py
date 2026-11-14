"""Medical device candidate retrieval — research prototype, not regulatory advice."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='의료기기 품목 검색 · 최종 연구용', page_icon='🩺', layout='wide')

@st.cache_data
def csv_file(name):
    return pd.read_csv(BASE/name, dtype=str, encoding='utf-8-sig').fillna('')

@st.cache_resource
def tfidf_index(documents):
    vec = TfidfVectorizer(analyzer='char', ngram_range=(2,4), sublinear_tf=True)
    return vec, vec.fit_transform(documents)

@st.cache_resource(show_spinner='Sentence-BERT 임베딩을 생성하는 중입니다 (첫 실행 시 다운로드 필요).')
def sbert_index(documents):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    matrix = model.encode(list(documents), normalize_embeddings=True, show_progress_bar=False)
    return model, np.asarray(matrix)

@st.cache_data(show_spinner=False)
def score_tfidf(query, documents):
    vectorizer, matrix = tfidf_index(documents)
    return cosine_similarity(vectorizer.transform([query]), matrix).ravel()

@st.cache_data(show_spinner=False)
def score_sbert(query, documents):
    model, matrix = sbert_index(documents)
    q = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(q @ matrix.T).ravel()

def rank(query, method, documents):
    scores = score_tfidf(query,documents) if method=='TF-IDF' else score_sbert(query,documents)
    return np.argsort(-scores), scores

def parts(raw):
    match=re.search('[가-힣]',raw)
    return (raw[:match.start()].strip(),raw[match.start():].strip()) if match else (raw.strip(),'')

def evaluate(tests, method, documents, codes):
    results=[]
    for _,t in tests.iterrows():
        order,scores=rank(t['입력 설명'],method,documents)
        picks=[codes[int(i)] for i in order[:3]]
        expected=t['기대 품목분류번호']
        results.append({'번호':t['번호'],'입력 설명':t['입력 설명'],'기대 품목코드':expected,'1위 코드':picks[0], '상위 3개 코드':' / '.join(picks), 'Top-1 일치':int(expected==picks[0]),'Top-3 포함':int(expected in picks),'1위 유사도':round(float(scores[order[0]]),4)})
    return pd.DataFrame(results)

def download_csv(label, data, filename):
    st.download_button(label, data.to_csv(index=False).encode('utf-8-sig'), file_name=filename, mime='text/csv')

rows=csv_file('medical_devices_500.csv')
expected_cols=['품목분류번호','품목명','등급','PDF 원문 추출(영문명·정의 포함)','검수상태']
if list(rows.columns)!=expected_cols:
    st.error('DB의 원래 5개 열 구성이 달라졌습니다.'); st.stop()
if len(rows)!=500 or rows['품목분류번호'].duplicated().any():
    st.error('DB 행 수 또는 품목코드 중복을 확인해 주세요.'); st.stop()
texts=tuple(((r['품목명']+' ')*8+r['PDF 원문 추출(영문명·정의 포함)']).strip() for _,r in rows.iterrows())
codes=rows['품목분류번호'].tolist()
code_set=set(codes)
tests=csv_file('평가문항_30개.csv')
if not set(tests['기대 품목분류번호']).issubset(code_set):
    st.error('평가 문항 중 DB에 없는 기대 품목코드가 있습니다.'); st.stop()

st.title('🩺 의료기기 품목 유사도 검색 · 성능 평가')
st.caption('식약처 [별표 1] 추출 기반 1·2등급 500개 품목 · 캡스톤 연구용 프로토타입')
st.warning('검색 결과는 품목 후보일 뿐 법적 분류 판정이 아닙니다. 유사도 수치는 분류 확률이 아닙니다. DB 밖의 품목이나 3·4등급 의료기기는 적절히 검색되지 않을 수 있습니다.')
method=st.radio('검색 방식', ['TF-IDF','Sentence-BERT'], horizontal=True)
search_tab,eval_tab,errors_tab,db_tab=st.tabs(['🔎 품목 검색','📊 동일 문항 모델 비교','🧪 DB 외 입력 점검','📋 원본 DB'])
with search_tab:
    query=st.text_area('기기의 사용목적·작동원리·사용부위를 설명해 주세요',value='귀에 넣어 적외선으로 체온을 측정하는 기기',height=90)
    st.caption('경고 기준은 검증된 의료기기 분류 임계값이 아닙니다. 모델별 유사도 분포가 달라 기준을 공유할 수 없습니다.')
    cutoff=st.slider('검색 후보 표시 최소 유사도 (실험용)',0.0,1.0,0.12 if method=='TF-IDF' else 0.35,0.01,key=f'cutoff_{method}')
    if st.button('검색 실행',type='primary'):
        if len(query.strip())<5:
            st.error('5글자 이상 입력해 주세요.')
        else:
            try:
                order,scores=rank(query.strip(),method,texts)
                selected=[int(i) for i in order[:3] if float(scores[i])>=cutoff]
                if not selected:
                    st.error('설정한 기준을 넘는 후보가 없습니다. 결과를 억지로 추천하지 않습니다. DB에 품목이 없다는 확정 판정은 아닙니다.')
                else:
                    if len(selected)<3: st.info(f'기준 이상인 후보 {len(selected)}개만 표시합니다. 낮은 점수의 후보는 숨겼습니다.')
                    else: st.info('상위 3개 후보입니다. 실제 품목 적합성은 정의와 규정을 대조해 확인하세요.')
                    for pos,i in enumerate(selected,1):
                        r=rows.iloc[i];eng,definition=parts(r['PDF 원문 추출(영문명·정의 포함)'])
                        with st.container(border=True):
                            st.subheader(f'{pos}위 · {r["품목명"]}')
                            st.write(f'품목분류번호: {r["품목분류번호"]}  |  등급: {r["등급"]}등급  |  DB 검수상태: {r["검수상태"]}')
                            st.metric('텍스트 유사도 (분류 확률 아님)',f'{float(scores[i]):.3f}')
                            st.write('영문명:',eng or '—')
                            st.write('한국어 정의:',definition or 'PDF 원문 확인 필요')
            except Exception as exc:
                st.error(f'모델 실행 실패: {exc}')
                st.info('Sentence-BERT의 첫 실행에는 인터넷과 모델 다운로드가 필요합니다. 실패 시 TF-IDF로 전환하세요.')
with eval_tab:
    st.write('**동일한 30개 입력과 동일한 기대 품목코드**를 기준으로 두 방법의 Top-1·Top-3 결과를 비교합니다.')
    st.caption('기대 품목코드는 연구자가 설정한 잠정 정답이며 전문가가 검증한 정답표가 아닙니다. 따라서 지표는 최종 규제 분류 정확도가 아닙니다.')
    st.write(f'평가 문항 {len(tests)}개 / DB 내 기대 품목 {tests["기대 품목분류번호"].isin(code_set).sum()}개')
    if st.button('TF-IDF · Sentence-BERT 둘 다 평가 실행',type='primary'):
        try:
            with st.spinner('두 검색 방식의 30문항을 평가하고 있습니다.'):
                a=evaluate(tests,'TF-IDF',texts,codes)
                b=evaluate(tests,'Sentence-BERT',texts,codes)
            summary=pd.DataFrame([{'모델':'TF-IDF','Top-1 일치율':a['Top-1 일치'].mean(),'Top-3 포함률':a['Top-3 포함'].mean()}, {'모델':'Sentence-BERT','Top-1 일치율':b['Top-1 일치'].mean(),'Top-3 포함률':b['Top-3 포함'].mean()}])
            st.dataframe(summary.style.format({'Top-1 일치율':'{:.1%}','Top-3 포함률':'{:.1%}'}),hide_index=True,use_container_width=True)
            st.bar_chart(summary.set_index('모델')[['Top-1 일치율','Top-3 포함률']])
            detail=a.merge(b,on=['번호','입력 설명','기대 품목코드'],suffixes=('_TFIDF','_SBERT'))
            st.subheader('문항별 결과 및 오류 분석')
            st.dataframe(detail,use_container_width=True,hide_index=True)
            download_csv('두 모델 비교 결과 다운로드',detail,'모델비교_30문항_상세.csv')
            download_csv('두 모델 요약 다운로드',summary,'모델비교_30문항_요약.csv')
        except Exception as exc:
            st.error(f'비교 평가 실행 실패: {exc}')
            st.info('Sentence-BERT 실행 환경과 모델 다운로드 상태를 확인하세요. TF-IDF 단독 평가는 아래 버튼으로 실행할 수 있습니다.')
    if st.button('TF-IDF 단독 평가'):
        a=evaluate(tests,'TF-IDF',texts,codes)
        c1,c2=st.columns(2)
        c1.metric('Top-1 일치율',f'{a["Top-1 일치"].mean():.1%}')
        c2.metric('Top-3 포함률',f'{a["Top-3 포함"].mean():.1%}')
        st.dataframe(a,use_container_width=True,hide_index=True)
        download_csv('TF-IDF 결과 다운로드',a,'TFIDF_30문항_평가.csv')
    with st.expander('평가 문항과 잠정 기대 품목코드 확인'):
        st.dataframe(tests,use_container_width=True,hide_index=True)
with errors_tab:
    st.write('**DB 밖 입력 5문항**의 검색 점수를 살펴보고, 관련성이 낮은 경우 후보 표시를 제한하는지 확인합니다.')
    st.caption('이 5문항이 실제로 DB에 없는지 검증된 정답표는 없습니다. 임계값은 실험용이며, DB 외 판별 정확도·재현율을 주장할 수 없습니다.')
    out=csv_file('DB외_시험문항_5개.csv')
    if st.button('DB 외 사례 5개 시험'):
        try:
            vals=[]
            cutoff=0.12 if method=='TF-IDF' else 0.35
            for _,r in out.iterrows():
                order,scores=rank(r['입력 설명'],method,texts)
                top=int(order[0]); score=float(scores[top])
                vals.append({'입력 설명':r['입력 설명'],'추천 1위':rows.iloc[top]['품목명'],'최상위 유사도':round(score,4),'실험 기준':cutoff,'표시 여부':'후보 보류' if score<cutoff else '후보 표시'})
            result=pd.DataFrame(vals)
            st.dataframe(result,use_container_width=True,hide_index=True)
            download_csv('DB 외 입력 결과 다운로드',result,'DB외_5문항_시험결과.csv')
        except Exception as exc:st.error(f'시험 실행 실패: {exc}')
with db_tab:
    st.write('원본 500개 DB는 **기존 5개 열을 변경하지 않고** 그대로 사용합니다.')
    st.dataframe(rows,use_container_width=True,hide_index=True)
    download_csv('원본 DB 다운로드',rows,'medical_devices_500.csv')
st.divider()
st.caption('모델 비교와 경고 기능은 연구·실험용입니다. 최종발표 전에 정답 라벨의 전문가 검토, 외부 테스트 세트, 모델 실행 환경 검증이 필요합니다.')
