# 의료기기 품목 검색·평가 웹사이트 — 개선 통합본

## 구성
- `app.py`: Streamlit 웹사이트 (검색, 두 모델 비교, DB 외 입력 점검, DB 확인)
- `medical_devices_500.csv`: 기존 500개 DB **원본 5개 열 유지**
- `requirements.txt`: 실행 의존성
- `평가문항_30개.csv`: 잠정 기대 품목코드가 포함된 30개 문항
- `TFIDF_30문항_재평가결과.csv`: 개선본 검색 코드로 재실행한 결과
- `DB외_시험문항_5개.csv`: DB 외 입력 후보 사례 (DB 외 여부 전문가 검증 전)
- `최종_개선결과_보고서.md`: 완료 내용과 남은 검증 사항

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub / Streamlit Community Cloud
GitHub 저장소 루트에 압축을 푼 파일들을 업로드하고 Commit changes를 누릅니다. Streamlit Cloud의 앱이 이 저장소와 `app.py`에 연결돼 있다면 변경사항이 반영됩니다. Python 의존성 설치 및 Sentence-BERT 모델 다운로드에는 시간이 걸릴 수 있습니다.

## 주요 개선
1. 동일 30문항으로 TF-IDF와 Sentence-BERT의 Top-1/Top-3 비교를 웹에서 직접 실행
2. 두 모델 문항별 결과를 나란히 비교하고 CSV 다운로드
3. 5개 DB 외 입력 사례를 탐색하고 경고 기준 표시
4. 기준 이하 후보는 강제로 3개 추천하지 않고 숨김
5. 원본 DB의 5개 열, 500개 행을 유지

## 중요한 한계
- **TF-IDF는 30문항 재실행 검증 완료:** Top-1 7/30 (23.3%), Top-3 13/30 (43.3%).
- **Sentence-BERT 실제 모델 실행 및 수치 비교는 아직 미완료:** 이 파일 제작 환경에 `sentence-transformers`가 설치되지 않아 실행 검증하지 못했습니다. Streamlit 배포 환경에서 모델을 다운로드해 비교 버튼을 실행해야 합니다.
- 30문항의 기대 품목코드는 연구자가 설정한 잠정 기준이며 의료기기 전문가가 정답으로 검증하지 않았습니다.
- 5개 DB 외 입력 사례의 실제 DB 외 여부도 확정되지 않았습니다. 실험용 임계값은 DB 외 판별의 검증된 기준이 아닙니다.
- 검색 결과는 품목 후보이며, 식약처의 법적 분류 판정이 아닙니다.
