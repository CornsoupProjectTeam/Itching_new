import os
import json
import numpy as np
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from tqdm import tqdm
import warnings
import logging

# 경고 메시지 필터링
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="sentence_transformers")


# 환경 변수 로드
load_dotenv()
ELASTICSEARCH_CLOUD_ID = os.getenv("ELASTICSEARCH_CLOUD_ID")
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")

class BGEM3Embedder:
    """BGE 임베딩 모델"""
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        """텍스트를 임베딩으로 변환"""
        return self.model.encode([text], normalize_embeddings=True)[0]


class BGEReranker:
    """BGE 리랭커 모델"""
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = SentenceTransformer(model_name)

    def rank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """리랭커로 문서 정렬"""
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        doc_texts = [doc["chunk_with_context"] for doc in documents]
        doc_embeddings = self.model.encode(doc_texts, normalize_embeddings=True)
        scores = cosine_similarity(query_embedding, doc_embeddings).flatten()
        return [{"text": doc["chunk_with_context"], "score": float(score)} for doc, score in zip(documents, scores)]


def fetch_knn_results(es, index, query_embedding, size=20):
    """Elasticsearch kNN 쿼리"""
    response = es.search(index=index, body={
        "knn": {
            "field": "embedding",
            "query_vector": query_embedding.tolist(),
            "k": size,
            "num_candidates": 100
        }
    })
    return [
        {
            "_id": doc["_id"],
            "_score": doc["_score"],
            "sentence": doc["_source"].get("sentence", ""),
            "chunk_with_context": doc["_source"].get("chunk_with_context", ""),
            "factor": doc["_source"].get("factor", ""),
            "impact": doc["_source"].get("impact", "")
        }
        for doc in response["hits"]["hits"]
    ]


def fetch_bm25_results(es, index, query, size=20):
    """Elasticsearch BM25 쿼리"""
    response = es.search(index=index, body={
        "query": {"match": {"sentence": query}},
        "size": size
    })
    return [
        {
            "_id": doc["_id"],
            "_score": doc["_score"],
            "sentence": doc["_source"].get("sentence", ""),
            "chunk_with_context": doc["_source"].get("chunk_with_context", ""),
            "factor": doc["_source"].get("factor", ""),
            "impact": doc["_source"].get("impact", "")
        }
        for doc in response["hits"]["hits"]
    ]


def combine_results(knn_results, bm25_results, knn_weight=0.8, bm25_weight=0.2):
    """kNN과 BM25 결과 결합 및 정렬, 중복 제거"""
    seen = set()  # 이미 본 문장의 ID를 기록
    unique_results = []
    
    # knn_results + bm25_results 합치기
    for result in knn_results + bm25_results:
        doc_id = result["_id"]
        
        # 이미 본 문장이면 건너뛰기
        if doc_id in seen:
            continue
        
        seen.add(doc_id)
        weight = knn_weight if result in knn_results else bm25_weight
        result["combined_score"] = result["_score"] * weight  # 점수 계산
        
        unique_results.append(result)
    
    # 결과를 combined_score 기준으로 정렬
    return sorted(unique_results, key=lambda x: x["combined_score"], reverse=True)


def normalize_score(score, impact):
    """유사도 점수 정규화"""
    if impact == "긍정":
        return 50 + (score * 50)
    elif impact == "부정":
        return 50 - (score * 50)
    return 50


def calculate_factors(results, user_response, embedder, top_k=5):
    """요인별 평균 점수 계산"""
    factors = {}
    response_embedding = embedder.encode(user_response)

    for result in results[:top_k]:
        factor = result.get("factor", "")
        impact = result.get("impact", "")
        doc_embedding = embedder.encode(result["chunk_with_context"])
        score = cosine_similarity([response_embedding], [doc_embedding])[0][0]
        normalized_score = normalize_score(score, impact)

        if factor not in factors:
            factors[factor] = []
        factors[factor].append(normalized_score)

    # 'float32' 값을 'float'로 변환 후 소수 둘째 자리로 반올림
    return {factor: round(float(np.mean(scores)), 2) for factor, scores in factors.items()}


def process_user_responses(es: Elasticsearch, index: str, user_responses: List[str], output_file: str):
    """사용자 응답 처리 및 결과 저장"""
    embedder = BGEM3Embedder()
    reranker = BGEReranker()
    final_results = []
    overall_factors = {}  # 사용자 전체 Big Five 점수를 저장

    for response in tqdm(user_responses, desc="Processing responses"):
        # 사용자 쿼리 임베딩 생성
        query_embedding = embedder.encode(response)

        # kNN 및 BM25 결과 가져오기
        knn_results = fetch_knn_results(es, index, query_embedding, size=50)
        bm25_results = fetch_bm25_results(es, index, response, size=50)

        # 결과 결합
        combined_results = combine_results(knn_results, bm25_results)

        # 상위 문장 리랭킹
        reranked = reranker.rank(response, combined_results)
        reranked.sort(key=lambda x: x["score"], reverse=True)

        # 상위 문장 데이터 포맷팅
        top_sentences = []
        for doc in reranked[:5]:
            matched_doc = next((item for item in combined_results if item["chunk_with_context"] == doc["text"]), {})
            if matched_doc:
                factor = matched_doc.get("factor", "")
                impact = matched_doc.get("impact", "")
                combined_score = doc["score"]

                # 정규화 점수 계산
                normalized_score = normalize_score(combined_score, impact)

                top_sentences.append({
                    "원문": matched_doc.get("sentence", ""),  # 정상적으로 sentence 필드를 가져옴
                    "맥락": matched_doc.get("chunk_with_context", ""),  # 문맥
                    "성격 요인": factor,  # 요인
                    "영향": impact,  # 긍정/부정
                    "유사도 점수": round(combined_score, 2),  # 유사도 점수
                    "정규화된 유사도 점수": round(normalized_score, 2)  # 정규화된 점수
                })

        # 요인별 점수 계산
        factor_details = calculate_factors(combined_results, response, embedder)

        # 요인별 평균 점수 계산 (상위 문장 기반)
        factor_averages = factor_details

        # 결과 저장
        final_results.append({
            "사용자 응답": response,
            "상위 문장": top_sentences,
            "요인별 평균 점수": factor_averages
        })

        # 전체 요인 점수 합산
        for factor, avg_score in factor_averages.items():
            if factor not in overall_factors:
                overall_factors[factor] = []
            overall_factors[factor].append(avg_score)

    # 최종 사용자 요인별 점수 계산 (전체 응답 기반)
    overall_factors_averages = {factor: round(np.mean(scores), 2) for factor, scores in overall_factors.items()}

    # 결과를 JSON 파일로 저장
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"results": final_results, "overall_factors": overall_factors_averages}, f, ensure_ascii=False, indent=4)

    print(f"결과가 {output_file}에 저장되었습니다.")

def interactive_process():
    """사용자 입력을 받아 결과를 즉시 출력하는 Interactive 모드"""
    es = Elasticsearch(
        cloud_id=os.getenv("ELASTICSEARCH_CLOUD_ID"),
        basic_auth=(os.getenv("ELASTICSEARCH_USERNAME"), os.getenv("ELASTICSEARCH_PASSWORD"))
    )
    print("Interactive 모드 시작 (종료하려면 'exit' 입력).")
    
    user_responses = []  # 사용자 응답을 저장할 리스트
    final_results = []  # 최종 결과 저장 리스트

    while True:
        user_input = input("사용자 입력: ").strip()  # 사용자 입력 받기
        if user_input.lower() == "exit":  # 종료 명령어 처리
            break
        
        user_responses.append(user_input)  # 응답 리스트에 추가
        
        # 사용자 응답에 대한 처리
        process_user_responses(es, "cornsoup_vector_cloud", [user_input], "temp_result.json")
        
        # 결과 파일을 읽어와서 출력하기
        with open("temp_result.json", "r", encoding="utf-8") as f:
            result_data = json.load(f)
        
        # 각 응답에 대한 결과 출력
        print(f"\n{'='*50}")
        print(f"응답: {user_input}\n")
        
        print("상위 문장:")
        for idx, sentence_data in enumerate(result_data['results'][0]['상위 문장'], start=1):  # 수정: 'responses' -> 'results'
            print(f"  {idx}. 문장: {sentence_data['원문']}")
            print(f"     - 문맥: {sentence_data['맥락']}")
            print(f"     - 요인: {sentence_data['성격 요인']}")
            print(f"     - 영향: {sentence_data['영향']}")
            print(f"     - 유사도 점수: {sentence_data['유사도 점수']:.2f}")
            print(f"     - 정규화된 점수: {sentence_data['정규화된 유사도 점수']:.2f}")
            print("-" * 50)
        
        print("요인별 평균 점수:")
        for factor, score in result_data['results'][0]['요인별 평균 점수'].items():
            print(f"  {factor}: {score:.2f}")
        
        print(f"{'='*50}\n")
        
        # 각 응답에 대한 최종 결과 리스트에 저장
        final_results.append(result_data['results'][0])

    # 종료 시, 최종 결과 저장
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({
            "responses": final_results,
            "사용자 Big5 점수": result_data.get("overall_factors", {})  # 마지막 사용자 Big5 점수 저장
        }, f, ensure_ascii=False, indent=4)

    print("\nInteractive 결과가 'results.json'에 저장되었습니다.")


def batch_process(input_file, output_file):
    """Batch 모드: 대화 기록 파일에서 데이터를 읽어 결과를 분석 및 저장"""
    es = Elasticsearch(
        cloud_id=os.getenv("ELASTICSEARCH_CLOUD_ID"),
        basic_auth=(os.getenv("ELASTICSEARCH_USERNAME"), os.getenv("ELASTICSEARCH_PASSWORD"))
    )
    with open(input_file, "r", encoding="utf-8") as f:
        chat_data = json.load(f)
        user_responses = [entry["사용자"] for entry in chat_data.get("conversation", [])]

    if user_responses:
        process_user_responses(es, "cornsoup_vector_cloud", user_responses, output_file)
        print(f"Batch 결과가 '{output_file}'에 저장되었습니다.")


if __name__ == "__main__":
    # 기본 main 함수로 실행될 때의 동작
    es = Elasticsearch(
        cloud_id=os.getenv("ELASTICSEARCH_CLOUD_ID"),
        basic_auth=(os.getenv("ELASTICSEARCH_USERNAME"), os.getenv("ELASTICSEARCH_PASSWORD"))
    )
    user_responses = [
        "나는 요즘 새로운 일을 시작했는데 기대가 되면서도 좀 불안해.",
        "사람들과 어울리기보다 혼자 책을 읽는 게 좋아.",
        "가끔은 새로운 경험을 해보고 싶어서 도전을 하지만 무서울 때도 있어."
    ]
    process_user_responses(es, "cornsoup_vector_cloud", user_responses, "result.json")