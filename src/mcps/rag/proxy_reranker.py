import requests
from lancedb.rerankers import Reranker
import pyarrow as pa

class LiteLLMProxyReranker(Reranker):
    def __init__(self, model_name: str, api_key: str, proxy_url: str):
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key
        self.proxy_url = proxy_url

    def rerank_hybrid(self, query: str, vector_results: pa.Table, fts_results: pa.Table) -> pa.Table:
        # 1. Merge the structural outputs from vector and keyword search
        combined_table = self.merge_results(vector_results, fts_results)
        df = combined_table.to_pandas()
        
        if df.empty:
            return combined_table

        # 2. Prepare payload matching LiteLLM's expected JSON format
        documents = df["text"].tolist()  # Swap "text" out for your primary data column
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 3. Request rescoring from LiteLLM proxy
        response = requests.post(f"{self.proxy_url}/v1/rerank", json=payload, headers=headers)
        response.raise_for_status()
        rerank_results = response.json().get("results", [])

        # 4. Map the newly computed relevance scores back to LanceDB rows
        scores = [0.0] * len(df)
        for item in rerank_results:
            scores[item["index"]] = item["relevance_score"] # LiteLLM standardized index format

        df["_relevance_score"] = scores
        df = df.sort_values(by="_relevance_score", ascending=False)
        
        return pa.Table.from_pandas(df)
