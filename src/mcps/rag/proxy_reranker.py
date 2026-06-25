import pyarrow as pa
import requests
from lancedb.rerankers import Reranker


class ProxyReranker(Reranker):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        proxy_url: str,
        column: str = "content",
    ):
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key
        self.proxy_url = proxy_url
        self.column = column

    def rerank_hybrid(
        self,
        query: str,
        vector_results: pa.Table,
        fts_results: pa.Table,
    ) -> pa.Table:
        # 1. Merge the structural outputs from vector and keyword search
        combined_table = self.merge_results(vector_results, fts_results)
        return self._rerank_results(query, combined_table)

    def rerank_vector(self, query: str, vector_results: pa.Table) -> pa.Table:
        """Rerank vector search results."""
        return self._rerank_results(query, vector_results)

    def rerank_fts(self, query: str, fts_results: pa.Table) -> pa.Table:
        """Rerank full-text search results."""
        return self._rerank_results(query, fts_results)


    def _rerank_results(
        self,
        query: str,
        combined_table: pa.Table,
    ) -> pa.Table:
        if combined_table.num_rows == 0:
            return combined_table.append_column(
                "_relevance_score",
                pa.array([], type=pa.float32()),
            )

        documents = self._table_to_documents(combined_table)
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 3. Request rescoring from LiteLLM proxy
        response = requests.post(
            f"{self.proxy_url}/v1/rerank",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        rerank_results = response.json().get("results", [])

        # 4. Map the newly computed relevance scores back to LanceDB rows
        df = combined_table.to_pandas()
        scores = [0.0] * len(df)
        for item in rerank_results:
            scores[item["index"]] = item["relevance_score"]

        df["_relevance_score"] = scores
        df = df.sort_values(by="_relevance_score", ascending=False)

        return pa.Table.from_pandas(df)

    def _table_to_documents(self, results: pa.Table) -> list[str]:
        """Extract the text column as a list of plain Python strings."""
        text_column = results[self.column]
        return [
            str(text_column[index].as_py())
            for index in range(results.num_rows)
        ]
