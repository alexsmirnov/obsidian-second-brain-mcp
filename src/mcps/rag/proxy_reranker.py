import logging

import httpx
import pyarrow as pa
from lancedb.rerankers import Reranker, RRFReranker

logger = logging.getLogger(__file__)

FALLBACK = RRFReranker(return_score="relevance")

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
        self.column = column
        self._client = httpx.Client(
            base_url=proxy_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        """Release the pooled HTTP connections owned by this reranker."""
        self._client.close()

    def rerank_hybrid(
        self,
        query: str,
        vector_results: pa.Table,
        fts_results: pa.Table,
    ) -> pa.Table:
        # 1. Merge the structural outputs from vector and keyword search
        combined_table = self.merge_results(vector_results, fts_results)
        try:
            return self._rerank_results(query, combined_table)
        except httpx.HTTPError as error:
            self._log_failure(error)
            return FALLBACK.rerank_hybrid(query, vector_results, fts_results)

    def rerank_vector(self, query: str, vector_results: pa.Table) -> pa.Table:
        """Rerank vector search results."""
        try:
            return self._rerank_results(query, vector_results)
        except httpx.HTTPError as error:
            self._log_failure(error)
            return FALLBACK.rerank_vector(query, vector_results)

    def rerank_fts(self, query: str, fts_results: pa.Table) -> pa.Table:
        """Rerank full-text search results."""
        try:
            return self._rerank_results(query, fts_results)
        except httpx.HTTPError as error:
            self._log_failure(error)
            return FALLBACK.rerank_fts(query, fts_results)


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

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": self._table_to_documents(combined_table),
        }

        # 3. Request rescoring from LiteLLM proxy
        response = self._client.post("/v1/rerank", json=payload)
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

    @staticmethod
    def _log_failure(error: httpx.HTTPError) -> None:
        """Log the proxy failure; HTTP status errors carry the diagnostic body."""
        match error:
            case httpx.HTTPStatusError():
                response = error.response
                logger.warning(
                    "Rerank request failed: status=%s reason=%s body=%s",
                    response.status_code,
                    response.reason_phrase,
                    response.text,
                )
            case _:
                logger.warning("Rerank request could not be sent: %s", error)

    def _table_to_documents(self, results: pa.Table) -> list[str]:
        """Extract the text column as a list of plain Python strings."""
        text_column = results[self.column]
        return [
            str(text_column[index].as_py())
            for index in range(results.num_rows)
        ]
