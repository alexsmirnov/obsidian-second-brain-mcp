# LanceDB Hybrid Search Implementation Plan

This document outlines the implementation plan for enhancing the search functionality in the LanceDB vector store implementation, focusing on hybrid search and vector index optimization.

## 1. Required Imports

```python
from lancedb.index import FTS
from lancedb.rerankers import LinearCombinationReranker
from lancedb.index.vector import IVF_PQ, HNSW  # If using specific index types
```

## 2. Modifying the `search` Method for Hybrid Search

Enhance the existing `search` method to use hybrid search capability by default:

```python
async def search(self, query: str, tags: list[str] = None, file_path: str | None = None, scope: ScopeEnum = ScopeEnum.ALL, limit: int = 5, vector_weight: float = 0.7) -> list[Chunk]:
    """
    Search for chunks using hybrid search (vector similarity and keyword matching).
    
    Args:
        query (str): The search query text.
        tags (list[str], optional): List of tags to filter by (all must be present). Defaults to None.
        file_path (str | None, optional): Substring of source_path to filter results. Defaults to None.
        scope (ScopeEnum, optional): Where to search (CONTENT, TITLE, DESCRIPTION, or ALL). Defaults to ScopeEnum.ALL.
        limit (int, optional): Maximum number of results to return. Defaults to 5.
        vector_weight (float, optional): Weight given to vector search results vs FTS results (0-1). Defaults to 0.7.
    """
    if not self._initialized:
        await self.initialize()
        
    # Calculate embedding for the query
    query_embedding = self.embedding_function.compute_query_embeddings(query)[0]
    
    try:
        # Create reranker with specified weight
        reranker = LinearCombinationReranker(weight=vector_weight)
        
        # Start the search query with hybrid search type
        result = await self.table.search(query_type="hybrid").vector(query_embedding).text(query).rerank(reranker)
        
        # Apply filters based on parameters
        where_conditions = []
        
        # Filter by tags if provided - uses SQL CONTAINS for array containment
        if tags:
            tag_conditions = []
            for tag in tags:
                tag_conditions.append(f"tags CONTAINS '{tag}'")
            if tag_conditions:
                where_conditions.append(f"({' AND '.join(tag_conditions)})")
        
        # Filter by file path if provided - uses LIKE syntax for substring matching
        if file_path:
            where_conditions.append(f"source_path LIKE '%{file_path}%'")
        
        # Apply scope filter
        if scope != ScopeEnum.ALL:
            if scope == ScopeEnum.CONTENT:
                # Default search is in content, no additional filter needed
                pass
            elif scope == ScopeEnum.TITLE:
                where_conditions.append("metadata.title IS NOT NULL")
            elif scope == ScopeEnum.DESCRIPTION:
                where_conditions.append("metadata.description IS NOT NULL")
        
        # Combine all where conditions with AND
        if where_conditions:
            combined_where = " AND ".join(where_conditions)
            result = result.where(combined_where)
        
        # Execute the search and get results
        results = await result.limit(limit).to_list()
        
        # Convert results to Chunk objects
        return [Chunk.model_validate(result) for result in results]
    
    except Exception as e:
        logger.error(f"Failed to perform hybrid search in LanceDB: {e}")
        raise
    
    return []
```

## 3. Enhancing `initialize` for Index Creation

Update the initialization method to create both FTS and vector indices:

```python
async def initialize(self, create_fts_index: bool = True, create_vector_index: bool = True) -> None:
    """
    Initialize the LanceDB vector store.
    
    Args:
        create_fts_index: Whether to create FTS index on 'content' column
        create_vector_index: Whether to create vector index on 'embeddings' column
    """
    # Existing initialization code...
    
    # Create FTS index if requested
    if create_fts_index:
        try:
            await self.create_fts_index()
        except Exception as e:
            logger.warning(f"Failed to create FTS index during initialization: {e}")
            logger.warning("FTS functionality will not be available")
    
    # Create vector index if requested
    if create_vector_index:
        try:
            await self.create_vector_index()
        except Exception as e:
            logger.warning(f"Failed to create vector index during initialization: {e}")
            logger.warning("Vector search performance may be degraded")
```

## 4. Adding Vector Index Method

```python
async def create_vector_index(
    self,
    column: str = "embeddings",
    index_type: str = "IVF_PQ",
    distance_type: str = "cosine",
    replace: bool = True,
    **kwargs
) -> None:
    """
    Create a vector index for fast similarity search.
    
    Args:
        column: Column containing vector embeddings (default: "embeddings")
        index_type: Type of vector index ("IVF_PQ" or "HNSW")
        distance_type: Distance metric ("cosine", "l2", or "dot") 
        replace: Replace existing index if it exists
        **kwargs: Additional parameters for specific index types:
            - For IVF_PQ: num_partitions, num_sub_vectors, num_bits
            - For HNSW: M (max connections), ef_construction
    """
    if not self._initialized:
        await self.initialize(create_vector_index=False)
    
    try:
        # Set default parameters based on index type
        if index_type == "IVF_PQ":
            vector_dim = self.embedding_function.ndims()
            # Default settings based on vector dimension
            defaults = {
                "num_partitions": int(max(np.sqrt(1000), 256)),  # At least 256 partitions
                "num_sub_vectors": max(vector_dim // 16, 1),     # Dimension/16 or at least 1
                "num_bits": 8                                   # 8 bits per sub-vector
            }
            # Update defaults with any provided kwargs
            for key, value in kwargs.items():
                if key in defaults:
                    defaults[key] = value
                
            # Create the index with configured parameters    
            await self.table.create_index(
                column_name=column,
                index_type=index_type,
                distance_type=distance_type,
                num_partitions=defaults["num_partitions"],
                num_sub_vectors=defaults["num_sub_vectors"],
                num_bits=defaults["num_bits"],
                replace=replace
            )
            
        elif index_type == "HNSW":
            # Default settings for HNSW
            defaults = {
                "M": 16,                # Max connections in the graph
                "ef_construction": 100  # Size of dynamic candidate list during construction
            }
            # Update defaults with any provided kwargs
            for key, value in kwargs.items():
                if key in defaults:
                    defaults[key] = value
                    
            # Create the index with configured parameters
            await self.table.create_index(
                column_name=column,
                index_type=index_type,
                distance_type=distance_type,
                M=defaults["M"],
                ef_construction=defaults["ef_construction"],
                replace=replace
            )
            
        else:
            logger.error(f"Unsupported index type: {index_type}")
            raise ValueError(f"Unsupported index type: {index_type}")
            
        logger.info(f"Created vector index of type {index_type} on column '{column}'")
        
    except Exception as e:
        logger.error(f"Failed to create vector index: {e}")
        raise
```

## 5. Tag Filtering Implementation

The current tag filtering approach is not optimal for checking if all provided tags are present in the tags array. The `CONTAINS` operator only checks if the array contains the single value, but we need to check if multiple values are all present.

For a proper implementation in LanceDB's SQL syntax:

```python
# Filter by tags if provided - ensuring ALL tags are present 
if tags:
    tag_conditions = []
    for tag in tags:
        # For each tag, check if it's contained in the array
        tag_conditions.append(f"tags CONTAINS '{tag}'")
    if tag_conditions:
        # Use AND to ensure all tags are present
        where_conditions.append(f"({' AND '.join(tag_conditions)})")
```

This creates SQL like: `(tags CONTAINS 'tag1' AND tags CONTAINS 'tag2' AND tags CONTAINS 'tag3')`.

## References to LanceDB Documentation:

1. **Hybrid Search API**:
   - https://lancedb.github.io/lancedb/hybrid_search/hybrid_search/
   - Example: `table.search(query_type="hybrid").vector(query).text(query).rerank(reranker=reranker)`

2. **Vector Indexing**:
   - https://lancedb.github.io/lancedb/ann_indexes/
   - Supported indices: IVF_PQ, HNSW, IVF_HNSW_SQ
   - Performance comparison: HNSW generally better quality but higher memory usage 

3. **SQL Filtering Syntax**:
   - Array containment: `CONTAINS` operator - checks if a value is in an array
   - String pattern matching: `LIKE` operator with wildcards (%)
   - Documentation: https://lancedb.github.io/lancedb/sql/

4. **Reranking**:
   - https://lancedb.github.io/lancedb/reranking/
   - `LinearCombinationReranker` is built-in and balances vector/text results

5. **Performance Optimization**:
   - Use `nprobes` parameter to control search breadth in IVF indices
   - Use `refine_factor` for better result accuracy
   - Example: `.search(query).nprobes(20).refine_factor(10)`

6. **Async API**:
   - All LanceDB operations have async counterparts
   - Documentation: https://lancedb.github.io/lancedb/python/async_api/

The solution described in this plan combines the power of vector search for semantic similarity with the precision of full-text search, providing more accurate and relevant results while maintaining efficient querying through proper indexing.