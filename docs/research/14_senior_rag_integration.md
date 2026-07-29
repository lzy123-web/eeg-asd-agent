# Senior RAG Minimum Integration

**Date:** 2026-07-29  
**Result:** live runtime NOT RUNNABLE; same-interface offline local index integrated.

## Located assets

- Graph query entry:
  `KG/knowledgegraph_rag/graph_recommend.py::query_graph`
- Recommendation graph readers:
  `recommendation/AbilityKG_Recommendation.py::retrieve_ability_nodes`,
  `retrieve_training_tasks`, `retrieve_experts`, `retrieve_institutions`
- Individual graph retrieval:
  `recommendation/AbilityKG_indivisualize.py::retrieve_graph_knowledge`
- Graph backend: local Neo4j over Bolt.
- Generation backend: an external Spark chat API.
- Exported KG documents:
  `entity/evidence/ability_nodes.csv` and related relation tables.
- Exported literature:
  `entity/paper/paper.csv` and `paper_processed_unit.csv`.

No reusable vector index or embedding-backed retrieval entry was found in the runtime path. The
operational retrieval code is Neo4j/Cypher lookup followed by external generation.

## Actual live probe

Commands:

```bash
.venv/bin/python -c "import neo4j"
nc -z -w 2 127.0.0.1 7687
```

Exact blocking conditions:

1. the project environment raises `ModuleNotFoundError: No module named 'neo4j'`;
2. `127.0.0.1:7687` is closed, so no local Neo4j service is available;
3. the source contains hard-coded credentials and Windows-only data paths; they were neither copied
   nor used;
4. the generation path depends on an external service and is outside the closed offline workflow.

The senior source was kept read-only. Installing a driver would not solve the absent database, so no
dependency or service was added merely to make the import pass.

## Integrated fallback

`SeniorKnowledgeAdapter.retrieve(query, limit)` returns `StructuredKnowledgeResult`:

- `query`
- `items`
  - `title`
  - `source`
  - `chunk`
  - `score`
  - `metadata`
- `mode`

The local index reads the senior exports, performs deterministic lexical ranking, retains source-file
metadata, and always reports `mode=offline_local_index`. The core tool then normalizes this into the
workflow-specific knowledge allowlist. This implementation is an offline adapter over senior assets;
it is not described as successful live RAG integration.

## Validation

- unit test for structured results and source metadata;
- no-result degradation test;
- core Agent E2E through `retrieve_asd_knowledge`;
- nested allowlist test ensuring Adapter metadata and undeclared fields do not enter report Context.
