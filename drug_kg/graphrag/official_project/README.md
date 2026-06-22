# Official GraphRAG BYOG

This project uses the current medical KG as a Bring-Your-Own-Graph input for official GraphRAG.

## 1. Export the current KG to official parquet tables

```powershell
python drug_kg\graphrag\export_official_byog.py
```

This writes the following files to `drug_kg/graphrag/official_byog/output`:

- `entities.parquet`
- `relationships.parquet`
- `text_units.parquet`
- `documents.parquet`

## 2. Configure model endpoints

Edit `.env` in this directory:

- `GRAPHRAG_COMPLETION_*` for community report generation and query answering
- `GRAPHRAG_EMBEDDING_*` for entity/text/community embeddings

If the root project already has a valid DashScope config, sync it first:

```powershell
python drug_kg\graphrag\prepare_official_env.py
```

## 3. Run official GraphRAG indexing on top of the existing KG

```powershell
graphrag index -r drug_kg\graphrag\official_project
```

This project is configured to run only the workflows needed for BYOG:

- `create_communities`
- `create_community_reports`
- `generate_text_embeddings`

If you want to run the official pipeline in phases, use:

```powershell
python drug_kg\graphrag\run_official_index.py --phase communities
python drug_kg\graphrag\run_official_index.py --phase reports
python drug_kg\graphrag\run_official_index.py --phase embeddings
```

If you want a no-LLM startup version of community reports first, use:

```powershell
python drug_kg\graphrag\bootstrap_community_reports.py
```

## 4. Run official queries

Local search:

```powershell
graphrag query -r drug_kg\graphrag\official_project --method local "华法林和什么药有相互作用"
```

Global search:

```powershell
graphrag query -r drug_kg\graphrag\official_project --method global "这个图谱里最常见的高风险相互作用模式是什么"
```
