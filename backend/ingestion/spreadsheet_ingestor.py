import os
from typing import List, Dict, Any, Tuple
import pandas as pd
from backend.ingestion.base import BaseIngestor
from backend.models import DocumentChunk, ChunkMetadata

class SpreadsheetIngestor(BaseIngestor):
    """Parses CSV and Excel files into structured tabular chunks with schema context."""

    def _df_to_markdown(self, df: pd.DataFrame) -> str:
        """Convert dataframe to clean markdown table safely."""
        try:
            return df.to_markdown(index=False)
        except Exception:
            # Fallback pure-python table generator
            headers = [str(c) for c in df.columns]
            rows = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
            for _, r in df.iterrows():
                row_vals = [str(v).replace("\n", " ") for v in r.values]
                rows.append(f"| {' | '.join(row_vals)} |")
            return "\n".join(rows)

    def ingest(self, file_path: str, filename: str) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        clean_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1].lower()
        chunks: List[DocumentChunk] = []

        sheets_dict = {}
        if ext == ".csv":
            df = pd.read_csv(file_path)
            sheets_dict["Sheet1"] = df
        elif ext in [".xlsx", ".xls"]:
            sheets_dict = pd.read_excel(file_path, sheet_name=None)
        else:
            raise ValueError(f"Unsupported spreadsheet format: {ext}")

        for sheet_name, df in sheets_dict.items():
            if df.empty:
                continue

            total_rows = len(df)
            cols = list(df.columns)
            
            chunk_size = 30
            for start_idx in range(0, total_rows, chunk_size):
                sub_df = df.iloc[start_idx:start_idx + chunk_size]
                md_table = self._df_to_markdown(sub_df)
                
                chunk_id = f"sheet_{clean_name}_{sheet_name}_r{start_idx+1}"
                header_context = f"[SPREADSHEET TABLE: {filename} | Sheet: {sheet_name} | Rows {start_idx+1}-{min(start_idx+chunk_size, total_rows)} of {total_rows}]\n"
                stats_summary = f"Columns: {', '.join([str(c) for c in cols])}\n"
                
                full_content = f"{header_context}{stats_summary}\n{md_table}"
                
                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    source_file=filename,
                    modality="table",
                    page_number=1,
                    table_html=md_table,
                    extra_metadata={
                        "sheet_name": sheet_name,
                        "row_start": start_idx + 1,
                        "row_end": min(start_idx + chunk_size, total_rows),
                        "total_rows": total_rows,
                        "columns": cols
                    }
                )
                
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=full_content,
                    metadata=meta
                ))

        return chunks, {
            "total_pages": len(sheets_dict),
            "figures_extracted": 0,
            "tables_extracted": len(chunks),
            "total_chunks": len(chunks)
        }
