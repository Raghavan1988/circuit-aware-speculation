"""In-memory trace records (issue I02/I06 boundary).

The engine emits these dataclasses; the durable Parquet writer and full schema
validation are issue I06 (Step 3). Keeping the records here lets I02 be tested
and reasoned about without the storage layer.
"""
