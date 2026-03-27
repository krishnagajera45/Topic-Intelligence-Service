"""Master Prefect flow orchestrating the complete pipeline."""
from prefect import flow, get_run_logger
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
import json
import time
import mlflow
import pandas as pd
from src.etl.flows.data_ingestion import data_ingestion_flow
from src.etl.flows.bertopic_modeling import bertopic_modeling_flow
from src.etl.flows.lda_comparison import lda_comparison_flow
from src.etl.flows.nmf_comparison import nmf_comparison_flow
from src.etl.flows.drift_detection import drift_detection_flow
from src.utils import load_config, generate_batch_id, MLflowLogger, get_prefect_context
from src.utils import StorageManager
from src.dashboard.utils.api_client import APIClient


@flow(name="complete-pipeline-flow", log_prints=True)
def complete_pipeline_flow(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Master flow orchestrating the complete topic modeling pipeline.
    
    This flow automatically:
    1. Data Ingestion (ETL)
    2. Model Training (auto-detects seed vs online update)
    3. Drift Detection (if previous model exists)
    4. State Management
    
    Args:
        start_date: Start date for data window (YYYY-MM-DD)
        end_date: End date for data window (YYYY-MM-DD)
        
    Returns:
        Dictionary with pipeline results
    """
    logger = get_run_logger()
    
    config = load_config()
    
    logger.info("=" * 80)
    logger.info(f"Starting complete pipeline flow at {datetime.now()}")
    logger.info(f"Active dataset: {config.active_dataset}")
    logger.info("=" * 80)
    
    # Start timing
    pipeline_start_time = time.time()
    storage = StorageManager(config)
    
    # Get Prefect context for MLflow linking
    prefect_ctx = get_prefect_context()
    logger.info(f"Prefect Flow Run: {prefect_ctx.get('flow_run_name', 'N/A')}")
    logger.info(f"Prefect Flow Run ID: {prefect_ctx.get('flow_run_id', 'N/A')}")
    if prefect_ctx.get('flow_run_url'):
        logger.info(f"🔗 Prefect UI: {prefect_ctx['flow_run_url']}")
    
    # Determine window dates
    if start_date is None or end_date is None:
        state = storage.load_processing_state()
        last_processed = state.get('last_processed_timestamp')
        
        if last_processed:
            start_dt = datetime.fromisoformat(last_processed)
        else:
            start_dt = datetime.fromisoformat(config.dataset.default_start)
        
        window_minutes = getattr(config.scheduler, "window_minutes", None)
        if window_minutes:
            end_dt = start_dt + timedelta(minutes=window_minutes)
        else:
            end_dt = start_dt + timedelta(days=config.scheduler.window_days)
        
        start_date = start_dt.strftime('%Y-%m-%d %H:%M:%S')
        end_date = end_dt.strftime('%Y-%m-%d %H:%M:%S')
    
    batch_id = (
        f"batch_{datetime.fromisoformat(start_date).strftime('%Y%m%d_%H%M')}"
        f"_to_{datetime.fromisoformat(end_date).strftime('%Y%m%d_%H%M')}"
    )
    
    logger.info(f"Processing window: {start_date} to {end_date}")
    logger.info(f"Batch ID: {batch_id}")
    
    # Initialize MLflow logger
    mlflow_logger = MLflowLogger(
        tracking_uri=config.mlflow.tracking_uri,
        experiment_name=config.mlflow.experiment_name
    )
    
    try:
        # Start MLflow run with Prefect context
        mlflow_run = mlflow_logger.start_run_with_prefect_context(
            batch_id=batch_id,
            prefect_flow_run_id=prefect_ctx.get('flow_run_id'),
            prefect_flow_run_name=prefect_ctx.get('flow_run_name'),
            prefect_flow_run_url=prefect_ctx.get('flow_run_url')
        )
        
        # Log system info
        mlflow_logger.log_system_info()
        
        # ========== STEP 1: DATA INGESTION ==========
        logger.info("Step 1: Running data ingestion flow")
        step1_start = time.time()
        
        parquet_path = f"{config.data.processed_parquet_dir}{batch_id}.parquet"
        
        df = data_ingestion_flow(
            csv_path=config.dataset.raw_csv_path,
            output_parquet=parquet_path,
            start_date=start_date,
            end_date=end_date,
            min_docs=5  # Reduced for small datasets
        )
        
        documents = df['text_cleaned'].tolist()
        doc_ids = df['doc_id'].tolist()
        logger.info(f"Data ingestion complete: {len(documents)} documents")
        
        # Log step 1 timing and batch statistics
        step1_duration = time.time() - step1_start
        mlflow_logger.log_processing_time("data_ingestion", step1_duration)
        mlflow_logger.log_batch_statistics(
            documents=documents,
            batch_id=batch_id,
            window_start=start_date,
            window_end=end_date,
            df=df
        )
        
        # ========== STEP 2: BERTOPIC MODELING (Training + Metrics) ==========
        logger.info("Step 2: Running BERTopic modeling flow (training + evaluation metrics)")
        step2_start = time.time()
        
        topics, probs = bertopic_modeling_flow(
            documents=documents,
            doc_ids=doc_ids,
            batch_id=batch_id,
            window_start=start_date,
            window_end=end_date
        )
        
        step2_duration = time.time() - step2_start
        logger.info(f"BERTopic modeling complete: {len(set(topics))} topics")
        
        # Log model details to MLflow
        model_stage = 'initial' if not Path(config.storage.previous_model_path).exists() else 'online_update'
        from src.utils import load_bertopic_model
        model = load_bertopic_model(config.storage.current_model_path)
        mlflow_logger.log_processing_time("bertopic_modeling", step2_duration)
        mlflow_logger.log_model_details(
            model=model,
            topics=topics,
            probs=probs,
            model_config={
                "embedding_model": config.model.embedding_model,
                "min_cluster_size": config.model.min_cluster_size,
                "min_samples": config.model.min_samples,
                "n_neighbors": config.model.umap_n_neighbors,
                "n_components": config.model.umap_n_components,
                "top_n_words": config.model.top_n_words
            },
            is_initial=(model_stage == 'initial')
        )
        mlflow_logger.log_model_artifact(config.storage.current_model_path)
        
        # Log BERTopic metrics to MLflow (from saved file)
        try:
            metrics_path = Path(config.storage.metrics_dir) / "bertopic_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    bt_data = json.load(f)
                latest = bt_data.get("latest", bt_data)
                if latest.get("coherence_c_v") is not None:
                    mlflow_logger.log_metrics({
                        'bertopic_coherence_c_v': latest.get('coherence_c_v', 0),
                        'bertopic_silhouette_score': latest.get('silhouette_score', 0)
                    })
        except Exception:
            pass
        
        # ========== STEP 3: LDA MODELING (Optional) ==========
        step3_start = time.time()
        lda_metrics = None
        
        # Check if LDA comparison is enabled in config
        lda_enabled = getattr(config, 'lda', None) and getattr(config.lda, 'enabled', False)
        
        if lda_enabled:
            logger.info("Step 3: Running LDA modeling flow (cumulative corpus for fair benchmarking)")
            
            try:
                # Use cumulative corpus for LDA - same scope as BERTopic's merged model
                corpus_path = Path(config.storage.current_model_path).parent / (
                    Path(config.storage.current_model_path).stem + "_corpus.json"
                )
                lda_documents = documents
                if corpus_path.exists():
                    try:
                        with open(corpus_path, 'r') as f:
                            lda_documents = json.load(f)
                        logger.info(f"LDA using cumulative corpus: {len(lda_documents)} docs (same scope as BERTopic merged model)")
                    except Exception as e:
                        logger.warning(f"Could not load cumulative corpus for LDA: {e}, using current batch only")
                # Use the same number of topics as BERTopic discovered (excluding outlier topic -1)
                bertopic_num_topics = len(set(topics)) - (1 if -1 in topics else 0)
                # Use configured number of topics or BERTopic's count
                lda_num_topics = getattr(config.lda, 'num_topics', None)
                if lda_num_topics is None or lda_num_topics == 'auto':
                    lda_num_topics = max(bertopic_num_topics, 5)  # Minimum 5 topics
                logger.info(f"Training LDA with {lda_num_topics} topics on cumulative corpus (BERTopic: {bertopic_num_topics} excluding outliers)")
                lda_metrics = lda_comparison_flow(
                    documents=lda_documents,
                    num_topics=lda_num_topics,
                    batch_id=batch_id,
                    window_start=start_date,
                    window_end=end_date
                )
                # Log LDA metrics to MLflow
                if lda_metrics.get('status') == 'success':
                    mlflow_logger.log_metrics({
                        'lda_coherence_c_v': lda_metrics.get('coherence_c_v', 0.0),
                        'lda_diversity': lda_metrics.get('diversity', 0.0),
                        'lda_silhouette_score': lda_metrics.get('silhouette_score', 0.0),
                        'lda_num_topics': lda_metrics.get('num_topics', 0),
                        'lda_training_time_seconds': lda_metrics.get('training_time_seconds', 0.0)
                    })
                    logger.info("LDA comparison complete")
                else:
                    logger.warning(f"LDA comparison skipped or failed: {lda_metrics.get('reason', 'unknown')}")
            except Exception as e:
                logger.error(f"LDA comparison failed: {e}", exc_info=True)
                logger.warning("Pipeline will continue without LDA metrics")
                lda_metrics = {'status': 'error', 'error_message': str(e)}
            
            step3_duration = time.time() - step3_start
            mlflow_logger.log_processing_time("lda_modeling", step3_duration)
        else:
            logger.info("Step 3: LDA modeling disabled in config")

        # ========== STEP 3b: NMF MODELING (Optional) ==========
        step3b_start = time.time()
        nmf_metrics = None

        nmf_enabled = getattr(config, 'nmf', None) and getattr(config.nmf, 'enabled', False)

        if nmf_enabled:
            logger.info("Step 3b: Running NMF modeling flow (cumulative corpus — same scope as LDA / BERTopic)")
            try:
                # NMF uses the same cumulative corpus as LDA for a fair three-way comparison
                corpus_path = Path(config.storage.current_model_path).parent / (
                    Path(config.storage.current_model_path).stem + "_corpus.json"
                )
                nmf_documents = documents
                if corpus_path.exists():
                    try:
                        with open(corpus_path) as f:
                            nmf_documents = json.load(f)
                        logger.info(
                            f"NMF using cumulative corpus: {len(nmf_documents)} docs"
                        )
                    except Exception as e:
                        logger.warning(f"Could not load cumulative corpus for NMF: {e} — using current batch")

                # Align topic count with BERTopic (same as LDA)
                nmf_num_topics = getattr(config.nmf, 'num_topics', None)
                if nmf_num_topics is None or nmf_num_topics == 'auto':
                    bertopic_num_topics = len(set(topics)) - (1 if -1 in topics else 0)
                    nmf_num_topics = max(bertopic_num_topics, 5)
                logger.info(
                    f"Training NMF with {nmf_num_topics} topics on cumulative corpus"
                )

                nmf_metrics = nmf_comparison_flow(
                    documents=nmf_documents,
                    num_topics=nmf_num_topics,
                    batch_id=batch_id,
                    window_start=start_date,
                    window_end=end_date,
                )

                if nmf_metrics.get('status') == 'success':
                    mlflow_logger.log_metrics({
                        'nmf_coherence_c_v':        nmf_metrics.get('coherence_c_v', 0.0),
                        'nmf_diversity':             nmf_metrics.get('diversity', 0.0),
                        'nmf_silhouette_score':      nmf_metrics.get('silhouette_score', 0.0),
                        'nmf_num_topics':            nmf_metrics.get('num_topics', 0),
                        'nmf_training_time_seconds': nmf_metrics.get('training_time_seconds', 0.0),
                    })
                    logger.info("NMF comparison complete")
                else:
                    logger.warning(
                        f"NMF comparison skipped or failed: {nmf_metrics.get('reason', 'unknown')}"
                    )
            except Exception as e:
                logger.error(f"NMF comparison failed: {e}", exc_info=True)
                logger.warning("Pipeline will continue without NMF metrics")
                nmf_metrics = {'status': 'error', 'error_message': str(e)}

            step3b_duration = time.time() - step3b_start
            mlflow_logger.log_processing_time("nmf_modeling", step3b_duration)
        else:
            logger.info("Step 3b: NMF modeling disabled in config")

        # ========== STEP 4: DRIFT DETECTION ==========
        step4_start = time.time()
        if Path(config.storage.previous_model_path).exists():
            logger.info("Step 4: Running drift detection flow")
            
            # Load previous batch documents for drift comparison
            previous_docs = []
            try:
                # Try to find and load the previous batch's parquet file
                import glob
                parquet_files = sorted(glob.glob(f"{config.data.processed_parquet_dir}*.parquet"), reverse=True)
                if len(parquet_files) >= 2:
                    # Load second most recent parquet (previous batch)
                    previous_df = pd.read_parquet(parquet_files[1])
                    previous_docs = previous_df['text_cleaned'].tolist() if 'text_cleaned' in previous_df.columns else []
                    logger.info(f"Loaded {len(previous_docs)} previous documents for drift detection")
                else:
                    logger.warning(f"Could not find previous batch parquet file for drift detection")
            except Exception as e:
                logger.warning(f"Error loading previous documents for drift detection: {e}")
            
            drift_metrics = drift_detection_flow(
                current_docs=documents,  # Sample
                previous_docs=previous_docs if previous_docs else [],  # Sample
                window_start=start_date
            )
            
            logger.info("Drift detection complete")
            
            # Log drift metrics
            step4_duration = time.time() - step4_start
            mlflow_logger.log_processing_time("drift_detection", step4_duration)
            mlflow_logger.log_drift_metrics(drift_metrics, start_date)
            
            # Log alerts if any
            alerts = drift_metrics.get('alerts', [])
            mlflow_logger.log_alerts(alerts)
        else:
            logger.info("Step 4: Skipping drift detection (initial run or no previous model)")
            drift_metrics = None
            alerts = []
        
        # ========== STEP 5: UPDATE STATE ==========
        logger.info("Step 5: Updating processing state")
        
        storage.save_processing_state({
            'last_processed_timestamp': end_date,
            'last_batch_id': batch_id,
            'last_run_timestamp': datetime.now().isoformat(),
            'documents_processed': len(documents),
            'num_topics': len(set(topics)),
            'status': 'success'
        })
        
        # Calculate total pipeline duration
        pipeline_duration = time.time() - pipeline_start_time
        
        # Post-pipeline verification: Check corpus and assignments alignment for HITL readiness
        logger.info("=" * 80)
        logger.info("POST-PIPELINE VERIFICATION")
        logger.info("=" * 80)
        try:
            corpus_path = Path(config.storage.current_model_path).parent / (Path(config.storage.current_model_path).stem + "_corpus.json")
            assignments_path = Path(config.storage.doc_assignments_path)
            if corpus_path.exists() and assignments_path.exists():
                with open(corpus_path, 'r') as f:
                    corpus = json.load(f)
                assignments_df = pd.read_csv(assignments_path)
                logger.info(f"Corpus documents:     {len(corpus)}")
                logger.info(f"Assignments rows:     {len(assignments_df)}")
                
                if len(corpus) == len(assignments_df):
                    logger.info("✅ HITL READY: Corpus and assignments are perfectly aligned")
                    logger.info("   Topic merge/split operations will work correctly")
                else:
                    logger.warning(f"⚠️  ALIGNMENT ISSUE: {len(corpus)} corpus vs {len(assignments_df)} assignments")
                    logger.warning("   HITL operations may fail")
            else:
                logger.warning("⚠️  Could not verify alignment (files missing)")
        except Exception as e:
            logger.warning(f"Verification check failed: {e}")
        
        logger.info("=" * 80)
        
        # Log pipeline summary
        mlflow_logger.log_pipeline_summary(
            status='success',
            documents_processed=len(documents),
            num_topics=len(set(topics)),
            drift_detected=drift_metrics is not None and len(drift_metrics.get('alerts', [])) > 0,
            num_alerts=len(alerts) if drift_metrics else 0,
            total_duration_seconds=pipeline_duration
        )
        
        logger.info("=" * 80)
        logger.info("Complete pipeline flow finished successfully!")
        logger.info(f"Total duration: {pipeline_duration:.2f}s ({pipeline_duration/60:.2f}m)")
        logger.info("=" * 80)
        
        # End MLflow run
        mlflow.end_run()
        
        return {
            'status': 'success',
            'batch_id': batch_id,
            'documents_processed': len(documents),
            'num_topics': len(set(topics)),
            'drift_detected': drift_metrics is not None and len(drift_metrics.get('alerts', [])) > 0,
            'lda_comparison': lda_metrics if lda_metrics else None,
            'nmf_comparison': nmf_metrics if nmf_metrics else None,
            'mlflow_run_id': mlflow_run.info.run_id,
            'prefect_run_id': prefect_ctx.get('flow_run_id'),
            'duration_seconds': pipeline_duration
        }
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        
        # Log error to MLflow
        try:
            pipeline_duration = time.time() - pipeline_start_time
            mlflow_logger.log_pipeline_summary(
                status='error',
                documents_processed=0,
                num_topics=0,
                drift_detected=False,
                num_alerts=0,
                total_duration_seconds=pipeline_duration
            )
            mlflow.set_tag("error", str(e))
            mlflow.log_param("error_message", str(e)[:250])  # Truncate if too long
            mlflow.end_run(status="FAILED")
        except Exception as mlflow_error:
            logger.warning(f"Could not log error to MLflow: {mlflow_error}")
        
        # Update state with error (don't update last_processed_timestamp on error)
        storage.save_processing_state({
            'status': 'error',
            'error_message': str(e),
            'error_timestamp': datetime.now().isoformat()
        })
        
        raise


if __name__ == "__main__":
    # Run the complete pipeline
    result = complete_pipeline_flow()
    print(f"Pipeline result: {result}")

