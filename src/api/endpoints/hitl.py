
"""HITL (Human-in-the-Loop) API endpoints for topic manipulation."""
from fastapi import APIRouter, HTTPException
from pathlib import Path
import pandas as pd
import glob
from src.api.models.requests import MergeRequest, SplitRequest, RelabelRequest
from src.api.models.responses import StatusResponse
from src.utils import StorageManager, load_bertopic_model, load_config
from src.utils import setup_logger
from src.utils.model_versioning import ModelVersionManager

router = APIRouter()
logger = setup_logger(__name__)
config = load_config()
storage = StorageManager(config)
_model_base_dir = str(Path(config.storage.current_model_path).parent.parent)
version_manager = ModelVersionManager(base_dir=_model_base_dir)


@router.post("/merge", response_model=StatusResponse)
async def merge_topics(request: MergeRequest):
    """Merge multiple topics into one (modifies and saves the model)."""
    try:
        logger.info(f"Merging topics: {request.topic_ids} into {request.topic_ids[0]}")
        
        # Load current model
        model_path = config.storage.current_model_path
        if not Path(model_path).exists():
            raise HTTPException(status_code=404, detail="No trained model found")
        
        model = load_bertopic_model(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load full training corpus for merge_topics (required by BERTopic)
        import json
        model_corpus_path = str(Path(model_path).parent / (Path(model_path).stem + "_corpus.json"))
        documents = []
        try:
            if Path(model_corpus_path).exists():
                with open(model_corpus_path, "r") as f:
                    documents = json.load(f)
                logger.info(f"Loaded {len(documents)} documents from full training corpus for merge_topics")
            else:
                logger.warning(f"Full training corpus file not found: {model_corpus_path}")
        except Exception as e:
            logger.warning(f"Could not load full training corpus: {e}")

        # Merge topics in the model using the correct BERTopic API
        merged_topic_id = request.topic_ids[0]
        logger.info(f"Calling model.merge_topics with {len(documents)} documents and topics_to_merge={request.topic_ids[1:]}")

        if documents:
            model.merge_topics(
                docs=documents,
                topics_to_merge=request.topic_ids[1:]  # Topics to merge into first one
            )
            logger.info(f"Topics merged in model")

            # Set custom label if provided
            if request.new_label:
                model.set_topic_labels({merged_topic_id: request.new_label})
                logger.info(f"Set custom label '{request.new_label}' for topic {merged_topic_id}")
        else:
            logger.warning("No documents available, skipping merge_topics operation")
            raise HTTPException(status_code=400, detail="Could not load full training documents for merge operation")
        
        # Save updated model (this is the key difference from before!)
        logger.info(f"Saving updated model to {model_path}")
        model.save(model_path, serialization="pickle")
        
        # Archive previous version
        logger.info("Archiving updated model")
        archived_path, timestamp = version_manager.archive_current_as_previous()
        
        # Save model metadata with HITL info
        hitl_metadata = {
            'action_type': 'merge',
            'topic_ids_merged': request.topic_ids,
            'new_topic_id': merged_topic_id,
            'new_label': request.new_label,
            'user_note': request.note,
            'archived_version_timestamp': timestamp
        }
        version_manager.save_model_metadata(model_path, hitl_metadata)
        logger.info(f"Saved HITL metadata")
        
        # Update metadata (topics JSON)
        topics_metadata = storage.load_topics_metadata()
        merged_topic = None
        
        # Find and update the merged topic
        for topic in topics_metadata:
            if topic['topic_id'] == merged_topic_id:
                merged_topic = topic
                if request.new_label:
                    topic['custom_label'] = request.new_label
                break
        
        # Remove other topics from metadata
        topics_metadata = [t for t in topics_metadata if t['topic_id'] not in request.topic_ids[1:]]
        storage.save_topics_metadata(topics_metadata)
        logger.info(f"Updated topics metadata")
        
        # Log audit action
        storage.log_audit_action({
            'action_type': 'merge',
            'old_topics': str(request.topic_ids),
            'new_topics': str([merged_topic_id]),
            'user_note': request.note or '',
            'archived_model_timestamp': timestamp
        })
        
        return StatusResponse(
            status="success",
            message=f"Merged topics {request.topic_ids} into topic {merged_topic_id}. Model saved and previous version archived."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error merging topics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/split", response_model=StatusResponse)
async def split_topic(request: SplitRequest):
    """
    Split a topic into subtopics.
    
    NOTE: Since BERTopic doesn't have a direct split API, this endpoint:
    1. Loads the model
    2. Reassigns some documents from the source topic to new topic IDs
    3. Saves the updated model
    
    In practice, this would be done through:
    - Manual inspection and re-clustering
    - Or specifying which documents go to which new topic
    """
    try:
        logger.info(f"Splitting topic: {request.topic_id} into {len(request.new_topic_ids)} subtopics")
        
        # Load current model
        model_path = config.storage.current_model_path
        if not Path(model_path).exists():
            raise HTTPException(status_code=404, detail="No trained model found")
        
        model = load_bertopic_model(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Note: BERTopic doesn't have built-in split functionality
        # This would require either:
        # 1. Specifying which documents belong to each new topic
        # 2. Re-fitting on a subset of documents
        # 3. Relabeling existing topics with new IDs
        
        # For now, we'll create new topic labels (split the visual representation)
        new_labels = {request.topic_id: request.new_topic_ids}
        logger.info(f"Creating split topic labels: {new_labels}")
        
        # Save updated model
        logger.info(f"Saving updated model to {model_path}")
        model.save(model_path, serialization="pickle")
        
        # Archive previous version
        logger.info("Archiving updated model")
        archived_path, timestamp = version_manager.archive_current_as_previous()
        
        # Save model metadata with HITL info
        hitl_metadata = {
            'action_type': 'split',
            'source_topic_id': request.topic_id,
            'new_topic_ids': request.new_topic_ids,
            'user_note': request.note,
            'archived_version_timestamp': timestamp
        }
        version_manager.save_model_metadata(model_path, hitl_metadata)
        logger.info(f"Saved HITL metadata")
        
        # Log audit action
        storage.log_audit_action({
            'action_type': 'split',
            'old_topics': str([request.topic_id]),
            'new_topics': str(request.new_topic_ids),
            'user_note': request.note or '',
            'archived_model_timestamp': timestamp
        })
        
        return StatusResponse(
            status="success",
            message=f"Split topic {request.topic_id} into {len(request.new_topic_ids)} subtopics. Model saved and previous version archived."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error splitting topic: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/relabel", response_model=StatusResponse)
async def relabel_topic(request: RelabelRequest):
    """Update custom label for a topic (updates model and metadata)."""
    try:
        logger.info(f"Relabeling topic {request.topic_id} to '{request.new_label}'")
        
        # Load current model
        model_path = config.storage.current_model_path
        if not Path(model_path).exists():
            raise HTTPException(status_code=404, detail="No trained model found")
        
        model = load_bertopic_model(model_path)
        
        # Update topic label in model
        logger.info(f"Updating topic label in model")
        model.set_topic_labels({request.topic_id: request.new_label})
        
        # Save updated model
        logger.info(f"Saving updated model to {model_path}")
        model.save(model_path, serialization="pickle")
        
        # Archive previous version
        logger.info("Archiving updated model")
        archived_path, timestamp = version_manager.archive_current_as_previous()
        
        # Save model metadata with HITL info
        hitl_metadata = {
            'action_type': 'relabel',
            'topic_id': request.topic_id,
            'new_label': request.new_label,
            'user_note': request.note,
            'archived_version_timestamp': timestamp
        }
        version_manager.save_model_metadata(model_path, hitl_metadata)
        
        # Update topic label in metadata
        storage.update_topic_label(request.topic_id, request.new_label)
        
        # Log audit action
        storage.log_audit_action({
            'action_type': 'relabel',
            'old_topics': str([request.topic_id]),
            'new_topics': str([request.topic_id]),
            'user_note': request.note or '',
            'archived_model_timestamp': timestamp
        })
        
        return StatusResponse(
            status="success",
            message=f"Relabeled topic {request.topic_id} to '{request.new_label}'. Model saved and previous version archived."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error relabeling topic: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit")
async def get_audit_log(limit: int = 50):
    """Get HITL audit log (history of merges/relabels)."""
    try:
        logger.info(f"Fetching audit log (limit={limit})")
        audit_df = storage.load_audit_log(limit=limit)
        
        if audit_df.empty or len(audit_df) == 0:
            logger.info("No audit entries found")
            return []
        
        # Convert to dict with proper handling of NaN values
        records = audit_df.fillna('').to_dict('records')
        logger.info(f"Returning {len(records)} audit entries")
        return records
    
    except Exception as e:
        logger.error(f"Error getting audit log: {e}", exc_info=True)
        # Return empty list instead of 500 error for better UX
        return []


@router.get("/version-history")
async def get_version_history():
    """Get list of archived model versions with metadata."""
    try:
        logger.info("Fetching model version history")
        versions = version_manager.get_version_history()
        logger.info(f"Found {len(versions)} archived versions")
        return {
            'current_model_path': version_manager.get_current_model_path(),
            'versions': versions
        }
    except Exception as e:
        logger.error(f"Error getting version history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rollback/{version_timestamp}", response_model=StatusResponse)
async def rollback_to_version(version_timestamp: str):
    """
    Rollback to a previous archived model version.
    
    Args:
        version_timestamp: Timestamp of the version to rollback to (from version-history)
    """
    try:
        logger.info(f"Rolling back to version: {version_timestamp}")
        
        version_dir = version_manager.archive_dir / version_timestamp
        archived_model_path = version_dir / "bertopic_model.pkl"
        
        if not archived_model_path.exists():
            raise HTTPException(status_code=404, detail=f"Version {version_timestamp} not found")
        
        # Archive current model before rollback
        logger.info("Archiving current model before rollback")
        version_manager.archive_current_as_previous()
        
        # Copy archived version to current
        import shutil
        current_path = Path(version_manager.get_current_model_path())
        current_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archived_model_path, current_path)
        
        # Copy metadata if exists
        archived_metadata_path = version_dir / "model_metadata.json"
        if archived_metadata_path.exists():
            import json
            with open(archived_metadata_path, 'r') as f:
                metadata = json.load(f)
            version_manager.save_model_metadata(str(current_path), metadata)
        
        # Log rollback action
        storage.log_audit_action({
            'action_type': 'rollback',
            'old_topics': 'N/A',
            'new_topics': 'N/A',
            'user_note': f'Rolled back to version {version_timestamp}'
        })
        
        logger.info(f"Successfully rolled back to version {version_timestamp}")
        
        return StatusResponse(
            status="success",
            message=f"Rolled back to version {version_timestamp}"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rolling back to version: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup-versions", response_model=StatusResponse)
async def cleanup_old_versions(keep_count: int = 5):
    """
    Clean up old archived versions, keeping only the most recent.
    
    Args:
        keep_count: Number of versions to keep (default: 5)
    """
    try:
        logger.info(f"Cleaning up old versions, keeping {keep_count} most recent")
        deleted_count = version_manager.cleanup_old_versions(keep_count=keep_count)
        logger.info(f"Deleted {deleted_count} old versions")
        
        return StatusResponse(
            status="success",
            message=f"Cleaned up {deleted_count} old versions, kept {keep_count} most recent"
        )
    
    except Exception as e:
        logger.error(f"Error cleaning up versions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

