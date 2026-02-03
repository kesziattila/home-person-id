import sqlite3
import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def migrate(db_path):
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return

    logger.info(f"Starting migration for {db_path}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 1. Check reid_embeddings table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reid_embeddings'")
        if not cursor.fetchone():
            logger.info("Creating reid_embeddings table...")
            cursor.execute("""
                CREATE TABLE reid_embeddings (
                    id INTEGER PRIMARY KEY,
                    person_id INTEGER,
                    track_id VARCHAR(64),
                    timestamp DATETIME,
                    camera_id VARCHAR(64) NOT NULL,
                    embedding BLOB NOT NULL,
                    quality FLOAT,
                    visibility FLOAT,
                    snapshot_path VARCHAR(512),
                    FOREIGN KEY(person_id) REFERENCES persons(id),
                    FOREIGN KEY(track_id) REFERENCES tracks(id)
                )
            """)
        else:
            logger.info("reid_embeddings table already exists.")

        # 2. Check events table for new columns
        cursor.execute("PRAGMA table_info(events)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'face_embedding_id' not in columns:
            logger.info("Adding face_embedding_id to events table...")
            cursor.execute("ALTER TABLE events ADD COLUMN face_embedding_id INTEGER REFERENCES face_embeddings(id)")
        
        if 'reid_embedding_id' not in columns:
            logger.info("Adding reid_embedding_id to events table...")
            cursor.execute("ALTER TABLE events ADD COLUMN reid_embedding_id INTEGER REFERENCES reid_embeddings(id)")

        # 3. Check tracks table for status column (though it likely exists if recent)
        cursor.execute("PRAGMA table_info(tracks)")
        track_columns = [col[1] for col in cursor.fetchall()]
        if 'status' not in track_columns:
            logger.info("Adding status to tracks table...")
            cursor.execute("ALTER TABLE tracks ADD COLUMN status VARCHAR(32) DEFAULT 'active'")

        conn.commit()
        logger.info("Migration completed successfully.")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate Home Person ID database to the latest schema.")
    parser.add_argument("--db", default="data/database.db", help="Path to the database file")
    args = parser.parse_args()
    
    migrate(args.db)
