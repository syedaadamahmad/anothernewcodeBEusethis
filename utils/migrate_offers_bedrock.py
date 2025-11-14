"""
Data Migration Script - ADAPTED FOR YOUR STACK
Uses AWS Bedrock Titan embeddings (1024-dim) instead of sentence-transformers

Run: python scripts/migrate_offers_bedrock.py
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any
import logging
from pymongo import MongoClient
from dotenv import load_dotenv

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class OfferMigration:
    """
    Adapts payment_offers, general_offers, and gift_coupons to unified schema.
    PRESERVES existing collections - only adds offer_type field.
    """
    
    def __init__(self, mongo_uri: str, db_name: str):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        
        # Your existing collections
        self.gift_coupons = self.db.gift_coupons
        self.payment_offers = self.db.payment_offers
        self.general_offers = self.db.general_offers
    
    def add_offer_type_field(self):
        """
        Add offer_type field to each collection WITHOUT creating new collection.
        This ensures your existing RAG system continues to work.
        """
        logger.info("[MIGRATION_START] Adding offer_type fields...")
        
        results = {
            'gift_coupons_updated': 0,
            'payment_offers_updated': 0,
            'general_offers_updated': 0
        }
        
        try:
            # Update gift_coupons
            logger.info("[STEP_1] Updating gift_coupons...")
            result = self.gift_coupons.update_many(
                {'offer_type': {'$exists': False}},
                {'$set': {'offer_type': 'gc'}}
            )
            results['gift_coupons_updated'] = result.modified_count
            logger.info(f"  ✅ Updated {result.modified_count} gift_coupons")
            
            # Update payment_offers
            logger.info("[STEP_2] Updating payment_offers...")
            result = self.payment_offers.update_many(
                {'offer_type': {'$exists': False}},
                {'$set': {'offer_type': 'po'}}
            )
            results['payment_offers_updated'] = result.modified_count
            logger.info(f"  ✅ Updated {result.modified_count} payment_offers")
            
            # Update general_offers
            logger.info("[STEP_3] Updating general_offers...")
            result = self.general_offers.update_many(
                {'offer_type': {'$exists': False}},
                {'$set': {'offer_type': 'go'}}
            )
            results['general_offers_updated'] = result.modified_count
            logger.info(f"  ✅ Updated {result.modified_count} general_offers")
            
            logger.info("[MIGRATION_SUCCESS] ✅ All collections updated")
            return results
            
        except Exception as e:
            logger.error(f"[MIGRATION_FAILED] {str(e)}")
            raise
    
    def verify_migration(self):
        """Verify offer_type field exists in all collections."""
        logger.info("[VERIFICATION] Checking offer_type fields...")
        
        stats = {}
        for name, collection in [
            ('gift_coupons', self.gift_coupons),
            ('payment_offers', self.payment_offers),
            ('general_offers', self.general_offers)
        ]:
            total = collection.count_documents({})
            with_offer_type = collection.count_documents({'offer_type': {'$exists': True}})
            
            stats[name] = {
                'total': total,
                'with_offer_type': with_offer_type,
                'coverage': (with_offer_type / total * 100) if total > 0 else 0
            }
            
            logger.info(f"  {name}: {with_offer_type}/{total} ({stats[name]['coverage']:.1f}%)")
        
        return stats
    
    def close(self):
        """Close MongoDB connection."""
        self.client.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Add offer_type field to existing collections')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing fields')
    args = parser.parse_args()
    
    # Load from your .env
    mongo_uri = os.getenv("MONGO_DB_URI")
    db_name = os.getenv("DB_NAME", "smartbhaiDB")
    
    if not mongo_uri:
        logger.error("MONGO_DB_URI not found in environment")
        sys.exit(1)
    
    migrator = OfferMigration(mongo_uri, db_name)
    
    try:
        if args.verify_only:
            migrator.verify_migration()
        else:
            # Add offer_type fields
            results = migrator.add_offer_type_field()
            
            logger.info("\n=== MIGRATION RESULTS ===")
            logger.info(f"Gift coupons updated: {results['gift_coupons_updated']}")
            logger.info(f"Payment offers updated: {results['payment_offers_updated']}")
            logger.info(f"General offers updated: {results['general_offers_updated']}")
            
            # Verify
            migrator.verify_migration()
    
    finally:
        migrator.close()


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
# python utils/migrate_offers_bedrock.py --verify-only                for verification only