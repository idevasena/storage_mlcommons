import argparse
import gc
import logging
import mmap
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, Generator, Tuple

import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Add the parent directory to sys.path to import config_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdbbench.config_loader import load_config, merge_config_with_args
from vdbbench.compact_and_watch import monitor_progress

# Optional psutil for adaptive mode
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Memory Management Utilities
# =============================================================================

def parse_memory_string(mem_str: str) -> int:
    """Parse memory string like '4G', '512M' to bytes."""
    if isinstance(mem_str, (int, float)):
        return int(mem_str)
    if not mem_str:
        return 0
    mem_str = str(mem_str).strip().upper()
    multipliers = {'B': 1, 'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
    if mem_str[-1] in multipliers:
        return int(float(mem_str[:-1]) * multipliers[mem_str[-1]])
    return int(mem_str)


def get_memory_percent() -> float:
    """Get current memory usage percentage."""
    if PSUTIL_AVAILABLE:
        return psutil.virtual_memory().percent
    return 50.0  # Default if psutil not available


def get_available_memory() -> int:
    """Get available memory in bytes."""
    if PSUTIL_AVAILABLE:
        return psutil.virtual_memory().available
    return 8 * 1024**3  # Assume 8GB if psutil not available


class AdaptiveBatchController:
    """
    Adaptive batch size controller based on memory pressure.
    Only active when --adaptive flag is used.
    """
    
    def __init__(self, initial_batch_size: int, 
                 min_batch_size: int = 100,
                 max_batch_size: int = 100000,
                 memory_threshold: float = 80.0):
        self.current_batch_size = initial_batch_size
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.adjustment_count = 0
        self.batches_since_adjustment = 0
    
    def get_batch_size(self) -> int:
        """Get current batch size, adjusting based on memory if needed."""
        if not PSUTIL_AVAILABLE:
            return self.current_batch_size
        
        self.batches_since_adjustment += 1
        mem_percent = get_memory_percent()
        
        # Scale down if memory pressure
        if mem_percent > self.memory_threshold:
            new_size = max(self.min_batch_size, int(self.current_batch_size * 0.5))
            if new_size < self.current_batch_size:
                logger.info(f"[Adaptive] Memory at {mem_percent:.1f}%, reducing batch: "
                            f"{self.current_batch_size} -> {new_size}")
                self.current_batch_size = new_size
                self.adjustment_count += 1
                self.batches_since_adjustment = 0
                gc.collect()
        # Scale up if plenty of headroom
        elif (self.batches_since_adjustment > 10 and 
              mem_percent < self.memory_threshold - 25 and
              self.current_batch_size < self.initial_batch_size):
            new_size = min(self.initial_batch_size, int(self.current_batch_size * 1.25))
            if new_size > self.current_batch_size:
                logger.info(f"[Adaptive] Memory at {mem_percent:.1f}%, increasing batch: "
                            f"{self.current_batch_size} -> {new_size}")
                self.current_batch_size = new_size
                self.adjustment_count += 1
                self.batches_since_adjustment = 0
        
        return self.current_batch_size
    
    def force_scale_down(self):
        """Force scale down after an error."""
        new_size = max(self.min_batch_size, int(self.current_batch_size * 0.5))
        if new_size < self.current_batch_size:
            logger.info(f"[Adaptive] Forcing batch reduction: {self.current_batch_size} -> {new_size}")
            self.current_batch_size = new_size
            self.adjustment_count += 1
            gc.collect()


class DiskBackedBuffer:
    """
    Memory-mapped file buffer for disk-backed vector generation.
    Only used when --disk-backed flag is specified.
    """
    
    def __init__(self, dimension: int, max_vectors: int, 
                 temp_dir: Optional[str] = None):
        self.dimension = dimension
        self.max_vectors = max_vectors
        self.dtype_size = 4  # float32
        self.vector_size = dimension * self.dtype_size
        self.file_size = self.vector_size * max_vectors
        
        # Create temp file
        temp_dir_path = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        temp_dir_path.mkdir(parents=True, exist_ok=True)
        
        self.temp_file = tempfile.NamedTemporaryFile(
            dir=temp_dir_path, prefix='vdbbench_', suffix='.mmap', delete=False
        )
        self.temp_path = Path(self.temp_file.name)
        
        # Pre-allocate
        self.temp_file.seek(self.file_size - 1)
        self.temp_file.write(b'\0')
        self.temp_file.flush()
        
        # Memory map
        self.mmap = mmap.mmap(self.temp_file.fileno(), self.file_size)
        self.vectors_stored = 0
        
        logger.info(f"Created disk buffer: {self.temp_path} ({self.file_size / (1024**3):.2f} GB)")
    
    def write_batch(self, vectors: np.ndarray, start_index: int):
        """Write vectors to disk buffer."""
        start_offset = start_index * self.vector_size
        end_offset = start_offset + len(vectors) * self.vector_size
        self.mmap[start_offset:end_offset] = vectors.astype(np.float32).tobytes()
        self.vectors_stored = max(self.vectors_stored, start_index + len(vectors))
    
    def read_batch(self, start_index: int, count: int) -> np.ndarray:
        """Read vectors from disk buffer."""
        start_offset = start_index * self.vector_size
        end_offset = start_offset + count * self.vector_size
        data = self.mmap[start_offset:end_offset]
        return np.frombuffer(data, dtype=np.float32).reshape(count, self.dimension)
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.mmap.close()
            self.temp_file.close()
            if self.temp_path.exists():
                self.temp_path.unlink()
            logger.info(f"Cleaned up disk buffer: {self.temp_path}")
        except Exception as e:
            logger.warning(f"Error cleaning up disk buffer: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.cleanup()


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Load vectors into Milvus database")
    
    # Connection parameters
    parser.add_argument("--host", type=str, default="localhost", help="Milvus server host")
    parser.add_argument("--port", type=str, default="19530", help="Milvus server port")
    
    # Collection parameters
    parser.add_argument("--collection-name", type=str, help="Name of the collection to create")
    parser.add_argument("--dimension", type=int, help="Vector dimension")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of shards for the collection")
    parser.add_argument("--vector-dtype", type=str, default="float", choices=["FLOAT_VECTOR"],
                        help="Vector data type. Only FLOAT_VECTOR is supported for now")
    parser.add_argument("--force", action="store_true", help="Force recreate collection if it exists")
    
    # Data generation parameters
    parser.add_argument("--num-vectors", type=int, help="Number of vectors to generate")
    parser.add_argument("--distribution", type=str, default="uniform", 
                        choices=["uniform", "normal"], help="Distribution for vector generation")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for insertion")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of vectors to generate in each chunk (for memory management)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible vector generation")

    # Index parameters
    parser.add_argument("--index-type", type=str, default="DISKANN", help="Index type")
    parser.add_argument("--metric-type", type=str, default="COSINE", help="Metric type for index")
    parser.add_argument("--max-degree", type=int, default=16, help="DiskANN MaxDegree parameter")
    parser.add_argument("--search-list-size", type=int, default=200, help="DiskANN SearchListSize parameter")
    parser.add_argument("--M", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--ef-construction", type=int, default=200, help="HNSW efConstruction parameter")
    
    # Memory optimization parameters
    parser.add_argument("--adaptive", action="store_true",
                        help="Enable adaptive batch sizing based on memory pressure")
    parser.add_argument("--memory-budget", type=str, default="0",
                        help="Memory budget (e.g., 4G, 512M). Default: auto")
    parser.add_argument("--disk-backed", action="store_true",
                        help="Use disk-backed buffer for memory-constrained systems")
    parser.add_argument("--temp-dir", type=str,
                        help="Temp directory for disk-backed mode (fast NVMe recommended)")
    
    # Monitoring parameters
    parser.add_argument("--monitor-interval", type=int, default=5, help="Interval in seconds for monitoring index building")
    parser.add_argument("--compact", action="store_true", help="Perform compaction after loading")
    
    # Configuration file
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    
    # What-if option to print args and exit
    parser.add_argument("--what-if", action="store_true", help="Print the arguments after processing and exit")
    
    # Debug option to set logging level to DEBUG
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Track which arguments were explicitly set vs using defaults
    args.is_default = {
        'host': args.host == "localhost",
        'port': args.port == "19530",
        'num_shards': args.num_shards == 1,
        'vector_dtype': args.vector_dtype == "float",
        'distribution': args.distribution == "uniform",
        'batch_size': args.batch_size == 10000,
        'chunk_size': args.chunk_size == 1000000,
        'index_type': args.index_type == "DISKANN",
        'metric_type': args.metric_type == "COSINE",
        'max_degree': args.max_degree == 16,
        'search_list_size': args.search_list_size == 200,
        'M': args.M == 16,
        'ef_construction': args.ef_construction == 200,
        'monitor_interval': args.monitor_interval == 5,
        'compact': not args.compact,  # Default is False
        'force': not args.force,  # Default is False
        'what_if': not args.what_if,  # Default is False
        'debug': not args.debug,  # Default is False
        'adaptive': not args.adaptive,  # Default is False
        'memory_budget': args.memory_budget == "0",
        'disk_backed': not args.disk_backed,  # Default is False
        'seed': args.seed is None,
    }
    
    # Set logging level to DEBUG if --debug is specified
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Load configuration from YAML if specified
    if args.config:
        config = load_config(args.config)
        args = merge_config_with_args(config, args)
    
    # If what-if is specified, print the arguments and exit
    if args.what_if:
        logger.info("Running in what-if mode. Printing arguments and exiting.")
        print("\nConfiguration after processing arguments and config file:")
        print("=" * 60)
        for key, value in vars(args).items():
            if key != 'is_default':  # Skip the is_default dictionary
                source = "default" if args.is_default.get(key, False) else "specified"
                print(f"{key}: {value} ({source})")
        print("=" * 60)
        sys.exit(0)
    
    # Validate required parameters
    required_params = ['collection_name', 'dimension', 'num_vectors']
    missing_params = [param for param in required_params if getattr(args, param.replace('-', '_'), None) is None]
    
    if missing_params:
        parser.error(f"Missing required parameters: {', '.join(missing_params)}. "
                     f"Specify with command line arguments or in config file.")
    
    return args


# =============================================================================
# Milvus Connection and Collection Management
# =============================================================================

def connect_to_milvus(host, port):
    """Connect to Milvus server"""
    try:
        logger.debug(f"Connecting to Milvus server at {host}:{port}")
        connections.connect(
            "default", 
            host=host, 
            port=port,
            max_receive_message_length=514_983_574,
            max_send_message_length=514_983_574
        )
        logger.info(f"Connected to Milvus server at {host}:{port}")
        return True

    except Exception as e:
        logger.error(f"Error connecting to Milvus server: {str(e)}")
        return False


def create_collection(collection_name, dim, num_shards, vector_dtype, force=False):
    """Create a new collection with the specified parameters"""
    try:
        # Check if collection exists
        if utility.has_collection(collection_name):
            if force:
                Collection(name=collection_name).drop()
                logger.info(f"Dropped existing collection: {collection_name}")
            else:
                logger.warning(f"Collection '{collection_name}' already exists. Use --force to drop and recreate it.")
                return None

        # Define vector data type
        vector_type = DataType.FLOAT_VECTOR

        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=vector_type, dim=dim)
        ]
        schema = CollectionSchema(fields, description="Benchmark Collection")

        # Create collection
        collection = Collection(name=collection_name, schema=schema, num_shards=num_shards)
        logger.info(f"Created collection '{collection_name}' with {dim} dimensions and {num_shards} shards")

        return collection
    except Exception as e:
        logger.error(f"Failed to create collection: {str(e)}")
        return None


# =============================================================================
# Vector Generation (Enhanced with reproducibility and memory efficiency)
# =============================================================================

def generate_vectors(num_vectors: int, dim: int, distribution: str = 'uniform',
                     seed: Optional[int] = None, batch_index: int = 0) -> np.ndarray:
    """
    Generate random vectors based on the specified distribution.
    
    Args:
        num_vectors: Number of vectors to generate
        dim: Vector dimension
        distribution: 'uniform' or 'normal'
        seed: Optional seed for reproducibility (combined with batch_index)
        batch_index: Batch index for deterministic seeding across batches
    
    Returns:
        numpy array of shape (num_vectors, dim), normalized
    """
    # Use seeded RNG if seed provided (enables reproducibility)
    if seed is not None:
        rng = np.random.default_rng(seed + batch_index)
    else:
        rng = np.random.default_rng()
    
    if distribution == 'uniform':
        vectors = rng.uniform(-1, 1, (num_vectors, dim)).astype(np.float32)
    elif distribution == 'normal':
        vectors = rng.standard_normal((num_vectors, dim)).astype(np.float32)
    elif distribution == 'zipfian':
        # Simplified zipfian-like distribution
        base = rng.uniform(0, 1, (num_vectors, dim)).astype(np.float32)
        skew = rng.zipf(1.5, (num_vectors, 1)).astype(np.float32)
        vectors = base * (skew / 10)
    else:
        vectors = rng.uniform(-1, 1, (num_vectors, dim)).astype(np.float32)

    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    normalized_vectors = vectors / norms

    return normalized_vectors


def generate_vectors_streaming(total_vectors: int, dimension: int,
                               batch_size: int,
                               distribution: str = 'uniform',
                               seed: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Generator that yields vectors in batches without bulk allocation.
    
    This is the memory-efficient version that generates vectors on-demand.
    
    Yields:
        Tuple of (start_id, vectors_array)
    """
    num_batches = (total_vectors + batch_size - 1) // batch_size
    vectors_remaining = total_vectors
    current_id = 0
    
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, vectors_remaining)
        vectors = generate_vectors(current_batch_size, dimension, distribution, seed, batch_idx)
        
        yield current_id, vectors
        
        current_id += current_batch_size
        vectors_remaining -= current_batch_size


# =============================================================================
# Data Insertion Functions
# =============================================================================

def insert_data(collection, vectors, batch_size=10000, start_id=0):
    """Insert vectors into the collection in batches"""
    total_vectors = len(vectors) if isinstance(vectors, (list, np.ndarray)) else vectors.shape[0]
    num_batches = (total_vectors + batch_size - 1) // batch_size

    start_time = time.time()
    total_inserted = 0

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, total_vectors)
        batch_size_actual = batch_end - batch_start

        # Prepare batch data
        ids = list(range(start_id + batch_start, start_id + batch_end))
        if isinstance(vectors, np.ndarray):
            batch_vectors = vectors[batch_start:batch_end].tolist()
        else:
            batch_vectors = vectors[batch_start:batch_end]

        # Insert batch
        try:
            collection.insert([ids, batch_vectors])
            total_inserted += batch_size_actual

            # Log progress
            progress = total_inserted / total_vectors * 100
            elapsed = time.time() - start_time
            rate = total_inserted / elapsed if elapsed > 0 else 0
            mem_info = f", Mem: {get_memory_percent():.1f}%" if PSUTIL_AVAILABLE else ""

            logger.info(f"Inserted batch {i+1}/{num_batches}: {progress:.2f}% complete, "
                        f"rate: {rate:.2f} vectors/sec{mem_info}")

        except Exception as e:
            logger.error(f"Error inserting batch {i+1}: {str(e)}")

    return total_inserted, time.time() - start_time


def insert_data_standard(collection, num_vectors: int, dimension: int, 
                         distribution: str, batch_size: int,
                         seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Standard vector loading - generates and inserts vectors in batches.
    """
    logger.info(f"Loading {num_vectors:,} vectors in batches of {batch_size:,} (standard mode)")
    
    start_time = time.time()
    vectors_loaded = 0
    batch_idx = 0
    total_gen_time = 0.0
    total_insert_time = 0.0
    
    for batch_start in range(0, num_vectors, batch_size):
        batch_end = min(batch_start + batch_size, num_vectors)
        current_batch_size = batch_end - batch_start
        
        # Generate vectors
        gen_start = time.time()
        vectors = generate_vectors(current_batch_size, dimension, distribution, seed, batch_idx)
        total_gen_time += time.time() - gen_start
        
        # Prepare data
        ids = list(range(batch_start, batch_end))
        data = [ids, vectors.tolist()]
        
        # Insert
        insert_start = time.time()
        collection.insert(data)
        total_insert_time += time.time() - insert_start
        
        vectors_loaded += current_batch_size
        batch_idx += 1
        
        # Progress reporting
        if batch_idx % 100 == 0 or vectors_loaded == num_vectors:
            elapsed = time.time() - start_time
            rate = vectors_loaded / elapsed if elapsed > 0 else 0
            progress = (vectors_loaded / num_vectors) * 100
            mem_info = f", Mem: {get_memory_percent():.1f}%" if PSUTIL_AVAILABLE else ""
            logger.info(f"Progress: {vectors_loaded:,}/{num_vectors:,} ({progress:.1f}%) - "
                        f"Rate: {rate:,.0f} vec/s{mem_info}")
        
        # Cleanup
        del vectors, data
        if batch_idx % 50 == 0:
            gc.collect()
    
    total_time = time.time() - start_time
    
    return {
        'vectors_loaded': vectors_loaded,
        'total_time': total_time,
        'generation_time': total_gen_time,
        'insertion_time': total_insert_time,
        'batches': batch_idx,
        'rate': vectors_loaded / total_time if total_time > 0 else 0,
    }


def insert_data_adaptive(collection, num_vectors: int, dimension: int,
                         distribution: str, batch_size: int,
                         memory_budget: int = 0,
                         seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Adaptive vector loading with memory-aware batch sizing.
    
    Monitors memory pressure and adjusts batch sizes dynamically.
    """
    logger.info(f"Loading {num_vectors:,} vectors with adaptive batch sizing")
    if memory_budget > 0:
        logger.info(f"Memory budget: {memory_budget / (1024**3):.1f} GB")
    
    # Initialize adaptive controller
    controller = AdaptiveBatchController(
        initial_batch_size=batch_size,
        min_batch_size=max(100, batch_size // 20),
        max_batch_size=min(100000, batch_size * 5)
    )
    
    start_time = time.time()
    vectors_loaded = 0
    batch_idx = 0
    total_gen_time = 0.0
    total_insert_time = 0.0
    errors = 0
    
    while vectors_loaded < num_vectors:
        current_batch_size = controller.get_batch_size()
        remaining = num_vectors - vectors_loaded
        current_batch_size = min(current_batch_size, remaining)
        
        try:
            # Generate vectors
            gen_start = time.time()
            vectors = generate_vectors(current_batch_size, dimension, distribution, seed, batch_idx)
            total_gen_time += time.time() - gen_start
            
            # Prepare data
            ids = list(range(vectors_loaded, vectors_loaded + current_batch_size))
            data = [ids, vectors.tolist()]
            
            # Insert
            insert_start = time.time()
            collection.insert(data)
            total_insert_time += time.time() - insert_start
            
            vectors_loaded += current_batch_size
            batch_idx += 1
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            errors += 1
            controller.force_scale_down()
            continue
        
        # Progress reporting
        if batch_idx % 100 == 0 or vectors_loaded >= num_vectors:
            elapsed = time.time() - start_time
            rate = vectors_loaded / elapsed if elapsed > 0 else 0
            progress = (vectors_loaded / num_vectors) * 100
            logger.info(f"Progress: {vectors_loaded:,}/{num_vectors:,} ({progress:.1f}%) - "
                        f"Rate: {rate:,.0f} vec/s, Batch: {controller.current_batch_size:,}, "
                        f"Mem: {get_memory_percent():.1f}%")
        
        # Cleanup
        del vectors, data
        if batch_idx % 50 == 0:
            gc.collect()
    
    total_time = time.time() - start_time
    
    return {
        'vectors_loaded': vectors_loaded,
        'total_time': total_time,
        'generation_time': total_gen_time,
        'insertion_time': total_insert_time,
        'batches': batch_idx,
        'rate': vectors_loaded / total_time if total_time > 0 else 0,
        'batch_adjustments': controller.adjustment_count,
        'errors': errors,
    }


def insert_data_disk_backed(collection, num_vectors: int, dimension: int,
                            distribution: str, batch_size: int,
                            temp_dir: Optional[str] = None,
                            seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Disk-backed vector loading for memory-constrained systems.
    
    Two-phase approach:
    1. Generate all vectors to disk (memory-mapped file)
    2. Stream from disk to database
    """
    logger.info(f"Loading {num_vectors:,} vectors using disk-backed buffer")
    
    start_time = time.time()
    
    with DiskBackedBuffer(dimension, num_vectors, temp_dir) as disk_buffer:
        # Phase 1: Generate to disk
        logger.info("Phase 1/2: Generating vectors to disk...")
        gen_start = time.time()
        vectors_generated = 0
        batch_idx = 0
        
        while vectors_generated < num_vectors:
            remaining = num_vectors - vectors_generated
            current_batch_size = min(batch_size, remaining)
            
            vectors = generate_vectors(current_batch_size, dimension, distribution, seed, batch_idx)
            disk_buffer.write_batch(vectors, vectors_generated)
            
            vectors_generated += current_batch_size
            batch_idx += 1
            
            if batch_idx % 100 == 0:
                progress = (vectors_generated / num_vectors) * 100
                logger.info(f"Generation: {vectors_generated:,}/{num_vectors:,} ({progress:.1f}%)")
            
            del vectors
            if batch_idx % 50 == 0:
                gc.collect()
        
        gen_time = time.time() - gen_start
        logger.info(f"Phase 1 complete: {vectors_generated:,} vectors in {gen_time:.1f}s")
        
        # Phase 2: Load from disk to database
        logger.info("Phase 2/2: Loading vectors to database...")
        insert_start = time.time()
        vectors_loaded = 0
        insert_batch_idx = 0
        
        for start_id in range(0, num_vectors, batch_size):
            count = min(batch_size, num_vectors - start_id)
            vectors = disk_buffer.read_batch(start_id, count)
            
            ids = list(range(start_id, start_id + count))
            data = [ids, vectors.tolist()]
            
            collection.insert(data)
            vectors_loaded += count
            insert_batch_idx += 1
            
            if insert_batch_idx % 100 == 0 or vectors_loaded >= num_vectors:
                progress = (vectors_loaded / num_vectors) * 100
                logger.info(f"Loading: {vectors_loaded:,}/{num_vectors:,} ({progress:.1f}%)")
        
        insert_time = time.time() - insert_start
    
    total_time = time.time() - start_time
    
    return {
        'vectors_loaded': vectors_loaded,
        'total_time': total_time,
        'generation_time': gen_time,
        'insertion_time': insert_time,
        'batches': batch_idx + insert_batch_idx,
        'rate': vectors_loaded / total_time if total_time > 0 else 0,
    }


# =============================================================================
# Collection Operations
# =============================================================================

def flush_collection(collection):
    """Flush the collection"""
    flush_start = time.time()
    collection.flush()
    flush_time = time.time() - flush_start
    logger.info(f"Flush completed in {flush_time:.2f} seconds")


def create_index(collection, index_params):
    """Create an index on the collection"""
    try:
        start_time = time.time()
        logger.info(f"Creating index with parameters: {index_params}")
        collection.create_index("vector", index_params)
        index_creation_time = time.time() - start_time
        logger.info(f"Index creation command completed in {index_creation_time:.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Failed to create index: {str(e)}")
        return False


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    args = parse_args()

    # Determine loading mode
    if args.disk_backed:
        mode = "disk-backed"
    elif args.adaptive:
        mode = "adaptive"
    else:
        mode = "standard"
    
    memory_budget = parse_memory_string(args.memory_budget)

    # Print configuration summary
    logger.info("=" * 60)
    logger.info("VDB Benchmark - Vector Loader")
    logger.info("=" * 60)
    logger.info(f"Collection:     {args.collection_name}")
    logger.info(f"Vectors:        {args.num_vectors:,}")
    logger.info(f"Dimension:      {args.dimension}")
    logger.info(f"Distribution:   {args.distribution}")
    logger.info(f"Batch size:     {args.batch_size:,}")
    logger.info(f"Shards:         {args.num_shards}")
    logger.info(f"Mode:           {mode}")
    if args.seed:
        logger.info(f"Seed:           {args.seed}")
    if PSUTIL_AVAILABLE:
        logger.info(f"Available RAM:  {get_available_memory() / (1024**3):.1f} GB")
    logger.info("=" * 60)

    # Connect to Milvus
    if not connect_to_milvus(args.host, args.port):
        logger.error("Failed to connect to Milvus.")
        return 1

    logger.debug(f'Determining datatype for vector representation.')
    # Determine vector data type
    try:
        # Check if FLOAT16 is available in newer versions of pymilvus
        if hasattr(DataType, 'FLOAT16'):
            logger.debug(f'Using FLOAT16 data type for vector representation.")')
            vector_dtype = DataType.FLOAT16 if args.vector_dtype == 'float16' else DataType.FLOAT_VECTOR
        else:
            # Fall back to supported data types
            logger.warning("FLOAT16 data type not available in this version of pymilvus. Using FLOAT_VECTOR instead.")
            vector_dtype = DataType.FLOAT_VECTOR
    except Exception as e:
        logger.warning(f"Error determining vector data type: {str(e)}. Using FLOAT_VECTOR as default.")
        vector_dtype = DataType.FLOAT_VECTOR

    # Create collection
    collection = create_collection(
        collection_name=args.collection_name,
        dim=args.dimension,
        num_shards=args.num_shards,
        vector_dtype=vector_dtype,
        force=args.force
    )

    if collection is None:
        return 1

    # Create index with updated parameters
    index_params = {
        "index_type": args.index_type,
        "metric_type": args.metric_type,
        "params": {}
    }

    # Update only the parameters based on index_type
    if args.index_type == "HNSW":
        index_params["params"] = {
            "M": args.M,
            "efConstruction": args.ef_construction
        }
    elif args.index_type == "DISKANN":
        index_params["params"] = {
            "MaxDegree": args.max_degree,
            "SearchListSize": args.search_list_size
        }
    else:
        raise ValueError(f"Unsupported index_type: {args.index_type}")

    logger.debug(f'Creating index. This should be immediate on an empty collection')
    if not create_index(collection, index_params):
        return 1

    # Load vectors based on mode
    logger.info(f"Starting vector generation and insertion using {mode} mode")
    start_gen_time = time.time()
    
    if mode == "disk-backed":
        result = insert_data_disk_backed(
            collection, args.num_vectors, args.dimension, args.distribution, 
            args.batch_size, temp_dir=args.temp_dir, seed=args.seed
        )
    elif mode == "adaptive":
        result = insert_data_adaptive(
            collection, args.num_vectors, args.dimension, args.distribution,
            args.batch_size, memory_budget=memory_budget, seed=args.seed
        )
    else:
        # Standard mode - use chunk-based approach for large datasets
        if args.num_vectors > args.chunk_size:
            logger.info(f"Large vector count detected. Generating in chunks of {args.chunk_size:,} vectors")
            total_inserted = 0
            remaining = args.num_vectors
            chunks_processed = 0
            total_gen_time = 0.0
            total_insert_time = 0.0
            
            while remaining > 0:
                chunk_size = min(args.chunk_size, remaining)
                logger.info(f"Generating chunk {chunks_processed+1}: {chunk_size:,} vectors")
                chunk_start = time.time()
                chunk_vectors = generate_vectors(chunk_size, args.dimension, args.distribution, 
                                                 args.seed, chunks_processed)
                chunk_gen_time = time.time() - chunk_start
                total_gen_time += chunk_gen_time

                logger.info(f"Generated chunk {chunks_processed+1} ({chunk_size:,} vectors) in {chunk_gen_time:.2f} seconds. "
                            f"Progress: {(args.num_vectors - remaining):,}/{args.num_vectors:,} vectors "
                            f"({(args.num_vectors - remaining) / args.num_vectors * 100:.1f}%)")

                # Insert data
                logger.info(f"Inserting {chunk_size:,} vectors into collection '{args.collection_name}'")
                insert_start = time.time()
                inserted, insert_time = insert_data(collection, chunk_vectors, args.batch_size, 
                                                    start_id=args.num_vectors - remaining)
                total_insert_time += insert_time
                total_inserted += inserted
                logger.info(f"Inserted {inserted:,} vectors in {insert_time:.2f} seconds")

                remaining -= chunk_size
                chunks_processed += 1
                
                # Cleanup after each chunk
                del chunk_vectors
                gc.collect()
            
            result = {
                'vectors_loaded': total_inserted,
                'total_time': time.time() - start_gen_time,
                'generation_time': total_gen_time,
                'insertion_time': total_insert_time,
                'batches': chunks_processed,
                'rate': total_inserted / (time.time() - start_gen_time) if (time.time() - start_gen_time) > 0 else 0,
            }
        else:
            # For smaller vector counts, use the standard insertion function
            result = insert_data_standard(
                collection, args.num_vectors, args.dimension, args.distribution,
                args.batch_size, seed=args.seed
            )

    gen_time = time.time() - start_gen_time
    logger.info(f"Completed loading {result['vectors_loaded']:,} vectors in {gen_time:.2f} seconds")

    flush_collection(collection)

    # Monitor index building
    logger.info(f"Starting to monitor index building progress (checking every {args.monitor_interval} seconds)")
    monitor_progress(args.collection_name, args.monitor_interval, zero_threshold=10)

    if args.compact:
        logger.info(f"Compacting collection '{args.collection_name}'")
        collection.compact()
        monitor_progress(args.collection_name, args.monitor_interval, zero_threshold=30)
        logger.info(f"Collection '{args.collection_name}' compacted successfully.")

    # Summary
    logger.info("=" * 60)
    logger.info("Loading Summary")
    logger.info("=" * 60)
    logger.info(f"Vectors loaded:    {result['vectors_loaded']:,}")
    logger.info(f"Total time:        {result['total_time']:.1f}s")
    logger.info(f"Throughput:        {result['rate']:,.0f} vectors/sec")
    logger.info(f"Generation time:   {result['generation_time']:.1f}s")
    logger.info(f"Insertion time:    {result['insertion_time']:.1f}s")
    logger.info(f"Batches:           {result['batches']:,}")
    if 'batch_adjustments' in result and result['batch_adjustments'] > 0:
        logger.info(f"Batch adjustments: {result['batch_adjustments']}")
    if 'errors' in result and result['errors'] > 0:
        logger.info(f"Errors:            {result['errors']}")
    logger.info("=" * 60)
    
    logger.info("Benchmark completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
